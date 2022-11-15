import torch

from typing import List, Union

from SwissArmyTransformer.generation.autoregressive_sampling import update_mems, get_masks_and_position_ids_default
from SwissArmyTransformer.mpu import vocab_parallel_cross_entropy


def batch_filling_sequence(
        model,
        seqs,
        context_lengths,
        strategy,
        max_memory_length=100000,
        get_masks_and_position_ids=get_masks_and_position_ids_default,
        mems=None,
        **kw_args
        ):
    '''
        seq: [2, 3, 5, ..., -1(to be generated), -1, ...]
        mems: [num_layers, batch_size, len_mems(index), mem_hidden_size]
            cache, should be first mems.shape[1] parts of context_tokens.
            mems are the first-level citizens here, but we don't assume what is memorized.
            input mems are used when multi-phase generation.
    '''
    assert len(seqs.shape) == 2

    # building the initial tokens, attention_mask, and position_ids
    batch_size, context_length = seqs.shape
    seqs, attention_mask, position_ids = get_masks_and_position_ids(seqs)
    tokens = seqs[..., :context_length]
    if attention_mask.dtype != torch.bool:
        attention_mask = attention_mask.type_as(next(model.parameters())) # if fp16
    # initialize generation
    counter = context_length - 1 # Last fixed index is ``counter''
    index = 0 if mems is None else mems.shape[2] # Next forward starting index, also the length of cache.
    num_beams = 1
    # step-by-step generation
    while counter < seqs.shape[1] - 1:
        # Now, we want to generate seq[counter + 1],
        # token[:, index: counter+1] needs forwarding.
        # forward
        tokens = tokens.reshape(batch_size * num_beams, -1)
        mems = mems.reshape(mems.shape[0], batch_size * num_beams, mems.shape[-2], mems.shape[-1]) if mems is not None else None
        logits, *output_per_layers = model(
            tokens[:, index:],
            position_ids[..., index: counter+1],
            attention_mask[..., index: counter+1, :counter+1], # TODO memlen
            mems=mems,
            **kw_args
        )
        mem_kv = [o['mem_kv'] for o in output_per_layers]
        mems = update_mems(mem_kv, mems, max_memory_length=max_memory_length)
        if counter == context_length - 1:
            logits = logits[torch.arange(batch_size), context_lengths - 1]
        else:
            logits = logits[:, -1]
        counter += 1
        index = counter
        # if torch.distributed.get_rank() == 0:
        #     print(f"counter: {counter}: logits: {logits.float().abs().mean()}")
        # sampling
        logits = logits.reshape(batch_size, num_beams, -1)
        tokens = tokens.reshape(batch_size, num_beams, -1)
        mems = mems.reshape(mems.shape[0], batch_size, num_beams, mems.shape[-2], mems.shape[-1])
        tokens, mems = strategy.forward(logits, tokens, mems)
        if len(tokens.shape) == 3 and num_beams == 1:
            num_beams = tokens.shape[1]
            position_ids = position_ids.unsqueeze(1).expand(batch_size, num_beams, -1).reshape(batch_size * num_beams, -1)
            attention_mask_shape = attention_mask.shape[-3:]
            attention_mask = attention_mask.unsqueeze(1).expand(batch_size, num_beams, -1, -1, -1).reshape(
                batch_size * num_beams, *attention_mask_shape)
        if strategy.is_done:
            break
    return strategy.finalize(tokens, mems)


class ModelForEvaluation(torch.nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.device = next(self.model.parameters()).device

    @staticmethod
    def process_data(batch, device):
        return (
            batch["tokens"].to(device=device).long(),
            batch["position_ids"].to(device=device).long(),
            batch["attention_mask"].to(device=device).bool().unsqueeze(1),
        )

    def cond_log_prob(self, batch) -> List[List[float]]:
        """
        @return: Conditional log probability of each option
        """
        tokens, position_ids, attention_mask = self.process_data(batch, self.device)
        choices_batch, choice_target_ids_batch = batch["choices"], batch["choice_target_ids"]
        is_single_token = batch["is_single_token"]

        self.model.eval()
        with torch.no_grad():
            logits, *output_per_layers = self.model(tokens, position_ids, attention_mask, log_attention_weights=None)
            logits_batch = torch.nn.functional.log_softmax(logits, dim=-1)

        # output: [b, sq, vocab]
        log_probs = []

        if is_single_token:  # Single token
            for logits, choices, choice_target_ids in zip(logits_batch, choices_batch, choice_target_ids_batch):
                log_probs.append(logits[choice_target_ids[0], choices].tolist())
        else:  # Multi token
            for output, choices, choice_target_ids in zip(logits_batch, choices_batch, choice_target_ids_batch):
                log_probs_single = []
                for choice, choice_target_id in zip(choices, choice_target_ids):
                    tmp = output[choice_target_id, choice]
                    log_probs_single.append(tmp.sum().tolist())
                log_probs.append(log_probs_single)
        return log_probs

    def generate_text(self, sample, strategy, return_all_beams=False) -> Union[
        List[List[int]], List[List[List[int]]]]:
        """
        @return: A list of text model generated, sorted by score in descending order
        """

        seqs = sample["tokens"].to(device=self.device).long()
        context_lengths = sample["context_length"].long()

        def get_masks_and_position_ids(seq):
            batch_size = seq.shape[0]
            max_gen_length = sample['target_position_ids'].shape[-1]
            tokens = torch.nn.functional.pad(seq, (0, max_gen_length), mode='constant', value=-1)
            position_ids = torch.cat((sample['position_ids'], sample['target_position_ids']), dim=-1)
            position_ids = position_ids.to(device=self.device).long()
            attention_mask = sample["attention_mask"].to(device=self.device)
            context_mask = attention_mask[torch.arange(batch_size), context_lengths - 1].unsqueeze(1).repeat(1,
                                                                                                             max_gen_length,
                                                                                                             1)
            causal_mask = torch.tril(context_mask.new_ones((batch_size, max_gen_length, max_gen_length))) < 0.5
            generation_mask = torch.cat(
                (context_mask, causal_mask), dim=-1)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, max_gen_length), mode='constant', value=1)
            attention_mask = torch.cat((attention_mask, generation_mask), dim=1)
            attention_mask = attention_mask.bool().unsqueeze(1)
            return tokens, attention_mask, position_ids

        self.model.eval()
        with torch.no_grad():
            output = batch_filling_sequence(
                self.model,
                seqs,
                context_lengths,
                get_masks_and_position_ids=get_masks_and_position_ids,
                strategy=strategy,
            )[0]

        if isinstance(output, torch.Tensor):  # different strategies
            output = output.tolist()

        output_targets = []
        context_length = seqs.shape[1]
        for lines in output:
            lines = lines.tolist() if isinstance(lines, torch.Tensor) else lines
            output_target = []
            if not isinstance(lines, list):
                lines = [lines]
            for line in lines:
                unfinished = line.index(-1) if -1 in line else len(line)
                if line[unfinished - 1] in strategy.end_tokens:
                    unfinished -= 1
                line = line[context_length:unfinished]
                output_target.append(line)
            if not return_all_beams:
                output_targets.append(output_target[0])
            else:
                output_targets.append(output_target)
        return output_targets


    def calculate_loss(self, batch) -> List[float]:
        tokens, position_ids, attention_mask = self.process_data(batch, self.device)
        targets, loss_masks = (
            batch["targets"].to(device=self.device).long(),
            batch["loss_masks"].to(device=self.device).long(),
        )

        original_parallel_output = self.model.transformer.parallel_output
        self.model.transformer.parallel_output = True
        self.model.eval()

        with torch.no_grad():
            logits, *output_per_layers = self.model(tokens, position_ids, attention_mask, log_attention_weights=None)
            losses = vocab_parallel_cross_entropy(logits.contiguous().float(), targets)
            loss = torch.sum(losses * loss_masks, dim=-1)

        self.model.transformer.parallel_output = original_parallel_output

        return loss.tolist()
