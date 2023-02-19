import numpy as np
import torch

from typing import List, Union
from scipy.linalg import block_diag

from SwissArmyTransformer.generation.autoregressive_sampling import update_mems, get_masks_and_position_ids_default
from SwissArmyTransformer.mpu import vocab_parallel_cross_entropy
from SwissArmyTransformer import get_tokenizer


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
    """
    seq: [2, 3, 5, ..., -1(to be generated), -1, ...]
    mems: [num_layers, batch_size, len_mems(index), mem_hidden_size]
        cache, should be first mems.shape[1] parts of context_tokens.
        mems are the first-level citizens here, but we don't assume what is memorized.
        input mems are used when multi-phase generation.
    """
    assert len(seqs.shape) == 2

    # building the initial tokens, attention_mask, and position_ids
    batch_size, context_length = seqs.shape
    seqs, attention_mask, position_ids = get_masks_and_position_ids(seqs)
    tokens = seqs[..., :context_length]
    if attention_mask.dtype != torch.bool:
        attention_mask = attention_mask.type_as(next(model.parameters()))  # if fp16
    # initialize generation
    counter = context_length - 1  # Last fixed index is ``counter''
    index = 0 if mems is None else mems.shape[2]  # Next forward starting index, also the length of cache.
    num_beams = 1
    # step-by-step generation
    while counter < seqs.shape[1] - 1:
        # Now, we want to generate seq[counter + 1],
        # token[:, index: counter+1] needs forwarding.
        # forward
        tokens = tokens.reshape(batch_size * num_beams, -1)
        mems = (
            mems.reshape(mems.shape[0], batch_size * num_beams, mems.shape[-2], mems.shape[-1])
            if mems is not None
            else None
        )
        logits, *output_per_layers = model(
            tokens[:, index:],
            position_ids[..., index : counter + 1],
            attention_mask[..., index : counter + 1, : counter + 1],  # TODO memlen
            mems=mems,
            **kw_args
        )
        mem_kv = [o["mem_kv"] for o in output_per_layers]
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
            position_ids = (
                position_ids.unsqueeze(1)
                .expand((batch_size, num_beams) + position_ids.shape[1:])
                .reshape((batch_size * num_beams,) + position_ids.shape[1:])
            )
            attention_mask_shape = attention_mask.shape[-3:]
            attention_mask = (
                attention_mask.unsqueeze(1)
                .expand(batch_size, num_beams, -1, -1, -1)
                .reshape(batch_size * num_beams, *attention_mask_shape)
            )
        if strategy.is_done:
            break
    return strategy.finalize(tokens, mems)


class ModelForEvaluation(torch.nn.Module):
    def __init__(self, model, position_encoding_2d):
        super().__init__()

        self.model = model
        self.position_encoding_2d = position_encoding_2d
        self.device = next(self.model.parameters()).device

    @staticmethod
    def process_data(batch, device):
        return (
            batch["tokens"].to(device=device).long(),
            batch["position_ids"].to(device=device).long(),
            batch["attention_mask"].to(device=device).bool().unsqueeze(1),
        )

    def build_multiple_choice_sample(
        self,
        text,
        choices,
        is_single_token,
        unified_multitask_encoding=False,
        unidirectional=False,
        use_task_mask=False,
    ):
        tokenizer = get_tokenizer()

        sop_id = tokenizer.get_command("sop")
        mask_id = tokenizer.get_command("[gMASK]") if use_task_mask else tokenizer.get_command("[MASK]")

        token = np.array(text, dtype=np.int64)
        target = np.array(text, dtype=np.int64)
        position_id = np.arange(len(text), dtype=np.int64)
        block_position_id = np.zeros(len(text), dtype=np.int64)
        choice_target_id = []

        blank_filling = mask_id in text
        if not blank_filling:
            if unidirectional:
                assert use_task_mask, "Unidirectional attention only support gMASK"
                token = np.concatenate(([mask_id, sop_id], token[:-1]))
                target = np.concatenate(([mask_id, sop_id], target[:-1]))
                position_id = np.zeros(len(token), dtype=np.int64)
                if self.position_encoding_2d:
                    block_position_id = np.arange(len(token), dtype=np.int64)
                mask_position = len(token)
            else:
                mask_position = len(token)
                token = np.concatenate((token, [mask_id]))
                target = np.concatenate((target, [mask_id]))
                position_id = np.arange(len(token), dtype=np.int64)
                if self.position_encoding_2d:
                    block_position_id = np.zeros(len(token), dtype=np.int64)
        else:
            assert not unidirectional, "Unidirectional attention doesn't support blank filling"
            assert not use_task_mask, "Blank filling only support MASK"
            mask_position = text.index(mask_id)

        division = len(token)
        attention_mask = [np.ones((len(token), len(token)), dtype=np.int64)]
        if unidirectional:
            attention_mask[0] = np.tril(attention_mask[0])

        for choice in choices:
            if not choice:
                choice = [tokenizer.get_command("eop")]

            target = np.concatenate((target, choice))
            choice_target_id.append(np.arange(len(token), len(token) + len(choice), dtype=np.int64))
            attention_mask.append(np.tril(np.ones((len(choice), len(choice)), dtype=np.int64)))

            if unidirectional:
                if self.position_encoding_2d:
                    position_id = np.concatenate((position_id, [0] * len(choice)))
                    block_position_id = np.concatenate(
                        (block_position_id, np.arange(mask_position, mask_position + len(choice), dtype=np.int64))
                    )
                else:
                    position_id = np.concatenate(
                        (
                            position_id,
                            np.arange(mask_position, mask_position + len(choice), dtype=np.int64),
                        )
                    )

                token = np.concatenate((token, [text[-1]], choice[:-1]))
            else:
                if self.position_encoding_2d:
                    position_id = np.concatenate((position_id, [mask_position] * len(choice)))
                    block_position_id = np.concatenate(
                        (block_position_id, np.arange(1, 1 + len(choice), dtype=np.int64))
                    )
                else:
                    position_id = np.concatenate(
                        (
                            position_id,
                            [mask_position] * len(choice)
                            if (blank_filling or not unified_multitask_encoding) and not use_task_mask
                            else np.arange(mask_position, mask_position + len(choice), dtype=np.int64),
                        )
                    )

                token = np.concatenate((token, [sop_id], choice[:-1]))

            if is_single_token:
                break

        attention_mask = block_diag(*attention_mask)
        attention_mask[division:, :division] = 1

        if is_single_token:
            choices = np.array(choices, dtype=np.int64).squeeze().tolist()

        if self.position_encoding_2d:
            position_id = np.stack((position_id, block_position_id), axis=0)

        item = {
            "token": token,
            "position_id": position_id,
            "attention_mask": attention_mask,
            "choices": choices,
            "choice_target_ids": choice_target_id[0] if is_single_token else choice_target_id,
        }
        return item

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

        # if torch.distributed.get_rank() == 0:
        #     import pdb
        #
        #     pdb.set_trace()
        # torch.distributed.barrier()

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

    def build_generation_sample(self, text, max_gen_length, use_task_mask, unidirectional):
        tokenizer = get_tokenizer()

        sop_id = tokenizer.get_command("sop")
        mask_id = tokenizer.get_command("[gMASK]") if use_task_mask else tokenizer.get_command("[MASK]")

        token = np.array(text, dtype=np.int64)
        position_id = np.arange(len(text), dtype=np.int64)
        block_position_id = np.zeros(len(text), dtype=np.int64)
        target_position_id = np.zeros(len(text), dtype=np.int64)
        target_block_position_id = np.zeros(len(text), dtype=np.int64)

        blank_filling = mask_id in text

        if unidirectional:
            assert use_task_mask, "Unidirectional attention only support gMASK"
            assert not blank_filling, "Unidirectional attention doesn't support blank filling"
            token = np.concatenate(([mask_id, sop_id], token))
            if self.position_encoding_2d:
                position_id = np.zeros(len(token), dtype=np.int64)
                target_position_id = np.zeros(max_gen_length, dtype=np.int64)
                block_position_id = np.arange(len(token), dtype=np.int64)
                target_block_position_id = np.arange(len(token), len(token) + max_gen_length, dtype=np.int64)
            else:
                position_id = np.arange(len(token), dtype=np.int64)
                target_position_id = np.arange(len(token), len(token) + max_gen_length, dtype=np.int64)
        else:
            if not blank_filling:
                mask_position = len(token)
                token = np.concatenate((token, [mask_id, sop_id]))
            else:
                assert not use_task_mask, "Blank filling only support MASK"
                mask_position = text.index(mask_id)
                token = np.concatenate((token, [sop_id]))

            position_id = np.concatenate((np.arange(len(token) - 1, dtype=np.int64), [mask_position]))
            target_position_id = np.full(max_gen_length, mask_position, dtype=np.int64)
            if self.position_encoding_2d:
                block_position_id = np.zeros(len(token), dtype=np.int64)
                target_block_position_id = np.arange(1, max_gen_length + 1, dtype=np.int64)
            elif use_task_mask:
                position_id = np.arange(len(token), dtype=np.int64)
                target_position_id = np.arange(len(token), len(token) + max_gen_length, dtype=np.int64)

        context_length = len(token)
        attention_mask = np.tril(np.ones((context_length, context_length), dtype=np.int64))
        if not unidirectional:
            attention_mask[: context_length - 1, : context_length - 1] = 1

        if self.position_encoding_2d:
            position_id = np.stack((position_id, block_position_id), axis=0)
            target_position_id = np.stack((target_position_id, target_block_position_id), axis=0)

        item = {
            "token": token,
            "position_id": position_id,
            "target_position_id": target_position_id,
            "attention_mask": attention_mask,
            "context_length": context_length,
        }
        return item

    def generate_text(self, sample, strategy, return_all_beams=False) -> Union[List[List[int]], List[List[List[int]]]]:
        """
        @return: A list of text model generated, sorted by score in descending order
        """

        seqs = sample["tokens"].to(device=self.device).long()
        context_lengths = sample["context_length"].long()

        def get_masks_and_position_ids(seq):
            batch_size = seq.shape[0]
            max_gen_length = sample["target_position_ids"].shape[-1]
            tokens = torch.nn.functional.pad(seq, (0, max_gen_length), mode="constant", value=-1)
            position_ids = torch.cat((sample["position_ids"], sample["target_position_ids"]), dim=-1)
            position_ids = position_ids.to(device=self.device).long()
            attention_mask = sample["attention_mask"].to(device=self.device)
            context_mask = (
                attention_mask[torch.arange(batch_size), context_lengths - 1].unsqueeze(1).repeat(1, max_gen_length, 1)
            )
            causal_mask = torch.tril(context_mask.new_ones((batch_size, max_gen_length, max_gen_length))) < 0.5
            generation_mask = torch.cat((context_mask, causal_mask), dim=-1)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, max_gen_length), mode="constant", value=1)
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

    def build_language_model_sample(
        self,
        tokens: List[int],
        is_first_segment: bool,
        max_seq_length: int,
        generation_length: int,
        unidirectional: bool,
        use_gmask: bool,
    ):
        tokenizer = get_tokenizer()
        sop_id = tokenizer.get_command("sop")
        mask_id = tokenizer.get_command("[gMASK]") if use_gmask else tokenizer.get_command("[MASK]")

        if is_first_segment or unidirectional:
            prompt, text = [], tokens
        else:
            prompt_length = max_seq_length - 1 - generation_length
            prompt, text = tokens[:prompt_length], tokens[prompt_length:]

        seq_length = len(prompt) + len(text) + 1
        attention_mask = np.tril(np.ones((seq_length, seq_length), dtype=np.int64))
        attention_mask[: len(prompt) + 1, : len(prompt) + 1] = 1

        gen_length = min(len(text), generation_length)

        position_id = np.arange(0, seq_length, dtype=np.int64)
        if self.position_encoding_2d:
            position_id = np.concatenate(
                (np.arange(0, seq_length - gen_length, dtype=np.int64), [seq_length - gen_length - 1] * gen_length)
            )
            block_position_id = np.concatenate(
                ([0] * (seq_length - gen_length - 1), np.arange(0, gen_length + 1, dtype=np.int64))
            )
            position_id = np.stack((position_id, block_position_id), axis=0)

        return {
            "tokens": np.array(prompt + [mask_id, sop_id] + text[:-1], dtype=np.int64),
            "targets": np.array(prompt + [mask_id] + text, dtype=np.int64),
            "position_ids": position_id,
            "attention_mask": attention_mask < 0.5,
            "loss_masks": np.array(
                [0] * (seq_length - gen_length) + [1] * gen_length,
                dtype=np.int64,
            ),
        }

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
