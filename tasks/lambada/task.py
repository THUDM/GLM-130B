from string import punctuation
from functools import partial
from typing import List

from evaluation import qa_evaluate, GenerationTask

from .strategy import BeamSearchStrategyForLAMBADA


def exact_match_score(prediction, ground_truth):
    return prediction.strip() == ground_truth.strip()


class LAMBADA(GenerationTask):
    @property
    def metrics(self):
        return {"Accuracy": partial(qa_evaluate, metric=exact_match_score)}

    def __init__(self, model, tokenizer, config_path):
        super(LAMBADA, self).__init__(model, tokenizer, config_path)

        if self.config.sampling_strategy == "BeamSearchStrategy":
            banned_prefix = [[46010], [146337]]  # "'" and "``"
            invalid_slices = [20068, 146010, 146337]
            for p in punctuation:
                pp = tokenizer.tokenize(p)
                if len(pp) == 1:
                    invalid_slices.append(pp[0])
                banned_prefix.append(pp)
            self.strategy = BeamSearchStrategyForLAMBADA(
                batch_size=self.config.micro_batch_size,
                num_beams=self.config.num_beams,
                length_penalty=self.config.length_penalty,
                consider_end=True,
                end_tokens=self.strategy.end_tokens,
                invalid_slices=invalid_slices,
                banned_prefix=banned_prefix,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                min_gen_length=self.config.min_gen_length,
                deterministic=True,
            )

    def get_first_word_tokens(self, tokens):
        text = self.tokenizer.tokenizer.decode(tokens).strip()
        return self.tokenizer.tokenize(text.split(" ")[0])

    def predict_single_batch(self, batch):
        outputs_batch: List[List[List[int]]] = self.model.generate_text(batch, self.strategy, return_all_beams=True)
        predictions = []
        for outputs in outputs_batch:
            found = False
            for output in outputs:
                text = self.tokenizer.tokenizer.decode(output).strip()
                spl = text.split(" ")
                if len(spl) >= 2 and spl[1] in punctuation:
                    predictions.append(self.get_first_word_tokens(output))
                    found = True
                    break
            if not found:
                predictions.append(self.get_first_word_tokens(outputs[0]))
        return predictions
