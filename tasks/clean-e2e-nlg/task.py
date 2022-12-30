from string import punctuation
from functools import partial
from typing import List
from SwissArmyTransformer import mpu
import numpy as np
import torch
import os
from tqdm import tqdm

from evaluation import qa_evaluate, GenerationTask
from collections import defaultdict
from typing import Dict, Tuple


from rouge_score import rouge_scorer
from bleurt import score


from evaluation.utils import (
    print_rank_0,
    get_tokenized_input,
)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


class E2E(GenerationTask):
    def __init__(self, model, tokenizer, config_path):
        super(E2E, self).__init__(model, tokenizer, config_path)
        self.bleurt_checkpoint = "BLEURT CHECKPOINT PATH"

    def E2EMetric(self, predictions, examples):
        metrics_dict = defaultdict(lambda: [])
        scorer_rouge = rouge_scorer.RougeScorer(["rouge2", "rougeL"], use_stemmer=True)
        scorer_bleurt = score.BleurtScorer(self.bleurt_checkpoint)
        for text, target in tqdm(zip(predictions, examples)):
            text_de = self.tokenizer.detokenize(text)
            target_de = self.tokenizer.detokenize(target["targets"][0])

            scores_rouge = scorer_rouge.score(text_de, target_de)
            scores_bleurt = scorer_bleurt.score(references=[target_de], candidates=[text_de])
            rouge2_precision = scores_rouge["rouge2"].precision
            rouge2_recall = scores_rouge["rouge2"].recall
            rouge2_fmeasure = scores_rouge["rouge2"].fmeasure
            rougeL_precision = scores_rouge["rougeL"].precision
            rougeL_recall = scores_rouge["rougeL"].recall
            rougeL_fmeasure = scores_rouge["rougeL"].fmeasure
            metrics_dict["rouge2_precision"].append(rouge2_precision)
            metrics_dict["rouge2_recall"].append(rouge2_recall)
            metrics_dict["rouge2_fmeasure"].append(rouge2_fmeasure)
            metrics_dict["rougeL_precision"].append(rougeL_precision)
            metrics_dict["rougeL_recall"].append(rougeL_recall)
            metrics_dict["rougeL_fmeasure"].append(rougeL_fmeasure)
            metrics_dict["bleurt"].append(scores_bleurt[0])

        return metrics_dict

    @property
    def metrics(self):
        return {"e2e": self.E2EMetric}

    def predict_single_batch(self, batch) -> List[List[int]]:
        output = self.model.generate_text(batch, self.strategy, return_all_beams=False)
        return output

    def report_single_metrics(self, file: str, result_dict: Dict[str, float]):
        pass

    def report_group_metrics(
        self,
        group_name,
        result_dict_group: Dict[str, Tuple[Dict[str, float], int]],
        level=1,
    ):
        print("report")
        for tmp1 in result_dict_group.values():
            tmp1 = tmp1[0]
            for result in tmp1.values():
                for key, values in result.items():
                    print_rank_0(key, np.mean(values))
