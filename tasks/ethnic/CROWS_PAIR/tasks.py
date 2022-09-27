import os
import json
import numpy as np
import torch
from typing import List
from dataclasses import dataclass, field
from evaluation import (
    BaseConfig,
    MultiChoiceTask,
    BaseTask,
    MultiChoiceTaskConfig,
)
from abc import ABC
from os.path import join
from evaluation.utils import print_rank_0
import torch
import time
import numpy as np
import torch.distributed as dist
from typing import Dict, Tuple, List
from abc import ABC
from collections import defaultdict
from typing import List
import torch
from evaluation.configs import (
    BaseConfig,
    GenerationTaskConfig,
    MultiChoiceTaskConfig,
    LanguageModelTaskConfig,
)
from evaluation.dataset import (
    EvaluationDataset,
    GenerationTaskDataset,
    MultiChoiceTaskDataset,
    LanguageModelTaskDataset,
)
from evaluation.utils import (
    build_data_loader,
    gather_result,
    print_rank_0,
    get_tokenized_input,
)
from evaluation.metrics import DEFAULT_METRICS


@dataclass
class CrowsPairTaskConfig(BaseConfig):
    module = "tasks.ethnic.CROWS_PAIR.tasks.CrowsPairTask"
    metrics: List[str] = field(default_factory=lambda: ["CP"])


class CrowsPairTask(BaseTask, ABC):
    config: CrowsPairTaskConfig

    @classmethod
    def config_class(cls):
        return CrowsPairTaskConfig

    def build_dataset(self, relative_path):
        return CrowsPairDataset(join(self.config.path, relative_path), self.config)

    def predict_single_batch(self, batch) -> List[int]:
        log_probs = self.model.cond_log_prob(batch)
        return log_probs

    def CrowsPairMetric(self, predictions, examples):
        print_rank_0("Special metric for CrowsPair")
        results = defaultdict(float)
        labels = defaultdict()
        for prediction, example in zip(predictions, examples):
            prediction = prediction[0]
            if example["sent_ID"] == 1:
                results[example["pair_ID"]] = results[example["pair_ID"]] + prediction
            else:
                results[example["pair_ID"]] = results[example["pair_ID"]] - prediction
            labels[example["pair_ID"]] = example["bias_type"]
        cat_postivie = defaultdict(int)
        cat_tt = defaultdict(int)
        final = defaultdict(int)
        for val1, val2 in zip(results.values(), labels.values()):
            if val1 >= 0:
                cat_postivie[val2] = cat_postivie[val2] + 1
            else:
                cat_postivie[val2] = cat_postivie[val2]
            cat_tt[val2] = cat_tt[val2] + 1
        for key, val in cat_postivie.items():
            final[key] = val / cat_tt[key]
        return final

    def report_single_metrics(self, file: str, result_dict: Dict[str, float]):
        pass

    @property
    def metrics(self):
        return {"CP": self.CrowsPairMetric}

    """
    def evaluate(self):
        dist.barrier()
        start = time.time()
        print_rank_0("\n")
        print_rank_0(f"{self.config}")
        print_rank_0(f"Evaluating task {self.config.name}:")

        result_dict_all = {}

        for group_name, filelist in self.file_groups.items():
            print_rank_0(f"    Evaluating group {group_name}:")

            result_dict_group = {}
            for file in filelist:
                dataset = self.build_dataset(file)
                dataloader = build_data_loader(
                    dataset,
                    micro_batch_size=self.config.micro_batch_size,
                    num_workers=1,
                    drop_last=False,
                    collate_fn=dataset.collate_fn if dataset.has_collate_fn else None,
                )

                prediction = []
                with torch.no_grad():
                    for _, batch in enumerate(dataloader):
                        prediction.append(self.predict_single_batch(batch))

                prediction = gather_result(prediction, len(dataset), self.config.micro_batch_size)
                result_list = self.CrowsPairMetric(prediction, dataset.data)
                result_dict_group = (result_list, len(dataset))

            result_dict_all[group_name] = result_dict_group

        print_rank_0(f"Evaluation results of task {self.config.name}:")
        if self.verbose:
            for group_name, result_dict_group in result_dict_all.items():
                self.report_group_metrics(group_name, result_dict_group)

        print_rank_0(f"Finish task {self.config.name} in {time.time() - start:.1f}s.")"""

    def report_group_metrics(self, group_name, result_dict_group: Dict[str, Tuple[Dict[str, float], int]], level=1):
        for result in result_dict_group.values():
            result = result[0]
            for value1 in result.items():
                value1 = value1[1]
                for key, value in value1.items():
                    print_rank_0("category:{cat}  score:{score}".format(cat=key, score=value * 100))


class CrowsPairDataset(MultiChoiceTaskDataset):

    config: MultiChoiceTaskConfig

    def __init__(self, path, config: MultiChoiceTaskConfig):
        self.is_single_token = True  # set to False later in process_single_item func
        self.eval_data = []
        super().__init__(path, config)

    def process_single_item(self, item):
        text, choices, label = (
            get_tokenized_input(item, "inputs"),
            get_tokenized_input(item, "choices"),
            item["label"],
        )
        # "ID":example.ID,"bias_type":example.bias_type,"goal_label":goal_label
        pair_ID, sent_ID, bias_type = (
            item["pair_ID"],
            item["sent_ID"],
            item["bias_type"],
        )
        tgt_seq_length = sum([len(choice) for choice in choices])
        if tgt_seq_length == len(choices):
            # For single token, we only insert one [sop]
            tgt_seq_length = 1

        assert tgt_seq_length < self.config.max_seq_length
        if len(text) + tgt_seq_length + 2 > self.config.max_seq_length:
            text_length = self.config.max_seq_length - tgt_seq_length - 2
            text = text[len(text) - text_length : len(text)]

        assert not (
            self.mask_id in text and self.config.use_multitask_encoding
        ), "Unified multitask encoding don't support blank filling"

        if tgt_seq_length != 1:
            self.is_single_token = False

        dataset = {
            "text": text,
            "choices": choices,
            "label": label,
            "pair_ID": pair_ID,
            "sent_ID": sent_ID,
            "bias_type": bias_type,
        }

        return dataset
