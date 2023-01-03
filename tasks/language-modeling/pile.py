import os
import math
import json

from typing import *
from os.path import join
from bisect import bisect_right
from itertools import accumulate
from collections import defaultdict

from evaluation import LanguageModelTask, LanguageModelTaskDataset, print_rank_0


def calculate_bpb_score(loss: List[float], data: List[Dict]):
    loss_per_category = defaultdict(lambda: 0.0)
    utf8_length_per_category = defaultdict(lambda: 0.0)
    weights = []
    for item in data:
        weights.append(item["num_sequences"])
        utf8_length_per_category[item["meta"]["pile_set_name"]] += item["utf8_length"]
    weights = list(accumulate(weights))
    for idx in range(len(loss)):
        document_idx = bisect_right(weights, idx)
        loss_per_category[data[document_idx]["meta"]["pile_set_name"]] += loss[idx]
    return {
        name: (loss_per_category[name] / utf8_length_per_category[name] / math.log(2)) for name in loss_per_category
    }


class Pile(LanguageModelTask):
    @property
    def metrics(self) -> Dict[str, Callable]:
        return {"BPB": calculate_bpb_score}

    def build_dataset(self, relative_path):
        return PileDataset(join(self.config.path, relative_path), self.model, self.config)

    def report_single_metrics(self, file: str, result_dict: Dict[str, float]):
        pass

    def report_group_metrics(
        self, group_name, result_dict_group: Dict[str, Tuple[Dict[str, Dict[str, float]], int]], level=1
    ):
        output_str = f"    Finish group {group_name}:\n"
        result = list(result_dict_group.values())[0][0]["BPB"]
        for key, value in result.items():
            output_str += f"        {key} = {value:.3f}\n"
        print_rank_0(output_str)
        pass

    def report_overall_metrics(self, result_dict_all: Dict[str, Tuple[Dict[str, float], int]]):
        pass


class PileDataset(LanguageModelTaskDataset):
    def __len__(self):
        return self.weights[-1]

    def process_single_file(self, path):
        num_sequences = []
        with open(os.path.join(path), "r", encoding="utf-8") as file:
            for line in file:
                item = json.loads(line)
                if len(item["text"]) == 0:
                    continue
                self.data.append(
                    {
                        "raw_text": item["text"],
                        "utf8_length": len(item["text_pretokenized"].encode("utf-8")),
                        "num_sequences": max(
                            math.ceil(
                                max(len(item["text"]) - (self.config.max_seq_length - 1), 0)
                                / self.config.generation_length
                            )
                            + 1,
                            1,
                        ),
                        "meta": item["meta"],
                    }
                )
                num_sequences.append(self.data[-1]["num_sequences"])
            self.weights = list(accumulate(num_sequences))
            self.left_weights = [0] + self.weights[:-1]
