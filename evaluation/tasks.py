import os
import time
import torch
import numpy as np
import torch.distributed as dist

from typing import Dict, Callable, Type, Tuple, List, Any
from abc import ABC, abstractmethod
from glob import glob
from os.path import join, relpath
from collections import defaultdict

from SwissArmyTransformer.tokenization.icetk_glm_130B.ice_tokenizer import _IceTokenizer

from generation import BaseStrategy, BeamSearchStrategy
from .configs import BaseConfig, GenerationTaskConfig, MultiChoiceTaskConfig, LanguageModelTaskConfig
from .model import ModelForEvaluation
from .dataset import EvaluationDataset, GenerationTaskDataset, MultiChoiceTaskDataset, LanguageModelTaskDataset
from .utils import build_data_loader, gather_result, print_rank_0
from .metrics import DEFAULT_METRICS


class BaseTask(ABC):
    model: ModelForEvaluation
    tokenizer: _IceTokenizer
    config: BaseConfig
    file_groups: Dict[str, List[str]]

    @classmethod
    def config_class(cls) -> Type[BaseConfig]:
        return BaseConfig

    @property
    def metrics(self) -> Dict[str, Callable]:
        return {metric: DEFAULT_METRICS[metric] for metric in self.config.metrics}

    def __init__(self, model: ModelForEvaluation, tokenizer: _IceTokenizer, config: BaseConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.config.metrics = list(self.metrics.keys())

        self.file_groups = self.get_file_groups()
        self.verbose = dist.get_rank() == 0
        self.save_prediction = config.save_prediction

    def save_prediction_to_file(self, file, prediction, data):
        pass

    def get_file_groups(self):
        pattern_group = {}
        if isinstance(self.config.file_pattern, str):
            pattern_group["all"] = self.config.file_pattern
        else:
            pattern_group = self.config.file_pattern
        return {
            name: [
                relpath(path, start=self.config.path)
                for path in sorted(glob(join(self.config.path, pattern), recursive=True))
            ]
            for name, pattern in pattern_group.items()
        }

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
                result_dict = {key: metric(prediction, dataset.data) for key, metric in self.metrics.items()}
                result_dict_group[file] = (result_dict, len(dataset))
                if torch.distributed.get_rank() == 0 and self.save_prediction:
                    self.save_prediction_to_file(file, prediction, dataset.data)

                if self.verbose:
                    self.report_single_metrics(file, result_dict)

            result_dict_all[group_name] = result_dict_group

        print_rank_0(f"Evaluation results of task {self.config.name}:")

        if self.verbose:
            for group_name, result_dict_group in result_dict_all.items():
                self.report_group_metrics(group_name, result_dict_group)
            self.report_overall_metrics(
                {k: v for result_dict_group in result_dict_all.values() for k, v in result_dict_group.items()},
            )

        print_rank_0(f"Finish task {self.config.name} in {time.time() - start:.1f}s.")

    def report_single_metrics(self, file: str, result_dict: Dict[str, float]):
        output_str = f"        Finish {file}"
        for key, value in result_dict.items():
            output_str += f", {key} = {value:.3f}"
        print_rank_0(output_str)

    @staticmethod
    def calc_group_metrics(result_dict_group: Dict[str, Tuple[Dict[str, float], int]]):
        metrics_dict = defaultdict(lambda: [])
        weight = []
        for file, (result_dict, length) in result_dict_group.items():
            for key, value in result_dict.items():
                metrics_dict[key].append(value)
            weight.append(length)
        return {
            name: {
                "max": np.max(value),
                "median": np.median(value),
                "average": np.average(value, weights=weight),
            }
            for name, value in metrics_dict.items()
        }

    def report_group_metrics(self, group_name, result_dict_group: Dict[str, Tuple[Dict[str, float], int]], level=1):
        stats_dict = self.calc_group_metrics(result_dict_group)
        if len(stats_dict) == 1:
            name, stats = next(iter(stats_dict.items()))
            print_rank_0(
                "    " * level + f"Group {group_name} {name}: max = {stats['max']:.3f}, "
                f"median = {stats['median']:.3f}, average = {stats['average']:.3f}"
            )
        else:
            print_rank_0("    " * level + f"  Group {group_name}: ")
            for name, stats in stats_dict.items():
                print(
                    "    " * (level + 1) + f"Metric {name}: max = {stats['max']:.3f}, "
                    f"median = {stats['median']:.3f}, average = {stats['average']:.3f}"
                )

    def report_overall_metrics(self, result_dict_all: Dict[str, Tuple[Dict[str, float], int]]):
        pass

    @abstractmethod
    def predict_single_batch(self, batch) -> List[Any]:
        pass

    @abstractmethod
    def build_dataset(self, relative_path: str) -> EvaluationDataset:
        pass


class GenerationTask(BaseTask, ABC):
    config: GenerationTaskConfig

    @classmethod
    def config_class(cls):
        return GenerationTaskConfig

    def build_dataset(self, relative_path):
        return GenerationTaskDataset(join(self.config.path, relative_path), self.model, self.config)

    def save_prediction_to_file(self, file, prediction, data):
        filename = os.path.join("outputs", self.config.name, f"{file}.predict")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as file:
            for item in prediction:
                file.write(self.tokenizer.detokenize(item) + "\n")

    def __init__(self, model: ModelForEvaluation, tokenizer: _IceTokenizer, config: GenerationTaskConfig):
        super(GenerationTask, self).__init__(model, tokenizer, config)

        end_tokens = [tokenizer.get_command("eop"), tokenizer.get_command("eos")]
        if self.config.end_tokens:
            for token in self.config.end_tokens:
                end_tokens.append(self.tokenizer.tokenize(token)[-1])
            print_rank_0(f"End tokens {end_tokens}")
        if self.config.sampling_strategy == "BaseStrategy":
            self.strategy = BaseStrategy(
                batch_size=self.config.micro_batch_size, temperature=1.0, top_k=1, end_tokens=end_tokens
            )
        elif self.config.sampling_strategy == "BeamSearchStrategy":
            self.strategy = BeamSearchStrategy(
                self.config.micro_batch_size,
                self.config.num_beams,
                length_penalty=self.config.length_penalty,
                consider_end=True,
                end_tokens=end_tokens,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                min_gen_length=self.config.min_gen_length,
                deterministic=True,  # For evaluation, we need a determined generation strategy
            )
        else:
            raise ValueError(f"unknown strategy {self.config.sampling_strategy}")

    def predict_single_batch(self, batch) -> List[List[int]]:
        output = self.model.generate_text(batch, self.strategy, return_all_beams=False)
        return output


class MultiChoiceTask(BaseTask, ABC):
    config: MultiChoiceTaskConfig

    @classmethod
    def config_class(cls):
        return MultiChoiceTaskConfig

    def build_dataset(self, relative_path):
        return MultiChoiceTaskDataset(join(self.config.path, relative_path), self.model, self.config)

    def predict_single_batch(self, batch) -> List[int]:
        log_probs = self.model.cond_log_prob(batch)
        return [np.argmax(log_probs_single).item() for log_probs_single in log_probs]


class LanguageModelTask(BaseTask, ABC):
    config: LanguageModelTaskConfig

    @classmethod
    def config_class(cls):
        return LanguageModelTaskConfig

    def build_dataset(self, relative_path):
        return LanguageModelTaskDataset(join(self.config.path, relative_path), self.model, self.config)

    def predict_single_batch(self, batch) -> List[float]:
        return self.model.calculate_loss(batch)
