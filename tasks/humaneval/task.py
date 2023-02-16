import os
import json
import time
import torch
import torch.distributed as dist

from glob import glob
from dataclasses import dataclass, field
from typing import Union, List
from functools import partial
from tqdm import tqdm
from datetime import datetime

from evaluation import GenerationTask
from evaluation.configs import GenerationTaskConfig
from evaluation.model import ModelForEvaluation
from evaluation.tasks import GenerationTask, GenerationTaskDataset
from evaluation.utils import build_data_loader, gather_result, print_rank_0
from SwissArmyTransformer.tokenization.icetk_glm_130B.ice_tokenizer import _IceTokenizer

from .strategy import CodeBaseStrategy
from .utils import LANGUAGE_TAG, cleanup_code
from .metric import HumanEvalEvaluator

@dataclass
class HumanEvalConfig(GenerationTaskConfig):
    module = "tasks.humaneval.task.HumanEvalTask"
    language: str = 'python'
    num_samples: int = 200
    pass_k: List[int] = field(default_factory=lambda: [1, 10, 100])
    max_gen_length: int = 512
    temperature: float = 0.8
    top_k: int = 200
    top_p: float = 0

class HumanEvalDataset(GenerationTaskDataset):
    config: HumanEvalConfig

    @classmethod
    def config_class(cls):
        return HumanEvalConfig

    def __init__(self, path: Union[str, List[str]], model: ModelForEvaluation, config: HumanEvalConfig):
        language = config.language.lower()
        self.language_prefix = ""
        if language in LANGUAGE_TAG:
            self.language_prefix = LANGUAGE_TAG[language] + '\n'
        super().__init__(path, model, config)

    def process_single_item(self, item, **kwargs):
        item["text"] = self.tokenizer.tokenize(self.language_prefix + item["prompt"].lstrip())
        return [item] * self.config.num_samples
        

class HumanEvalTask(GenerationTask):
    config: HumanEvalConfig

    @classmethod
    def config_class(cls):
        return HumanEvalConfig
    
    @property
    def metrics(self):
        metric_dict = {}
        for k in self.config.pass_k:
            metric_dict[f'pass@{k}'] = (lambda k: (lambda predictions, examples: self.evaluator.evaluate_pass_k(predictions, examples, k)))(k)
        return metric_dict

    def build_dataset(self, relative_path):
        return HumanEvalDataset(os.path.join(self.config.path, relative_path), self.model, self.config)

    def __init__(self, model: ModelForEvaluation, tokenizer: _IceTokenizer, config: HumanEvalConfig):
        super(HumanEvalTask, self).__init__(model, tokenizer, config)

        end_tokens = [tokenizer.get_command("eop"), tokenizer.get_command("eos")]
        if self.config.end_tokens:
            for token in self.config.end_tokens:
                end_tokens.append(self.tokenizer.tokenize(token)[-1])
        
        if self.config.sampling_strategy == "BaseStrategy":
            self.strategy = CodeBaseStrategy(
                language=self.config.language,
                batch_size=self.config.micro_batch_size, 
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                end_tokens=end_tokens
            )
        # elif self.config.sampling_strategy == "BeamSearchStrategy":
        #     self.strategy = CodeBeamSearchStrategy(
        #         language=self.config.language,
        #         batch_size=self.config.micro_batch_size,
        #         num_beams=self.config.num_beams,
        #         length_penalty=self.config.length_penalty,
        #         consider_end=True,
        #         end_tokens=end_tokens,
        #         no_repeat_ngram_size=self.config.no_repeat_ngram_size,
        #         min_gen_length=self.config.min_gen_length,
        #         deterministic=True,  # For evaluation, we need a determined generation strategy
        #     )
        else:
            raise ValueError(f"unknown strategy {self.config.sampling_strategy}")
        
        problem_file = glob(os.path.join(self.config.path, self.config.file_pattern))[0]
        self.evaluator = HumanEvalEvaluator(self.config.language, problem_file, self.tokenizer)
    
    def predict_single_batch(self, batch):        
        outputs_batch: List[List[List[int]]] = self.model.generate_text(batch, self.strategy, return_all_beams=False)
        predictions = []
        for output in outputs_batch:
            text = self.tokenizer.tokenizer.decode(output)
            print_rank_0([text])
            text = cleanup_code(text, self.config.language)
            predictions.append(self.tokenizer.tokenizer.encode(text))
        return predictions

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
                    for batch in tqdm(dataloader):
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

    def save_prediction_to_file(self, file, predictions, data):
        file_name = file.split(".")[0]
        out_file = os.path.join("outputs", self.config.name + "_" + datetime.now().strftime("%m-%d-%H-%M_") + f"{file_name}.jsonl")
        print(f"Writing results to {out_file}...")
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        out_file = os.path.expanduser(out_file)
        with open(out_file, 'w') as fp:
            for i, sample in enumerate(tqdm(data)):
                task_id = sample["task_id"]
                result = self.evaluator.results[task_id].pop(0)
                sample["result"] = result[1]["result"]
                sample["passed"] = result[1]["passed"]
                sample["completion"] = self.tokenizer.tokenizer.decode(predictions[i])
                if "text" in sample:
                    sample.pop("text")
                fp.write(json.dumps(sample) + '\n')
