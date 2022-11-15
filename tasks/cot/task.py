import os
import json
import re
from typing import Union, List, Dict, Callable
from datetime import datetime
from evaluation.tasks import GenerationTask, GenerationTaskDataset, GenerationTaskConfig
from evaluation.utils import print_rank_0
from dataclasses import dataclass


@dataclass
class ChainOfThoughtConfig(GenerationTaskConfig):
    prompt_path: str = None


def read_examples(prompt_path):
    examples = []
    item = {"question": None, "answer": None}
    with open(prompt_path) as file:
        for line in file:
            line = line.strip()
            if line.startswith("Q:"):
                question = line[3:]
                item["question"] = question
            elif line.startswith("A:"):
                answer = line[3:]
                item["answer"] = answer
                examples.append(item)
                item = {"question": None, "answer": None}
            else:
                raise NotImplementedError
    return examples


def build_prompt(examples):
    prompts = []
    for item in examples:
        question, answer = item["question"], item["answer"]
        prompts.append(f"Question: {question} Answer: {answer}")
    prompt = " ".join(prompts)
    return prompt


def extract_answer(prediction, task_name):
    if task_name == "gsm8k":
        prediction = prediction.lower()
        match = re.search(r'(?<=the answer is )\d+', prediction)
        if match:
            answer = match.group(0)
        else:
            answer = ""
    else:
        raise NotImplementedError(task_name)
    return answer


class ChainOfThoughtDataset(GenerationTaskDataset):

    def __init__(self, path: Union[str, List[str]], config: ChainOfThoughtConfig):
        self.labeled_examples = read_examples(config.prompt_path)
        self.labeled_prompt = build_prompt(self.labeled_examples)
        print_rank_0(self.labeled_prompt)
        self.printed_count = 0
        super().__init__(path, config)

    def process_single_item(self, item, **kwargs):
        question = item["question"]
        targets = item["answer"].split("####")[1].strip()
        text = self.labeled_prompt + f" Question: {question} Answer:"
        text, targets = self.tokenizer.tokenize(text), self.tokenizer.tokenize(targets)
        if len(text) + self.config.max_gen_length + 2 > self.config.max_seq_length:
            text_length = self.config.max_seq_length - self.config.max_gen_length - 2
            text = text[len(text) - text_length: len(text)]
        if self.printed_count < 3:
            print_rank_0(self.tokenizer.detokenize(text))
            self.printed_count += 1
        return [{"text": text, "targets": targets, **kwargs}]


class ChainOfThoughtTask(GenerationTask):
    config: ChainOfThoughtConfig

    @classmethod
    def config_class(cls):
        return ChainOfThoughtConfig

    @property
    def metrics(self) -> Dict[str, Callable]:
        return {'acuracy': self.extracted_accuracy_metric}

    def extracted_accuracy_metric(self, predictions, examples):
        count = 0
        num_predictions = max(len(predictions), 1)
        assert len(predictions) == len(examples)
        for prediction, example in zip(predictions, examples):
            output = self.tokenizer.detokenize(prediction)
            prediction = extract_answer(output, self.config.name).strip()
            target = self.tokenizer.detokenize(example["targets"]).strip()
            count += prediction == target
        return count * 100.0 / num_predictions

    def build_dataset(self, relative_path, split):
        return ChainOfThoughtDataset(os.path.join(self.config.path, relative_path), self.config)

    def save_prediction_to_file(self, file, predictions, data):
        results = []
        for output, item in zip(predictions, data):
            output = self.tokenizer.detokenize(output)
            prediction = extract_answer(output, self.config.name)
            target = self.tokenizer.detokenize(item["targets"])
            results.append({"output": output, "prediction": prediction, "answer": target})
        file_name = file.split(".")[0]
        if not os.path.exists("outputs"):
            os.mkdir("outputs")
        with open("outputs/" + self.config.name + "_" + datetime.now().strftime(
                '%m-%d-%H-%M_') + file_name + ".json", "w") as output:
            for result in results:
                output.write(json.dumps(result) + "\n")
