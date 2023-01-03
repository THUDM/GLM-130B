import os
import json
import re
from typing import Union, List, Dict, Callable
from datetime import datetime
from evaluation.model import ModelForEvaluation
from evaluation.tasks import GenerationTask, GenerationTaskDataset, GenerationTaskConfig
from evaluation.utils import print_rank_0
from dataclasses import dataclass


@dataclass
class ChainOfThoughtConfig(GenerationTaskConfig):
    prompt_path: str = None
    chain_of_thought: bool = True
    prompt_type: str = None


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


def build_prompt(examples, task_name, chain_of_thought=True, prompt_type=None):
    prompts = []
    for i, item in enumerate(examples):
        question, answer = item["question"], item["answer"]
        if not chain_of_thought:
            answer = extract_answer(answer, task_name)
        if prompt_type == "number":
            prompts.append(f"{i+1}. Question: {question} Answer: {answer}")
        else:
            prompts.append(f"Question: {question} Answer: {answer}")
    if prompt_type == "return":
        prompt = " <n>".join(prompts)
    else:
        prompt = " ".join(prompts)
    return prompt


def extract_answer(prediction, task_name, chain_of_thought=True):
    if task_name.startswith("gsm8k"):
        prediction = prediction.lower()
        if chain_of_thought:
            pattern = r"(?<=the answer is )\d+"
        else:
            pattern = r"\d+"
        match = re.search(pattern, prediction)
        if match:
            answer = match.group(0)
        else:
            answer = ""
    elif task_name.startswith("sports") or task_name.startswith("coinflip"):
        prediction = prediction.lower()
        if chain_of_thought:
            pattern = r"(?<=the answer is )(yes|no)"
        else:
            pattern = r"yes|no"
        match = re.search(pattern, prediction)
        if match:
            answer = match.group(0)
        else:
            answer = "no"
    elif task_name.startswith("lastletter"):
        prediction = prediction.lower()
        if chain_of_thought:
            pattern = r"(?<=the answer is )[a-z]+"
        else:
            pattern = r"[a-z]+"
        match = re.search(pattern, prediction)
        if match:
            answer = match.group(0)
        else:
            answer = ""
    elif task_name.startswith("reverse"):
        prediction = prediction.lower()
        if chain_of_thought:
            pattern = r'(?<=the answer is ")[a-z|,| ]+'
        else:
            pattern = r"[a-z|,| ]+"
        match = re.search(pattern, prediction)
        if match:
            answer = match.group(0)
        else:
            answer = ""
    elif task_name.startswith("date"):
        prediction = prediction.lower()
        date_regex = r"(((0[0-9])|(1[012]))\/((0[1-9])|([12][0-9])|(3[01]))\/((20[012]\d|19\d\d)|(1\d|2[0123])))"
        if chain_of_thought:
            pattern = r"(?<=the answer is )" + date_regex
        else:
            pattern = date_regex
        match = re.search(pattern, prediction)
        if match:
            answer = match.group(0)
        else:
            answer = ""
    else:
        raise NotImplementedError(task_name)
    return answer


class ChainOfThoughtDataset(GenerationTaskDataset):
    config: ChainOfThoughtConfig

    def __init__(self, path: Union[str, List[str]], model: ModelForEvaluation, config: ChainOfThoughtConfig):
        self.labeled_examples = read_examples(config.prompt_path)
        self.labeled_prompt = build_prompt(
            self.labeled_examples, config.name, chain_of_thought=config.chain_of_thought, prompt_type=config.prompt_type
        )
        # print_rank_0(self.labeled_prompt)
        self.printed_count = 0
        super().__init__(path, model, config)
        # print_rank_0(len(self.tokenizer.tokenize(self.labeled_prompt)))

    def process_single_item(self, item, **kwargs):
        question, targets = item["question"], item["targets"]
        if self.config.prompt_type == "number":
            text = self.labeled_prompt + f" {len(self.labeled_examples) + 1}. Question: {question} Answer:"
        elif self.config.prompt_type == "return":
            text = self.labeled_prompt + f" <n>Question: {question} Answer:"
        else:
            text = self.labeled_prompt + f" Question: {question} Answer:"
        text, targets = self.tokenizer.tokenize(text), self.tokenizer.tokenize(targets)
        if len(text) + self.config.max_gen_length + 2 > self.config.max_seq_length:
            text_length = self.config.max_seq_length - self.config.max_gen_length - 2
            text = text[len(text) - text_length : len(text)]
        # if self.printed_count < 3:
        #     print_rank_0(self.tokenizer.detokenize(text))
        #     self.printed_count += 1
        return [{"text": text, "targets": targets, **kwargs}]


class GSM8KDataset(ChainOfThoughtDataset):
    def process_single_item(self, item, **kwargs):
        item["targets"] = item["answer"].split("####")[1].strip()
        return super().process_single_item(item, **kwargs)


class SportsDataset(ChainOfThoughtDataset):
    def process_single_file(self, path):
        with open(path) as file:
            dataset = json.load(file)
        for item in dataset["examples"]:
            sentence = item["input"]
            item["question"] = f'Is the following sentence plausible? "{sentence}."'
            if item["target_scores"]["plausible"] == 1:
                item["targets"] = "yes"
            else:
                item["targets"] = "no"
            self.data.extend(self.process_single_item(item))


class DateDataset(ChainOfThoughtDataset):
    def process_single_file(self, path):
        with open(path) as file:
            dataset = json.load(file)
        for item in dataset["examples"]:
            sentence = item["input"]
            item["question"] = sentence
            for key, value in item["target_scores"].items():
                if value == 1:
                    item["targets"] = key
            self.data.extend(self.process_single_item(item))


class LastLetterDataset(ChainOfThoughtDataset):
    def process_single_item(self, item, **kwargs):
        first_name, last_name = item["first_name"], item["last_name"]
        question = f'Take the last letters of the words in "{first_name} {last_name}" and concatenate them.'
        item["question"] = question
        return super().process_single_item(item, **kwargs)


class ChainOfThoughtTask(GenerationTask):
    config: ChainOfThoughtConfig

    @classmethod
    def config_class(cls):
        return ChainOfThoughtConfig

    @property
    def metrics(self) -> Dict[str, Callable]:
        return {"Accuracy": self.extracted_accuracy_metric}

    def extracted_accuracy_metric(self, predictions, examples):
        count = 0
        num_predictions = max(len(predictions), 1)
        assert len(predictions) == len(examples)
        for prediction, example in zip(predictions, examples):
            output = self.tokenizer.detokenize(prediction)
            prediction = extract_answer(output, self.config.name, self.config.chain_of_thought).strip()
            target = self.tokenizer.detokenize(example["targets"]).strip()
            count += prediction == target
        return count * 100.0 / num_predictions

    def build_dataset(self, relative_path):
        if self.config.name.startswith("gsm8k"):
            return GSM8KDataset(os.path.join(self.config.path, relative_path), self.model, self.config)
        elif self.config.name.startswith("sports"):
            return SportsDataset(os.path.join(self.config.path, relative_path), self.model, self.config)
        elif self.config.name.startswith("lastletter"):
            return LastLetterDataset(os.path.join(self.config.path, relative_path), self.model, self.config)
        elif self.config.name.startswith("coinflip") or self.config.name.startswith("reverse"):
            return ChainOfThoughtDataset(os.path.join(self.config.path, relative_path), self.model, self.config)
        elif self.config.name.startswith("date"):
            return DateDataset(os.path.join(self.config.path, relative_path), self.model, self.config)
        else:
            raise NotImplementedError

    def save_prediction_to_file(self, file, predictions, data):
        results = []
        for output, item in zip(predictions, data):
            output = self.tokenizer.detokenize(output)
            prediction = extract_answer(output, self.config.name, self.config.chain_of_thought)
            target = self.tokenizer.detokenize(item["targets"])
            results.append({"output": output, "prediction": prediction, "answer": target})
        file_name = file.split(".")[0]
        if not os.path.exists("outputs"):
            os.mkdir("outputs")
        with open(
            "outputs/" + self.config.name + "_" + datetime.now().strftime("%m-%d-%H-%M_") + file_name + ".json", "w"
        ) as output:
            for result in results:
                output.write(json.dumps(result) + "\n")
