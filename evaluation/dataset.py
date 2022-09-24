import os
import math
import json

import numpy as np
import torch

from typing import List, Union
from abc import ABC, abstractmethod
from scipy.linalg import block_diag
from itertools import accumulate
from bisect import bisect_right

from SwissArmyTransformer import get_tokenizer

from .configs import BaseConfig, MultiChoiceTaskConfig, GenerationTaskConfig, LanguageModelTaskConfig
from .utils import get_tokenized_input


def pad_batch(tokens, position_ids, attention_mask, max_seq_length):
    attention_mask = np.pad(
        attention_mask,
        pad_width=((0, max_seq_length - len(tokens)),),
        mode="constant",
        constant_values=0,
    )
    tokens = np.concatenate((tokens, np.zeros(max_seq_length - len(tokens), dtype=np.int64)))
    position_ids = np.concatenate((position_ids, np.zeros(max_seq_length - len(position_ids), dtype=np.int64)))
    return tokens, position_ids, attention_mask


class EvaluationDataset(torch.utils.data.Dataset, ABC):
    """
    Jsonlines of {
        "text": context
        "choices": [choice_id1,...], if not None, len(target) == 1
        "label": If generation task -1, else [0, len(choices))
    }
    If [MASK] not in context, will append [MASK] after text
    """

    def __init__(self, path: Union[str, List[str]], config: BaseConfig):
        self.path = path if isinstance(path, list) else [path]
        self.config = config
        self.max_seq_length = self.config.max_seq_length
        self.dtype = np.int64

        self.tokenizer = get_tokenizer()
        self.mask_id = self.tokenizer.get_command("[MASK]")
        self.gmask_id = self.tokenizer.get_command("[gMASK]")

        self.data = []
        for p in self.path:
            self.process_single_file(p)

    @property
    def has_collate_fn(self) -> bool:
        return False

    def collate_fn(self, samples):
        return None

    def process_single_file(self, path):
        with open(os.path.join(path), "r", encoding="utf-8") as file:
            for line in file:
                item = json.loads(line)
                self.data.append(self.process_single_item(item))

    @abstractmethod
    def process_single_item(self, item) -> dict:
        pass

    def __len__(self):
        return len(self.data)


class GenerationTaskDataset(EvaluationDataset):
    config: GenerationTaskConfig

    def process_single_item(self, item):
        text, targets = get_tokenized_input(item, "inputs"), get_tokenized_input(item, "targets")
        if len(text) + self.config.max_gen_length + 2 > self.config.max_seq_length:
            text_length = self.config.max_seq_length - self.config.max_gen_length - 2
            text = text[len(text) - text_length : len(text)]
        return {"text": text, "targets": targets}

    @property
    def has_collate_fn(self) -> bool:
        return True

    def collate_fn(self, samples):
        TILE = 32
        length_to_pad = (max(map(lambda spl: len(spl["token"]), samples)) + TILE - 1) // TILE * TILE

        token_batch, position_id_batch, attention_mask_batch = [], [], []
        context_length_batch, target_position_id_batch = [], []

        for sample in samples:
            token, position_id, attention_mask = pad_batch(
                sample["token"], sample["position_id"], sample["attention_mask"], length_to_pad
            )
            token_batch.append(token)
            position_id_batch.append(position_id)
            attention_mask_batch.append(attention_mask)
            context_length_batch.append(sample['context_length'])
            target_position_id_batch.append(sample['target_position_id'])
        return {
            "tokens": torch.tensor(np.array(token_batch), dtype=torch.int64),
            "position_ids": torch.tensor(np.array(position_id_batch), dtype=torch.int64),
            "attention_mask": torch.tensor(np.array(attention_mask_batch), dtype=torch.int64) < 0.5,
            "context_length": torch.tensor(context_length_batch, dtype=torch.int64),
            "target_position_ids": torch.tensor(np.array(target_position_id_batch), dtype=torch.int64),
        }

    @staticmethod
    def build_generation_sample(text, max_gen_length, use_task_mask, unidirectional=True):
        tokenizer = get_tokenizer()

        sop_id = tokenizer.get_command("sop")
        mask_id = tokenizer.get_command("[gMASK]") if use_task_mask else tokenizer.get_command("[MASK]")

        token = np.array(text, dtype=np.int64)

        blank_filling = mask_id in text
        if blank_filling:
            assert not unidirectional, "Unidirectional attention doesn't support blank filling"
            assert not use_task_mask, "Unidirectional attention doesn't support task mask"
            mask_position = text.index(mask_id)
            token = np.concatenate((token, [sop_id]))
        else:
            mask_position = len(token)
            if unidirectional:
                token = np.concatenate(([mask_id, sop_id], token))
            else:
                token = np.concatenate((token, [mask_id, sop_id]))
        context_length = len(token)

        position_id = np.arange(0, context_length, dtype=np.int64)
        target_position_id = np.arange(context_length, context_length + max_gen_length, dtype=np.int64)
        if not use_task_mask:
            position_id[context_length - 1:] = mask_position
            target_position_id[:] = mask_position

        attention_mask = np.tril(np.ones((context_length, context_length), dtype=np.int64))
        if not unidirectional:
            attention_mask[: context_length - 1, : context_length - 1] = 1

        item = {
            "token": token,
            "position_id": position_id,
            "target_position_id": target_position_id,
            "attention_mask": attention_mask,
            "context_length": context_length,
        }
        return item

    def __getitem__(self, idx):
        item = self.data[idx]
        sample = self.build_generation_sample(
            item["text"],
            max_gen_length=self.config.max_gen_length,
            use_task_mask=self.config.use_task_mask,
            unidirectional=self.config.unidirectional,
        )
        sample["targets"] = [np.array(target, dtype=self.dtype) for target in item["targets"]]
        return sample


class MultiChoiceTaskDataset(EvaluationDataset):
    config: MultiChoiceTaskConfig

    def __init__(self, path, config: MultiChoiceTaskConfig):
        self.is_single_token = True  # set to False later in process_single_item func
        super().__init__(path, config)

    @property
    def has_collate_fn(self) -> bool:
        return True

    def collate_fn(self, samples):
        TILE = 32
        length_to_pad = (max(map(lambda spl: len(spl["token"]), samples)) + TILE - 1) // TILE * TILE

        token_batch, position_id_batch, attention_mask_batch = [], [], []
        choices_batch, choice_target_ids_batch = [], []

        for sample in samples:
            token, position_id, attention_mask = pad_batch(
                sample["token"], sample["position_id"], sample["attention_mask"], length_to_pad
            )
            token_batch.append(token)
            position_id_batch.append(position_id)
            attention_mask_batch.append(attention_mask)
            choices_batch.append(sample["choices"])
            choice_target_ids_batch.append(sample["choice_target_ids"])

        return {
            "tokens": torch.tensor(np.array(token_batch), dtype=torch.int64),
            "position_ids": torch.tensor(np.array(position_id_batch), dtype=torch.int64),
            "attention_mask": torch.tensor(np.array(attention_mask_batch), dtype=torch.int64) < 0.5,
            "choices": choices_batch,
            "choice_target_ids": choice_target_ids_batch,
            "is_single_token": self.is_single_token,
        }

    def process_single_item(self, item):
        text, choices, label = get_tokenized_input(item, "inputs"), get_tokenized_input(item, "choices"), item["label"]

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

        return {
            "text": text,
            "choices": choices,
            "label": label,
        }

    @staticmethod
    def build_multiple_choice_sample(text, choices, is_single_token, unified_multitask_encoding=False):
        tokenizer = get_tokenizer()

        sop_id = tokenizer.get_command("sop")
        mask_id = tokenizer.get_command("[MASK]")

        token = np.array(text, dtype=np.int64)
        target = np.array(text, dtype=np.int64)
        position_id = np.arange(len(text), dtype=np.int64)
        choice_target_id = []

        blank_filling = mask_id in text
        if not blank_filling:
            mask_position = len(token)
            token = np.concatenate((token, [mask_id]))
            target = np.concatenate((target, [mask_id]))
            position_id = np.concatenate((position_id, [mask_position]))
        else:
            mask_position = text.index(mask_id)

        division = len(token)
        attention_mask = [np.ones((len(token), len(token)), dtype=np.int64)]

        for choice in choices:
            position_id = np.concatenate(
                (
                    position_id,
                    [mask_position] * len(choice)
                    if blank_filling or not unified_multitask_encoding
                    else np.arange(mask_position, mask_position + len(choice), dtype=np.int64),
                )
            )
            choice_target_id.append(np.arange(len(token), len(token) + len(choice), dtype=np.int64))
            attention_mask.append(np.tril(np.ones((len(choice), len(choice)), dtype=np.int64)))
            token = np.concatenate((token, [sop_id], choice[:-1]))
            target = np.concatenate((target, choice))

            if is_single_token:
                break

        attention_mask = block_diag(*attention_mask)
        attention_mask[: len(token), :division] = 1

        if is_single_token:
            choices = np.array(choices, dtype=np.int64).squeeze().tolist()

        item = {
            "token": token,
            "position_id": position_id,
            "attention_mask": attention_mask,
            "choices": choices,
            "choice_target_ids": choice_target_id[0] if is_single_token else choice_target_id,
        }
        return item

    def __getitem__(self, idx):
        item = self.data[idx]
        sample = self.build_multiple_choice_sample(
            item["text"],
            item["choices"],
            is_single_token=self.is_single_token,
            unified_multitask_encoding=self.config.use_multitask_encoding,
        )
        sample["label"] = item["label"]
        return sample


class LanguageModelTaskDataset(EvaluationDataset):
    config: LanguageModelTaskConfig
    left_weights: List[int]
    weights: List[int]

    def process_single_file(self, path):
        num_sequences = []
        with open(os.path.join(path), "r", encoding="utf-8") as file:
            raw_text = file.read()
            tokens = self.tokenizer.tokenize(raw_text)
            self.data.append(
                {
                    "raw_text": tokens,
                    "num_original_tokens": len(raw_text.strip().split(" ")),
                    "num_sequences": max(
                        math.ceil(
                            max(len(tokens) - (self.config.max_seq_length - 1), 0) / self.config.generation_length
                        )
                        + 1,
                        1,
                    ),
                }
            )
            num_sequences.append(self.data[-1]["num_sequences"])
        self.weights = list(accumulate(num_sequences))
        self.left_weights = [0] + self.weights[:-1]

    def process_single_item(self, item):
        pass

    def __len__(self):
        return self.data[0]["num_sequences"]

    def __getitem__(self, idx):
        document_idx = bisect_right(self.weights, idx)
        idx = idx - self.left_weights[document_idx]
        start_idx = idx * self.config.generation_length
        end_idx = start_idx + self.config.max_seq_length - 1  # for additional [gMASK]
        tokens = self.data[document_idx]["raw_text"][start_idx:end_idx]

        mask_id = self.gmask_id if self.config.use_task_mask else self.mask_id
        sop_id = self.tokenizer.get_command("sop")

        if idx == 0 or self.config.unidirectional:
            prompt, text = [], tokens
        else:
            prompt_length = self.config.max_seq_length - 1 - self.config.generation_length
            prompt, text = tokens[:prompt_length], tokens[prompt_length:]

        seq_length = len(prompt) + len(text) + 1
        attention_mask = np.tril(np.ones((seq_length, seq_length), dtype=np.int64))
        attention_mask[: len(prompt) + 1, : len(prompt) + 1] = 1

        return {
            "tokens": np.array(prompt + [mask_id, sop_id] + text[:-1], dtype=np.int64),
            "targets": np.array(prompt + [mask_id] + text, dtype=np.int64),
            "position_ids": np.arange(0, seq_length, dtype=np.int64),
            "attention_mask": attention_mask < 0.5,
            "loss_masks": np.array([0] * (len(prompt) + 1) + [1] * len(text), dtype=np.int64),
        }
