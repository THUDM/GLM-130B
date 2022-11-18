from __future__ import annotations
from dataclass_wizard import YAMLWizard
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict


class TaskType(Enum):
    MULTICHOICE = "mul"
    GENERATION = "gen"
    LANGUAGE_MODEL = "lm"
    OTHER = "other"


@dataclass
class BaseConfig(YAMLWizard):
    name: str  # Task name
    type: TaskType  # Task type
    path: str  # task data path relative to DATA_PATH

    module: Optional[str] = None  # Custom task module file, optional
    metrics: List[str] = field(default_factory=list)  # Evaluation metrics

    use_task_mask: bool = False  # Whether to use [gMASK] for evaluation
    use_multitask_encoding: bool = False  # Not supported now
    unidirectional: bool = False  # Whether to use unidirectional attention
    max_seq_length: int = 2048  # Max sequence length
    file_pattern: str | Dict[str, str] = "**/*.json*"  # Organize data file in groups
    save_prediction: bool = False

    micro_batch_size: int = 1  # 'gen' task only support mbs = 1 for now

    def __post_init__(self):
        assert self.use_task_mask or not self.unidirectional, "[MASK] doesn't support unidirectional attention"


@dataclass
class MultiChoiceTaskConfig(BaseConfig):
    module = "evaluation.MultiChoiceTask"
    metrics: List[str] = field(default_factory=lambda: ["Accuracy"])


@dataclass
class GenerationTaskConfig(BaseConfig):
    module = "evaluation.GenerationTask"
    metrics: List[str] = field(default_factory=lambda: ["EM", "F1"])
    sampling_strategy: str = "BaseStrategy"
    num_beams: int = 4
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    min_gen_length: int = 0
    max_gen_length: int = 128
    end_tokens: List[str] = field(default_factory=lambda: [])


@dataclass
class LanguageModelTaskConfig(BaseConfig):
    module = "evaluation.LanguageModelTask"
    metrics: List[str] = field(default_factory=lambda: ["PPL"])

    generation_length: int = 256  # Generated length in each window
