from string import punctuation
from functools import partial
from typing import List
from dataclasses import dataclass, field
from evaluation import BaseConfig,MultiChoiceTask,BaseTask
from abc import ABC
from os.path import join

@dataclass
class StereoSetConfig(BaseConfig):
    module = "tasks.ethnic.tasks.StereoSetTask"
    metrics: List[str] = field(default_factory=lambda: ["SS"])

class StereoSetTask(BaseTask, ABC):
    config: StereoSetConfig

    @classmethod
    def config_class(cls):
        return StereoSetConfig

    def build_dataset(self, relative_path):
        return StereoSetConfig(join(self.config.path, relative_path), self.config)

    def predict_single_batch(self, batch) -> List[int]:
        log_probs = self.model.cond_log_prob(batch)
        return [np.argmax(log_probs_single).item() for log_probs_single in log_probs]


