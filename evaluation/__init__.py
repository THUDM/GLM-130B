from .configs import *
from .model import ModelForEvaluation
from .tasks import BaseTask, GenerationTask, MultiChoiceTask, LanguageModelTask
from .dataset import GenerationTaskDataset, MultiChoiceTaskDataset, LanguageModelTaskDataset
from .metrics import qa_evaluate
from .utils import print_rank_0

DEFAULT_CLASS = {
    TaskType.GENERATION: GenerationTask,
    TaskType.MULTICHOICE: MultiChoiceTask,
    TaskType.LANGUAGE_MODEL: LanguageModelTask,
}
