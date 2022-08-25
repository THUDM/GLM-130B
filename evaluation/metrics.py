import re
import math
import string
import functools

import numpy as np

from typing import Tuple, List
from collections import Counter

from SwissArmyTransformer import get_tokenizer


def accuracy_metric(predictions, examples):
    count = 0
    num_predictions = max(len(predictions), 1)
    assert len(predictions) == len(examples)
    for prediction, example in zip(predictions, examples):
        count += prediction == example["label"]
    return count * 100.0 / num_predictions


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    if not ground_truths:
        return 0.0
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def qa_evaluate(predictions, examples, metric):
    assert len(examples) == len(predictions)
    tokenizer = get_tokenizer()

    score = 0.0
    for example, prediction in zip(examples, predictions):
        ground_truths = [tokenizer.tokenizer.decode(target) for target in example["targets"]]
        prediction = tokenizer.tokenizer.decode(prediction)
        if ground_truths:
            score += metric_max_over_ground_truths(metric, prediction, ground_truths)
    score = 100.0 * score / len(predictions)
    return score


qa_exact_match = functools.partial(qa_evaluate, metric=exact_match_score)
qa_f1 = functools.partial(qa_evaluate, metric=f1_score)


def calculate_perplexity(loss: List[float], data):
    return math.exp(min(20, np.sum(loss) / data[0]["num_original_tokens"]))


DEFAULT_METRICS = {"EM": qa_exact_match, "F1": qa_f1, "Accuracy": accuracy_metric, "PPL": calculate_perplexity}
