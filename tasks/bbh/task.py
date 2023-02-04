import os
import re

from datetime import datetime
from functools import partial

from evaluation import qa_evaluate, GenerationTask

def extract_answer(prediction):
    pattern = r"(?<=(the|The) answer is ).*?(?=\.\n)"
    match = re.search(pattern, prediction)
    if match:
        answer = match.group(0)
    else:
        answer = ""
    return answer

def exact_match_score(prediction, ground_truth):
    return extract_answer(prediction) == ground_truth

class BBHGeneration(GenerationTask):
    @property
    def metrics(self):
        return {"Accuracy": partial(qa_evaluate, metric=exact_match_score)}

    def __init__(self, model, tokenizer, config):
        super(BBHGeneration, self).__init__(model, tokenizer, config)
        self.start_time = datetime.now()

    def save_prediction_to_file(self, file, prediction, data):
        filename = os.path.join(f"outputs_{self.start_time}", self.config.name, f"{file}.predict")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as file:
            for item in prediction:
                file.write(str([self.tokenizer.detokenize(item)]) + "\n")