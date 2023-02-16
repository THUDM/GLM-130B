import tqdm
import numpy as np

from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from evaluation.utils import print_rank_0
from .human_eval.data import read_problems
from .human_eval.evaluation import estimate_pass_at_k
from .human_eval.execution import check_correctness

class HumanEvalEvaluator:
    def __init__(
        self, 
        language, 
        problem_file,
        tokenizer,
        n_workers: int = 4,
        timeout: float = 3.0,
    ):
        self.language = language
        self.n_workers = n_workers
        self.timeout = timeout
        self.problems = read_problems(problem_file)
        self.tokenizer = tokenizer
        self.total = None
        self.correct = None
        self.results = {}
    
    def evaluate_pass_k(self, prediction, data, k):
        if self.total is None or self.correct is None or self.results is None:
            self.evaluate_functional_correctness(prediction, data)
        return estimate_pass_at_k(self.total, self.correct, k).mean()

    def evaluate_functional_correctness(self, prediction, data):
        # Check the generated samples against test suites.
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:

            futures = []
            completion_id = Counter()
            n_samples = 0
            results = defaultdict(list)

            print_rank_0("Reading samples...")
            for i, sample in enumerate(tqdm.tqdm(data)):
                task_id = sample["task_id"]
                completion = self.tokenizer.tokenizer.decode(prediction[i])
                args = (self.problems[task_id], completion, self.timeout, completion_id[task_id])
                future = executor.submit(check_correctness, *args)
                futures.append(future)
                completion_id[task_id] += 1
                n_samples += 1

            assert len(completion_id) == len(self.problems), "Some problems are not attempted."

            print_rank_0("Running test suites...")
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                results[result["task_id"]].append((result["completion_id"], result))

        # Calculate pass@k.
        total, correct = [], []
        for result in results.values():
            result.sort()
            passed = [r[1]["passed"] for r in result]
            total.append(len(passed))
            correct.append(sum(passed))
        self.total = np.array(total)
        self.correct = np.array(correct)
        self.results = results
