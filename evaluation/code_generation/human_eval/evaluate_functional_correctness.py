import fire
import sys

from data import HUMAN_EVAL
from evaluation import evaluate_functional_correctness


def entry_point(
        sample_file: str,
        k: str = "1,10,100",
        n_workers: int = 60,
        timeout: float = 10.0,
        problem_file: str = HUMAN_EVAL,
        score_field: str = None,
        exec_check: bool = False,
        task: str = 'humaneval'
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    if isinstance(k, str):
        k = list(map(int, k.split(",")))
    elif isinstance(k, tuple):
        k = list(k)
    results = evaluate_functional_correctness(
        sample_file, k, n_workers, timeout, problem_file, 
        score_field, exec_check, task
    )
    print(results)


def main():
    fire.Fire(entry_point)


sys.exit(main())
