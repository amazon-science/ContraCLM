import numpy as np
from typing import List, Union


def dcg_at_k(y_true, y_score, k=None, log_base=2):
    """Compute Discounted Cumulative Gain.
    Sum the true scores ranked in the order induced by the predicted scores,
    after applying a logarithmic discount.
    This ranking metric yields a high value if true labels are ranked high by
    ``y_score``.
    Parameters
    ----------
    y_true : ndarray of shape (n_samples, n_labels)
        True targets of multilabel classification, or true scores of entities
        to be ranked.
    y_score : ndarray of shape (n_samples, n_labels)
        Target scores, can either be probability estimates, confidence values,
        or non-thresholded measure of decisions (as returned by
        "decision_function" on some classifiers).
    k : int, default=None
        Only consider the highest k scores in the ranking. If `None`, use all
        outputs.
    log_base : float, default=2
        Base of the logarithm used for the discount. A low value means a
        sharper discount (top results are more important).
    Returns
    -------
    discounted_cumulative_gain : ndarray of shape (n_samples,)
        The DCG score for each sample.
    """
    discount = 1 / (np.log(np.arange(y_true.shape[1]) + 2) / np.log(log_base))
    if k is not None:
        discount[k:] = 0

    ranking = np.argsort(y_score)[:, ::-1]
    ranked = y_true[np.arange(ranking.shape[0])[:, np.newaxis], ranking]
    cumulative_gains = discount.dot(ranked.T)

    return cumulative_gains


def map_at_k(
        passed: List[List[int]], k: int
) -> np.ndarray:
    """
    Estimates map@k of each problem and returns the average.
    """

    def estimator(n: Union[List[int], np.ndarray], k: int) -> float:
        avg_p, c = 0.0, 0.0
        for j in range(min(k, len(n))):
            # non-zeo means completion passed partial/all test cases
            if n[j] != 0:
                c += 1
                avg_p += c / (j + 1)
        return avg_p / c if c > 0 else 0.0

    map = 0.0
    for i in range(len(passed)):
        map += estimator(passed[i], k)
    return np.round(map * 100 / len(passed), 2)


def precision_at_k(
        passed: List[List[int]], k: int
) -> np.ndarray:
    """
    Estimates precision@k of each problem and returns the average.
    """
    p_at_k = 0.0
    for i in range(len(passed)):
        c = sum(s != 0 for s in passed[i][:k])
        p_at_k += c / k
    return np.round(p_at_k * 100 / len(passed), 2)


def mrr_at_k(
        passed: List[List[int]], k: int
) -> np.ndarray:
    """
    Estimates mean_reciprocal_rank@k of each problem and returns the average.
    """
    mrr_at_k = 0.0
    for i in range(len(passed)):
        for j, is_passed in enumerate(passed[i][:k]):
            if is_passed != 0:
                mrr_at_k += 1.0 / (j + 1)
                break
    return np.round(mrr_at_k * 100 / len(passed), 2)
