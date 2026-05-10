"""
Evaluation metrics for link prediction and retrieval.

Supported metrics:
  - ROC-AUC           (sklearn)
  - Average Precision (sklearn)
  - Precision@K       (custom)

Main entry point: evaluate_all()
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


# ── Metric helpers ────────────────────────────────────────────────────────────

def build_label_score_arrays(
    pos_pairs: List[Tuple],
    neg_pairs: List[Tuple],
    scores: Dict[Tuple, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert scored positive/negative pairs into label and prediction arrays.

    Returns
    -------
    y_true  : int array of 0/1 labels  (1 = positive edge)
    y_score : float array of predicted scores
    """
    labels, preds = [], []
    for pair in pos_pairs:
        labels.append(1)
        preds.append(scores.get(pair, 0.0))
    for pair in neg_pairs:
        labels.append(0)
        preds.append(scores.get(pair, 0.0))
    return np.array(labels, dtype=int), np.array(preds, dtype=float)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """
    Precision@K: fraction of the top-K scored items that are true positives.

    Precision@K = |{i ∈ top-K : y_true[i] = 1}| / K
    """
    if k <= 0 or len(y_true) == 0:
        return 0.0
    top_k_idx = np.argsort(y_score)[::-1][:k]
    return float(np.sum(y_true[top_k_idx])) / k


# ── Per-method evaluation ─────────────────────────────────────────────────────

def evaluate_method(
    method_name: str,
    scores: Dict[Tuple, float],
    pos_pairs: List[Tuple],
    neg_pairs: List[Tuple],
    k_values: List[int] = (10, 20),
) -> Dict:
    """
    Evaluate a single scoring method and return a metrics dict.

    Parameters
    ----------
    method_name : label for this method
    scores      : dict mapping (u, v) → predicted score
    pos_pairs   : true positive edges (label = 1)
    neg_pairs   : true negative non-edges (label = 0)
    k_values    : K values for Precision@K

    Returns
    -------
    dict with keys: method, roc_auc, avg_precision, precision@K ...
    """
    y_true, y_score = build_label_score_arrays(pos_pairs, neg_pairs, scores)

    result = {"method": method_name}

    if len(np.unique(y_true)) < 2:
        result["roc_auc"] = None
        result["avg_precision"] = None
        for k in k_values:
            result[f"precision@{k}"] = None
        return result

    result["roc_auc"] = roc_auc_score(y_true, y_score)
    result["avg_precision"] = average_precision_score(y_true, y_score)
    for k in k_values:
        result[f"precision@{k}"] = precision_at_k(y_true, y_score, k)

    return result


# ── Batch evaluation ──────────────────────────────────────────────────────────

def evaluate_all(
    method_scores: Dict[str, Dict[Tuple, float]],
    pos_pairs: List[Tuple],
    neg_pairs: List[Tuple],
    k_values: List[int] = (10, 20),
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluate all methods and return a tidy summary DataFrame.

    Parameters
    ----------
    method_scores : dict  {method_name: {(u,v): score}}
    pos_pairs     : list of true positive test edges
    neg_pairs     : list of true negative test non-edges
    k_values      : K values for Precision@K
    verbose       : print one line per method while running

    Returns
    -------
    pd.DataFrame  with columns: method, roc_auc, avg_precision, precision@K ...
    """
    rows = []
    for name, scores in method_scores.items():
        row = evaluate_method(name, scores, pos_pairs, neg_pairs, k_values)
        rows.append(row)
        if verbose:
            auc = f"{row['roc_auc']:.4f}" if row["roc_auc"] is not None else "N/A"
            ap = (
                f"{row['avg_precision']:.4f}"
                if row["avg_precision"] is not None
                else "N/A"
            )
            pk_str = "  ".join(
                f"P@{k}={row.get(f'precision@{k}', 'N/A'):.4f}"
                if row.get(f"precision@{k}") is not None
                else f"P@{k}=N/A"
                for k in k_values
            )
            print(f"  {name:<32s}  AUC={auc}  AP={ap}  {pk_str}")

    return pd.DataFrame(rows)


def print_results_table(results_df: pd.DataFrame) -> None:
    """Pretty-print the evaluation results DataFrame."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False, float_format="%.4f"))
    print("=" * 80)
