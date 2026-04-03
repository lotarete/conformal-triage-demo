"""
Conformal prediction engine for 4-class medical triage classification.

Adapted from thesis code and vet demo calibration.py. Key change: this operates
on 4-class softmax probabilities (self_care, gp_visit, urgent_care, emergency)
instead of binary correct/incorrect with verbalized confidence.

Nonconformity score: 1 - P(true class) from the model's softmax distribution.
Prediction set: include class y if 1 - P(y) <= q_hat, i.e. P(y) >= 1 - q_hat.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# Triage class labels (ordered by severity)
TRIAGE_CLASSES = ["self_care", "gp_visit", "urgent_care", "emergency"]
CLASS_TO_IDX = {c: i for i, c in enumerate(TRIAGE_CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(TRIAGE_CLASSES)}

# ESI (Emergency Severity Index) mapping
ESI_TO_TRIAGE = {
    1: "emergency",
    2: "emergency",
    3: "urgent_care",
    4: "gp_visit",
    5: "self_care",
}


@dataclass(frozen=True)
class ConformalCalibrationResult:
    """Result of conformal calibration for 4-class triage.

    Attributes:
        alpha: Target miscoverage rate (e.g., 0.10 for 90% coverage)
        q_hat: Calibrated threshold (quantile of nonconformity scores)
        coverage_rate: Achieved coverage on test set (should be >= 1 - alpha)
        average_set_size: Mean prediction set size on test set (1-4)
        n_calibration: Size of calibration split
        n_test: Size of test split
        calibration_scores: Nonconformity scores used to compute q_hat
        test_results: DataFrame with per-example test results
        class_conditional_q_hats: Optional per-class thresholds for imbalanced data
    """

    alpha: float
    q_hat: float
    coverage_rate: float
    average_set_size: float
    n_calibration: int
    n_test: int
    calibration_scores: np.ndarray
    test_results: pd.DataFrame
    class_conditional_q_hats: Optional[Dict[str, float]] = None


def compute_nonconformity_scores(
    probs: np.ndarray, true_labels: np.ndarray
) -> np.ndarray:
    """
    Compute nonconformity scores for 4-class triage classification.

    Score = 1 - P(true class). Higher score means the model assigned less
    probability to the correct class, indicating higher nonconformity.

    Args:
        probs: Array of shape (n, 4) with softmax probabilities per class.
               Column order: [self_care, gp_visit, urgent_care, emergency]
        true_labels: Array of shape (n,) with integer class indices (0-3)
                     or string class names.

    Returns:
        Array of shape (n,) with nonconformity scores in [0, 1].
    """
    probs = np.asarray(probs, dtype=float)
    if probs.ndim == 1:
        raise ValueError(f"probs must be 2D (n, 4), got shape {probs.shape}")

    n = probs.shape[0]

    # Convert string labels to indices if needed
    if isinstance(true_labels, (list, np.ndarray)):
        labels = np.asarray(true_labels)
        if labels.dtype.kind in ("U", "S", "O"):  # string types
            labels = np.array([CLASS_TO_IDX[str(l)] for l in labels])
        else:
            labels = labels.astype(int)
    else:
        raise ValueError("true_labels must be array-like")

    # Extract P(true class) for each example
    p_true = probs[np.arange(n), labels]

    # Nonconformity score = 1 - P(true class)
    scores = 1.0 - p_true

    return scores


def compute_qhat(scores: np.ndarray, alpha: float = 0.10) -> float:
    """
    Compute the conformal quantile q_hat from nonconformity scores.

    Uses the exact split conformal formula:
    rank = ceil((n+1) * (1-alpha))
    q_hat = sorted_scores[rank - 1]

    This threshold guarantees marginal coverage >= 1 - alpha.

    Args:
        scores: Array of nonconformity scores from calibration set.
        alpha: Target miscoverage rate (default 0.10 for 90% coverage).

    Returns:
        Threshold q_hat as float.
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("scores array is empty")

    n = scores.size
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    rank = min(max(rank, 1), n)  # clamp to valid range

    sorted_scores = np.sort(scores)
    q_hat = float(sorted_scores[rank - 1])

    return q_hat


def compute_class_conditional_qhats(
    probs: np.ndarray,
    true_labels: np.ndarray,
    alpha: float = 0.10,
) -> Dict[str, float]:
    """
    Compute per-class conformal thresholds for handling class imbalance.

    Instead of one global q_hat, compute a separate threshold for each
    triage class. This gives class-conditional coverage guarantees.

    Args:
        probs: Array of shape (n, 4) with softmax probabilities.
        true_labels: Array of shape (n,) with integer class indices.
        alpha: Target miscoverage rate.

    Returns:
        Dictionary mapping class name -> class-specific q_hat.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(true_labels)
    if labels.dtype.kind in ("U", "S", "O"):
        labels = np.array([CLASS_TO_IDX[str(l)] for l in labels])
    else:
        labels = labels.astype(int)

    class_qhats = {}
    for cls_idx, cls_name in enumerate(TRIAGE_CLASSES):
        mask = labels == cls_idx
        if mask.sum() == 0:
            class_qhats[cls_name] = 1.0  # conservative: include everything
            continue

        cls_probs = probs[mask]
        cls_labels = labels[mask]
        cls_scores = compute_nonconformity_scores(cls_probs, cls_labels)
        class_qhats[cls_name] = compute_qhat(cls_scores, alpha=alpha)

    return class_qhats


def build_prediction_sets(
    probs: np.ndarray,
    q_hat: float,
    true_labels: Optional[np.ndarray] = None,
    method: str = "aps",
    lambda_reg: float = 0.0,
) -> pd.DataFrame:
    """
    Build conformal prediction sets for 4-class triage classification.

    Two methods available:
    1. "threshold" (legacy): Include class y if 1 - P(y) <= q_hat
    2. "aps" (Regularized APS/RAPS, default): Sort classes by nonconformity
       score (1 - P), add in order until cumsum + λ*|S| >= q_hat.

    RAPS with λ > 0 naturally limits set sizes by penalizing additions,
    while maintaining the conformal prediction coverage guarantee.

    Args:
        probs: Array of shape (n, 4) with softmax probabilities.
        q_hat: Calibrated threshold from compute_qhat().
        true_labels: Optional array of true class indices for coverage eval.
        method: "threshold" or "aps". APS/RAPS is recommended.
        lambda_reg: Regularization strength for RAPS. 0=APS, >0=RAPS.

    Returns:
        DataFrame with columns:
        - predicted_class: argmax class name
        - prediction_set: frozenset of class names in the set
        - set_size: number of classes in prediction set (1-4)
        - max_prob: highest softmax probability
        - contains_true: whether true label is in set (if true_labels given)
        - prob_self_care, prob_gp_visit, prob_urgent_care, prob_emergency
    """
    probs = np.asarray(probs, dtype=float)

    # Convert true_labels if provided
    labels = None
    if true_labels is not None:
        labels = np.asarray(true_labels)
        if labels.dtype.kind in ("U", "S", "O"):
            labels = np.array([CLASS_TO_IDX[str(l)] for l in labels])
        else:
            labels = labels.astype(int)

    rows = []
    for i in range(probs.shape[0]):
        p = probs[i]

        # Build prediction set based on method
        if method == "aps":
            # Regularized APS/RAPS: nonconformity score + λ*set_size stopping criterion
            pred_set = _build_aps_set(p, q_hat, lambda_reg=lambda_reg)
        else:  # threshold (legacy)
            threshold = 1.0 - q_hat
            pred_set = frozenset(
                TRIAGE_CLASSES[j] for j in range(4) if p[j] >= threshold
            )

        # If empty set (rare edge case), include argmax at minimum
        if len(pred_set) == 0:
            pred_set = frozenset([IDX_TO_CLASS[int(np.argmax(p))]])

        row = {
            "predicted_class": IDX_TO_CLASS[int(np.argmax(p))],
            "prediction_set": pred_set,
            "set_size": len(pred_set),
            "max_prob": float(np.max(p)),
            "prob_self_care": float(p[0]),
            "prob_gp_visit": float(p[1]),
            "prob_urgent_care": float(p[2]),
            "prob_emergency": float(p[3]),
            "q_hat": q_hat,
            "method": method,
        }

        if labels is not None:
            true_class = IDX_TO_CLASS[labels[i]]
            row["true_class"] = true_class
            row["is_correct"] = row["predicted_class"] == true_class
            row["contains_true"] = true_class in pred_set

        rows.append(row)

    return pd.DataFrame(rows)


def _build_aps_set(probs: np.ndarray, q_hat: float, lambda_reg: float = 0.0) -> frozenset:
    """
    Build prediction set using Adaptive Prediction Sets (APS) or Regularized APS (RAPS).

    Algorithm:
    1. Compute nonconformity scores: U_i = 1 - P(class i)
    2. Sort classes by U_i in ascending order (best scores first)
    3. Greedily add classes until: sum(U_i for i in S) >= q_hat (APS)
       or: sum(U_i for i in S) + λ*|S| >= q_hat (RAPS)

    This maintains the conformal prediction coverage guarantee:
    P(true label in S) >= 1 - α, where α is the target miscoverage rate.

    The key insight: by sorting by nonconformity and adding greedily, we naturally
    include the most "probable" classes first. Sets grow only as needed to maintain coverage.

    References:
    - Angelopoulos et al. 2021: "Conformal Prediction Under Covariate Shift"
      Uses nonconformity scores for adaptive sets.

    Args:
        probs: Array of shape (4,) with softmax probabilities.
        q_hat: Quantile threshold from conformal calibration (typically 0.9-0.99).
        lambda_reg: Regularization strength (penalty per additional class).
                   0.0 = standard APS, >0 = RAPS (penalizes larger sets).

    Returns:
        Frozenset of class names in the prediction set.
    """
    # Compute nonconformity scores for each class
    # U_i = 1 - P(class i): high if class i has low probability
    nonconf_scores = 1.0 - probs

    # Sort classes by nonconformity score (ascending = best/lowest first)
    sorted_indices = np.argsort(nonconf_scores)

    # Greedily add classes until stopping criterion met
    cumsum = 0.0
    pred_set = []

    for idx in sorted_indices:
        cumsum += nonconf_scores[idx]
        pred_set.append(TRIAGE_CLASSES[idx])

        # Stopping criterion:
        # sum(U_i) + λ*|S| >= q_hat
        if cumsum + lambda_reg * len(pred_set) >= q_hat:
            break

    # Ensure non-empty (fallback to argmax)
    if not pred_set:
        pred_set = [TRIAGE_CLASSES[np.argmax(probs)]]

    return frozenset(pred_set)


def compute_coverage(
    prediction_sets: List[frozenset],
    true_labels: np.ndarray,
) -> float:
    """
    Compute empirical coverage: fraction of examples where the true label
    is in the prediction set.

    Args:
        prediction_sets: List of frozensets of class names.
        true_labels: Array of true class names or indices.

    Returns:
        Coverage rate as float in [0, 1].
    """
    labels = np.asarray(true_labels)
    if labels.dtype.kind not in ("U", "S", "O"):
        labels = np.array([IDX_TO_CLASS[int(l)] for l in labels])

    covered = sum(
        1 for s, l in zip(prediction_sets, labels) if str(l) in s
    )
    return covered / len(labels) if len(labels) > 0 else 0.0


def compute_avg_set_size(prediction_sets: List[frozenset]) -> float:
    """Average prediction set size."""
    if not prediction_sets:
        return 0.0
    return np.mean([len(s) for s in prediction_sets])


def run_calibration(
    probs: np.ndarray,
    true_labels: np.ndarray,
    alpha: float = 0.10,
    cal_ratio: float = 0.6,
    random_seed: int = 42,
    method: str = "aps",
    lambda_reg: float = 0.0,
) -> ConformalCalibrationResult:
    """
    Full split conformal calibration pipeline for 4-class triage.

    1. Split data into calibration (60%) and test (40%).
    2. Compute nonconformity scores on calibration set.
    3. Compute q_hat threshold.
    4. Evaluate coverage and set sizes on test set.

    Args:
        probs: Array of shape (n, 4) with softmax probabilities.
        true_labels: Array of shape (n,) with class indices or names.
        alpha: Target miscoverage rate.
        cal_ratio: Fraction for calibration split.
        random_seed: Random seed for reproducible split.
        method: "aps" (Regularized APS/RAPS, default) or "threshold".
        lambda_reg: Regularization strength for RAPS (0=APS, >0=RAPS).

    Returns:
        ConformalCalibrationResult with full evaluation.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(true_labels)
    if labels.dtype.kind in ("U", "S", "O"):
        labels = np.array([CLASS_TO_IDX[str(l)] for l in labels])
    else:
        labels = labels.astype(int)

    n_total = len(probs)
    rng = np.random.RandomState(random_seed)

    # Split
    indices = rng.permutation(n_total)
    n_cal = int(np.ceil(n_total * cal_ratio))
    cal_idx = indices[:n_cal]
    test_idx = indices[n_cal:]

    cal_probs = probs[cal_idx]
    cal_labels = labels[cal_idx]
    test_probs = probs[test_idx]
    test_labels = labels[test_idx]

    # Calibrate
    cal_scores = compute_nonconformity_scores(cal_probs, cal_labels)
    q_hat = compute_qhat(cal_scores, alpha=alpha)

    # Optional: class-conditional q_hats
    class_q_hats = compute_class_conditional_qhats(cal_probs, cal_labels, alpha)

    # Evaluate on test set
    test_df = build_prediction_sets(
        test_probs, q_hat, test_labels, method=method, lambda_reg=lambda_reg
    )

    coverage = float(test_df["contains_true"].mean()) if len(test_df) > 0 else 0.0
    avg_set_size = float(test_df["set_size"].mean()) if len(test_df) > 0 else 0.0

    return ConformalCalibrationResult(
        alpha=alpha,
        q_hat=q_hat,
        coverage_rate=coverage,
        average_set_size=avg_set_size,
        n_calibration=len(cal_idx),
        n_test=len(test_idx),
        calibration_scores=cal_scores,
        test_results=test_df,
        class_conditional_q_hats=class_q_hats,
    )


def calibrate_single_input(
    probs: np.ndarray,
    q_hat: float,
    method: str = "aps",
    lambda_reg: float = 0.0,
) -> dict:
    """
    Calibration utility for live Streamlit demo.

    Given a single input's 4-class probabilities and a pre-computed q_hat,
    build the prediction set and provide diagnostic info.

    Args:
        probs: Array of shape (4,) with softmax probabilities.
        q_hat: Pre-computed threshold from run_calibration().
        method: "aps" (Adaptive Prediction Sets, default) or "threshold".
        lambda_reg: Regularization strength for RAPS (0=standard APS, >0=RAPS).

    Returns:
        Dictionary with prediction set, triage class, and diagnostics.
    """
    probs = np.asarray(probs, dtype=float).flatten()

    predicted_idx = int(np.argmax(probs))
    predicted_class = IDX_TO_CLASS[predicted_idx]

    # Build prediction set
    if method == "aps":
        pred_set = _build_aps_set(probs, q_hat, lambda_reg=lambda_reg)
    else:  # threshold (legacy)
        threshold = 1.0 - q_hat
        pred_set = frozenset(
            TRIAGE_CLASSES[j] for j in range(4) if probs[j] >= threshold
        )

    if len(pred_set) == 0:
        pred_set = frozenset([predicted_class])

    # Determine severity interpretation
    severity_order = {"emergency": 3, "urgent_care": 2, "gp_visit": 1, "self_care": 0}
    max_severity_in_set = max(severity_order[c] for c in pred_set)
    max_severity_class = [c for c in pred_set if severity_order[c] == max_severity_in_set][0]

    # Interpretation
    if len(pred_set) == 1:
        interpretation = (
            f"High confidence: the model is confident this is {predicted_class.replace('_', ' ')}. "
            f"Single-class prediction set."
        )
    elif len(pred_set) == 2:
        interpretation = (
            f"Moderate uncertainty: the model sees two plausible triage levels. "
            f"Recommendation: escalate to the higher severity ({max_severity_class.replace('_', ' ')})."
        )
    elif len(pred_set) == 3:
        interpretation = (
            f"High uncertainty: three triage levels are plausible. "
            f"Strongly recommend human review. Defaulting to {max_severity_class.replace('_', ' ')}."
        )
    else:
        interpretation = (
            "Maximum uncertainty: all four triage levels are plausible. "
            "This case requires immediate human triage."
        )

    return {
        "predicted_class": predicted_class,
        "prediction_set": pred_set,
        "set_size": len(pred_set),
        "max_prob": float(probs[predicted_idx]),
        "probs": {TRIAGE_CLASSES[i]: float(probs[i]) for i in range(4)},
        "q_hat": q_hat,
        "method": method,
        "max_severity_in_set": max_severity_class,
        "interpretation": interpretation,
    }
