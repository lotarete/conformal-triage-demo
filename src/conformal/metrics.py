"""
Evaluation metrics for conformal prediction on 4-class triage.

Adapted from vet demo metrics.py. Changed from binary to multi-class.
Reuses: ECE, reliability diagrams, set size distribution.
New: per-triage-class coverage, severity-weighted metrics.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .engine import TRIAGE_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS


def empirical_coverage(
    prediction_sets: Sequence[frozenset],
    true_labels: np.ndarray,
) -> float:
    """
    Empirical conformal coverage: fraction of examples where the true
    label is in the prediction set.

    Should be >= 1 - alpha for valid conformal prediction.
    """
    labels = np.asarray(true_labels)
    if labels.dtype.kind not in ("U", "S", "O"):
        labels = np.array([IDX_TO_CLASS[int(l)] for l in labels])

    if len(labels) == 0:
        return 0.0

    covered = sum(1 for s, l in zip(prediction_sets, labels) if str(l) in s)
    return covered / len(labels)


def expected_calibration_error(
    confidence: np.ndarray, is_correct: np.ndarray, n_bins: int = 10
) -> float:
    """
    Expected Calibration Error (ECE).

    For 4-class: confidence = max softmax prob, is_correct = (argmax == true).
    ECE close to 0 = well-calibrated. ECE > 0.1 = poorly calibrated.

    Reused directly from vet demo.
    """
    confidence = np.asarray(confidence, dtype=float)
    is_correct = np.asarray(is_correct, dtype=bool).astype(float)

    if len(confidence) == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(confidence)

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i < n_bins - 1:
            mask = (confidence >= lower) & (confidence < upper)
        else:
            mask = (confidence >= lower) & (confidence <= upper)

        if np.sum(mask) == 0:
            continue

        bin_accuracy = np.mean(is_correct[mask])
        bin_confidence = np.mean(confidence[mask])
        bin_size = np.sum(mask)

        ece += np.abs(bin_accuracy - bin_confidence) * (bin_size / total_samples)

    return float(ece)


def set_size_distribution(test_df: pd.DataFrame) -> dict:
    """
    Distribution of prediction set sizes (1-4 for 4-class triage).

    Size 1 = confident singleton, Size 2 = ambiguous between two levels,
    Size 3-4 = high uncertainty, needs human review.
    """
    if len(test_df) == 0:
        return {}
    return dict(test_df["set_size"].value_counts().sort_index())


def singleton_rate(test_df: pd.DataFrame) -> float:
    """Fraction of predictions with exactly one class in the set."""
    if len(test_df) == 0:
        return 0.0
    return float((test_df["set_size"] == 1).mean())


def coverage_by_triage_class(
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-class coverage breakdown.

    Shows whether CP maintains coverage across all four triage levels,
    or if it's weaker on certain classes (e.g., misses emergency cases).
    """
    if "true_class" not in test_df.columns or "contains_true" not in test_df.columns:
        return pd.DataFrame()

    rows = []
    for cls in TRIAGE_CLASSES:
        mask = test_df["true_class"] == cls
        if mask.sum() == 0:
            continue
        subset = test_df[mask]
        rows.append({
            "triage_class": cls,
            "n_examples": int(mask.sum()),
            "coverage": float(subset["contains_true"].mean()),
            "accuracy": float(subset["is_correct"].mean()) if "is_correct" in subset.columns else None,
            "avg_set_size": float(subset["set_size"].mean()),
            "singleton_rate": float((subset["set_size"] == 1).mean()),
            "mean_max_prob": float(subset["max_prob"].mean()),
        })

    return pd.DataFrame(rows)


def confidence_separation(
    max_probs: np.ndarray, is_correct: np.ndarray
) -> dict:
    """
    Measure confidence separation between correct and incorrect predictions.

    THIS IS THE CRITICAL METRIC that killed the vet demo (only 1.5-2%).
    We need >= 8% separation for a viable demo.

    Args:
        max_probs: Array of max softmax probabilities (model confidence).
        is_correct: Boolean array of whether argmax == true class.

    Returns:
        Dict with mean_correct, mean_incorrect, separation, and viable flag.
    """
    max_probs = np.asarray(max_probs, dtype=float)
    is_correct = np.asarray(is_correct, dtype=bool)

    correct_conf = max_probs[is_correct]
    incorrect_conf = max_probs[~is_correct]

    mean_correct = float(np.mean(correct_conf)) if len(correct_conf) > 0 else 0.0
    mean_incorrect = float(np.mean(incorrect_conf)) if len(incorrect_conf) > 0 else 0.0
    separation = mean_correct - mean_incorrect

    return {
        "mean_correct_confidence": mean_correct,
        "mean_incorrect_confidence": mean_incorrect,
        "separation": separation,
        "separation_pct": separation * 100,
        "n_correct": int(is_correct.sum()),
        "n_incorrect": int((~is_correct).sum()),
        "viable": separation >= 0.08,  # >= 8% separation
        "marginal": 0.03 <= separation < 0.08,
        "kill": separation < 0.03,
    }


def reliability_diagram_data(
    confidence: np.ndarray, is_correct: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """
    Compute reliability diagram data: confidence bins vs actual accuracy.

    Reused from vet demo with no changes needed.
    """
    confidence = np.asarray(confidence, dtype=float)
    is_correct = np.asarray(is_correct, dtype=bool).astype(float)

    if len(confidence) == 0:
        return pd.DataFrame()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i < n_bins - 1:
            mask = (confidence >= lower) & (confidence < upper)
        else:
            mask = (confidence >= lower) & (confidence <= upper)

        if np.sum(mask) == 0:
            continue

        rows.append({
            "bin_lower": float(lower),
            "bin_upper": float(upper),
            "bin_center": (lower + upper) / 2.0,
            "n_predictions": int(np.sum(mask)),
            "accuracy": float(np.mean(is_correct[mask])),
            "confidence": float(np.mean(confidence[mask])),
            "calibration_gap": float(abs(np.mean(is_correct[mask]) - np.mean(confidence[mask]))),
        })

    return pd.DataFrame(rows)
