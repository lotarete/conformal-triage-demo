"""
Weighted conformal prediction for OOD-aware triage.

Standard CP gives valid coverage on in-distribution data, but fails silently
on OOD inputs (still gives tight prediction sets even when it shouldn't).

Weighted CP uses embedding-based distances to re-weight calibration points
per test input, producing wider prediction sets when the input is far from
the calibration distribution.

Formulation follows Tibshirani et al. (2019) "Conformal Prediction Under
Covariate Shift": each calibration point i gets weight w_i proportional to
how relevant it is for the current test point. The test point itself gets
a fixed weight and is placed at score = +infinity (worst case).

For OOD test points: all calibration weights are small (no nearby calibration
data) → low effective sample size → q_hat increases → wider prediction sets.

For ID test points: nearby calibration points have high weight → acts like
standard CP with a focused calibration subset.

This is the novel part of the demo — the "visual punchline."
"""

from typing import Dict, List, Optional

import numpy as np

from .engine import TRIAGE_CLASSES, IDX_TO_CLASS


def compute_weighted_qhat(
    cal_scores: np.ndarray,
    cal_distances: np.ndarray,
    test_distance: float,
    alpha: float = 0.10,
    beta: float = 1.0,
) -> float:
    """
    Compute weighted conformal quantile for a single test point.

    Uses inverse-distance weights so that calibration points whose
    k-NN distance is similar to the test point's distance get MORE
    weight, while dissimilar ones get less. The test point itself is
    placed at score = +infinity with a weight that grows with its
    distance from the calibration set.

    Weight formula (soft, avoids exponential blow-up):
        w_i  = 1 / (1 + beta * max(0, d_test - d_cal_i))
        w_test = beta * max(0, d_test / d_ref - 1)

    where d_ref = 90th percentile of calibration distances (anchors
    the "normal" range). For ID test points d_test ≈ d_cal, so
    w_i ≈ 1 and w_test ≈ 0 → standard CP. For OOD test points
    d_test >> d_ref, so w_test dominates → q_hat is pushed toward 1.0
    → wider prediction sets.

    This gives GRADUAL widening (not all-or-nothing) as the test point
    moves further from the calibration distribution.

    Args:
        cal_scores: Nonconformity scores for calibration set, shape (n,).
        cal_distances: k-NN distances for each calibration point, shape (n,).
        test_distance: k-NN distance for the test point (scalar).
        alpha: Target miscoverage rate.
        beta: Sensitivity parameter (higher = more aggressive OOD widening).

    Returns:
        Weighted q_hat threshold for this test point.
    """
    n = len(cal_scores)

    # Reference distance: 90th percentile of calibration k-NN distances.
    # Points below this are "normal"; above it is OOD territory.
    d_ref = np.percentile(cal_distances, 90)
    if d_ref < 1e-8:
        d_ref = np.mean(cal_distances) + 1e-8  # safety

    # Calibration weights: downweight cal points that are much closer
    # than the test point (irrelevant for an OOD test input).
    excess = np.maximum(0.0, test_distance - cal_distances)
    cal_weights = 1.0 / (1.0 + beta * excess / d_ref)

    # Test point weight: how far beyond "normal" is this test input?
    # Placed at score = +infinity (worst-case nonconformity).
    ood_ratio = max(0.0, test_distance / d_ref - 1.0)
    test_weight = beta * ood_ratio  # 0 for ID, grows for OOD

    # Normalize all weights (calibration + test point) to sum to 1
    all_weights = np.append(cal_weights, test_weight)
    total = all_weights.sum()
    if total < 1e-12:
        # Degenerate — fall back to standard CP
        return float(np.quantile(cal_scores, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)))
    all_weights = all_weights / total

    # Sort calibration scores and their normalized weights
    sort_idx = np.argsort(cal_scores)
    sorted_scores = cal_scores[sort_idx]
    sorted_cal_weights = all_weights[:n][sort_idx]

    # Weighted quantile: find smallest score s such that
    # cumulative calibration weight >= 1 - alpha.
    # If we can't reach (1-alpha) with calibration alone, the test
    # point at +infinity absorbs the remainder → q_hat = 1.0.
    cumulative = np.cumsum(sorted_cal_weights)
    target = 1.0 - alpha

    idx = np.searchsorted(cumulative, target)
    if idx >= n:
        return 1.0

    return float(sorted_scores[idx])


def build_weighted_prediction_sets(
    probs: np.ndarray,
    cal_scores: np.ndarray,
    cal_distances: np.ndarray,
    test_distances: np.ndarray,
    alpha: float = 0.10,
    beta: float = 1.0,
) -> List[dict]:
    """
    Build prediction sets using weighted CP for multiple test points.

    Each test point gets its own q_hat based on its embedding distance.
    OOD points get wider sets; ID points get sets similar to standard CP.

    Args:
        probs: Array of shape (n_test, 4) with softmax probabilities.
        cal_scores: Nonconformity scores from calibration set.
        cal_distances: k-NN distances for calibration points.
        test_distances: k-NN distances for test points.
        alpha: Target miscoverage rate.
        beta: Sensitivity parameter.

    Returns:
        List of dicts with prediction_set, set_size, weighted_qhat, etc.
    """
    probs = np.asarray(probs, dtype=float)
    results = []

    for i in range(probs.shape[0]):
        p = probs[i]
        d = test_distances[i]

        # Compute per-point weighted q_hat
        w_qhat = compute_weighted_qhat(
            cal_scores, cal_distances, d, alpha, beta
        )

        threshold = 1.0 - w_qhat

        # Build prediction set
        pred_set = frozenset(
            TRIAGE_CLASSES[j] for j in range(4) if p[j] >= threshold
        )
        if len(pred_set) == 0:
            pred_set = frozenset([IDX_TO_CLASS[int(np.argmax(p))]])

        results.append({
            "predicted_class": IDX_TO_CLASS[int(np.argmax(p))],
            "prediction_set": pred_set,
            "set_size": len(pred_set),
            "weighted_qhat": w_qhat,
            "threshold": threshold,
            "embedding_distance": float(d),
            "max_prob": float(np.max(p)),
        })

    return results


def compare_standard_vs_weighted(
    standard_sets: List[frozenset],
    weighted_sets: List[frozenset],
    distances: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> dict:
    """
    Compare standard CP vs weighted CP on the same test points.

    This produces the data for the hero scatterplot:
    embedding distance (x) vs set size (y), with two series.

    Args:
        standard_sets: Prediction sets from standard CP.
        weighted_sets: Prediction sets from weighted CP.
        distances: Embedding distances for each test point.
        labels: Optional true labels for coverage comparison.

    Returns:
        Dict with comparison metrics and per-point data for plotting.
    """
    std_sizes = np.array([len(s) for s in standard_sets])
    wgt_sizes = np.array([len(s) for s in weighted_sets])
    distances = np.asarray(distances)

    # Overall comparison
    result = {
        "standard_avg_set_size": float(np.mean(std_sizes)),
        "weighted_avg_set_size": float(np.mean(wgt_sizes)),
        "size_increase": float(np.mean(wgt_sizes) - np.mean(std_sizes)),
        "n_examples": len(standard_sets),
        # Per-point data for scatterplot
        "per_point": {
            "distances": distances.tolist(),
            "standard_sizes": std_sizes.tolist(),
            "weighted_sizes": wgt_sizes.tolist(),
        },
    }

    if labels is not None:
        from .engine import IDX_TO_CLASS
        labs = np.asarray(labels)
        if labs.dtype.kind not in ("U", "S", "O"):
            labs = np.array([IDX_TO_CLASS[int(l)] for l in labs])

        std_cov = sum(1 for s, l in zip(standard_sets, labs) if str(l) in s) / len(labs)
        wgt_cov = sum(1 for s, l in zip(weighted_sets, labs) if str(l) in s) / len(labs)
        result["standard_coverage"] = std_cov
        result["weighted_coverage"] = wgt_cov

    return result
