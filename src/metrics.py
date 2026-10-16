"""Evaluation helpers. Thresholds are chosen on validation data, metrics are computed on test."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve


def threshold_max_f1(y_val: np.ndarray, scores: np.ndarray, default: float = 0.5, tolerance: float = 0.01) -> float:
    """Robust F1 threshold chosen on validation labels.

    Validation F1 is often flat near its maximum over a wide range of
    thresholds; the single arg-max can then land on an extreme value
    (e.g. 0.9994) that transfers badly to test. We take the middle of the
    score range whose validation F1 is within `tolerance` of the best.
    """
    y_val = np.asarray(y_val, dtype=int)
    if y_val.sum() == 0 or y_val.sum() == len(y_val):
        return float(default)
    precision, recall, thresholds = precision_recall_curve(y_val, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    near_best = thresholds[f1 >= np.nanmax(f1) - tolerance]
    return float((near_best.min() + near_best.max()) / 2)


def threshold_at_fpr(y_val: np.ndarray, scores: np.ndarray, fpr: float = 0.01) -> float:
    """Smallest threshold whose validation false-positive rate is at most `fpr`."""
    benign = np.asarray(scores)[np.asarray(y_val) == 0]
    if len(benign) == 0:
        return float(np.max(scores))
    return float(np.quantile(benign, 1 - fpr, method="higher")) + 1e-12


def classification_metrics(y_true, score, attack_types=None, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    result = {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "pr_auc": float(average_precision_score(y_true, score)) if 0 < y_true.sum() < len(y_true) else float("nan"),
        "support": {"benign": int((y_true == 0).sum()), "attack": int((y_true == 1).sum())},
    }
    if attack_types is not None:
        types = np.asarray(attack_types, dtype=str)
        # Detection rate per attack type = share of that type's flows flagged as attack.
        result["detection_rate_per_attack"] = {
            name: {"detection_rate": float(pred[types == name].mean()), "support": int((types == name).sum())}
            for name in sorted(set(types) - {"Benign"})
        }
    return result


def evaluate_scores(y_val, s_val, y_test, s_test, attack_test) -> dict:
    """Report test metrics at three operating points chosen without test labels."""
    return {
        "val_f1_threshold": classification_metrics(y_test, s_test, attack_test, threshold_max_f1(y_val, s_val)),
        "val_fpr_1pct_threshold": classification_metrics(y_test, s_test, attack_test, threshold_at_fpr(y_val, s_val, 0.01)),
        "fixed_0.5": classification_metrics(y_test, s_test, attack_test, 0.5),
    }
