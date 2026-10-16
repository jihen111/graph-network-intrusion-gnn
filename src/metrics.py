"""Evaluation helpers: all returned classification metrics are test-split metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score,
    confusion_matrix,
)


def classification_metrics(y_true, probability, attack_types=None, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    result = {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "macro_precision": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probability)) if len(np.unique(y_true)) > 1 else float("nan"),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "threshold": float(threshold),
        "support": {"benign": int((y_true == 0).sum()), "attack": int((y_true == 1).sum())},
    }
    if attack_types is not None:
        types = np.asarray(attack_types, dtype=str)
        per_class = {}
        for name in sorted(set(types) - {"Benign", "BENIGN", "benign"}):
            binary = (types == name).astype(int)
            per_class[name] = {
                "precision": float(precision_score(binary, pred, zero_division=0)),
                "recall": float(recall_score(binary, pred, zero_division=0)),
                "f1": float(f1_score(binary, pred, zero_division=0)),
                "support": int(binary.sum()),
            }
        result["per_attack_class"] = per_class
        present = [v["f1"] for v in per_class.values() if v["support"] > 0]
        result["macro_attack_f1"] = float(np.mean(present)) if present else float("nan")
    return result
