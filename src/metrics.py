from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


@dataclass
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    confusion: list[list[int]]  # JSON-friendly


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
    acc = float(accuracy_score(y_true, y_pred))
    mf1 = float(f1_score(y_true, y_pred, average="macro"))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    return ClassificationMetrics(accuracy=acc, macro_f1=mf1, confusion=cm)