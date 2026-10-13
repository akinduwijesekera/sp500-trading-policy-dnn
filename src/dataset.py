from __future__ import annotations

import numpy as np
import pandas as pd

from src.labels import make_labels


def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    lookback: int,
    horizon: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a feature dataframe into (X, y) suitable for sequence models.

    X shape: (N, lookback, F)
    y shape: (N,)
    """
    X_all = df[feature_cols].to_numpy(dtype=np.float32)
    y_all = make_labels(df, horizon=horizon, threshold=threshold)

    # y_all is shorter by horizon
    n = len(y_all)

    X_list = []
    y_list = []

    for i in range(lookback, n):
        X_list.append(X_all[i - lookback:i])
        y_list.append(y_all[i])

    X = np.stack(X_list)  # (N, lookback, F)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def time_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Chronological split (no shuffling) to avoid leakage.

    Returns: (train), (val), (test)
    """
    n = len(y)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def class_distribution(y: np.ndarray) -> dict[int, float]:
    """
    Useful for reporting imbalance.
    """
    values, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    return {int(v): float(c) / float(total) for v, c in zip(values, counts)}