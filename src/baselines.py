from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")


def majority_class_baseline(y_train: np.ndarray, n: int) -> np.ndarray:
    """
    Baseline 1: Majority Class Predictor
    
    Strategy: Always predict the most frequent class in training data.
    
    Rationale: Simplest possible baseline. Any useful model must beat this.
    Expected performance: ~accuracy = majority class frequency, low F1 (poor on minorities)
    
    Use case: Sanity check that model is learning something beyond trivial pattern
    """
    values, counts = np.unique(y_train, return_counts=True)
    majority = int(values[np.argmax(counts)])
    return np.full(n, majority, dtype=np.int64)


def last_return_baseline(X: np.ndarray, hold_band: float = 0.0) -> np.ndarray:
    """
    Baseline 2: Last Return Heuristic
    
    Strategy: Use the most recent 1-day return to make decision
    - If last return > hold_band => predict BUY (momentum continuation)
    - If last return < -hold_band => predict SELL (momentum continuation)
    - Otherwise => predict HOLD
    
    Rationale: Tests if simple momentum signal has predictive power.
    Assumes feature 0 is 'ret_1d' (as defined in features.py).
    
    Expected performance: Better than random if short-term momentum exists,
    but likely worse than sophisticated models that use full feature set.
    
    Args:
        X: Input sequences (N, lookback, features)
        hold_band: Threshold for HOLD decision (default 0.0)
    
    Returns:
        Predictions (N,)
    """
    last_ret = X[:, -1, 0]  # Last timestep, feature 'ret_1d'
    y_pred = np.full(len(last_ret), 1, dtype=np.int64)  # HOLD
    y_pred[last_ret > hold_band] = 2  # BUY
    y_pred[last_ret < -hold_band] = 0  # SELL
    return y_pred


def random_forest_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42
) -> np.ndarray:
    """
    Baseline 3: Random Forest Classifier
    
    Strategy: Ensemble of decision trees on flattened sequence features.
    
    Rationale: 
    - Strong non-linear baseline
    - Handles feature interactions well
    - No sequence modeling (treats each timestep as independent feature)
    - Industry-standard ML baseline
    
    Expected performance: Should outperform simple heuristics, but may
    underperform LSTM if temporal dependencies are important.
    
    Why this matters: If Random Forest matches LSTM, it suggests:
    (a) temporal patterns aren't critical, or
    (b) LSTM isn't properly learning temporal structure
    
    Args:
        X_train: Training sequences (N, lookback, features)
        y_train: Training labels
        X_test: Test sequences
        n_estimators: Number of trees
        max_depth: Maximum tree depth (prevents overfitting)
        random_state: Random seed
    
    Returns:
        Predictions on test set
    """
    # Flatten sequences: (N, lookback, features) -> (N, lookback * features)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Train Random Forest with class balancing
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',  # Handle class imbalance
        random_state=random_state,
        n_jobs=-1,  # Use all CPU cores
    )
    
    print(f"Training Random Forest ({n_estimators} trees, max_depth={max_depth})...")
    rf.fit(X_train_flat, y_train)
    
    y_pred = rf.predict(X_test_flat)
    return y_pred


def logistic_regression_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    random_state: int = 42
) -> np.ndarray:
    """
    Baseline 4: Logistic Regression
    
    Strategy: Linear classifier on flattened features with L2 regularization.
    
    Rationale:
    - Tests if problem is linearly separable
    - Fast to train, interpretable coefficients
    - Strong baseline for many classification tasks
    
    Expected performance: Likely weakest baseline (too simple), but useful
    to verify non-linearity is needed.
    
    Args:
        X_train: Training sequences (N, lookback, features)
        y_train: Training labels
        X_test: Test sequences
        random_state: Random seed
    
    Returns:
        Predictions on test set
    """
    # Flatten sequences
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Train logistic regression with class balancing
    lr = LogisticRegression(
        solver='lbfgs',
        max_iter=5000,
        class_weight='balanced',
        random_state=random_state,
    )
    
    print("Training Logistic Regression...")
    lr.fit(X_train_flat, y_train)
    
    y_pred = lr.predict(X_test_flat)
    return y_pred


def xgboost_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 6,
    random_state: int = 42
) -> np.ndarray:
    """
    Baseline 5: XGBoost Classifier
    
    Strategy: Gradient boosting on flattened features.
    
    Rationale:
    - State-of-the-art tree-based method
    - Often wins Kaggle competitions
    - Strong handling of complex feature interactions
    - More powerful than Random Forest typically
    
    Expected performance: Likely strongest non-deep-learning baseline.
    If LSTM doesn't beat XGBoost significantly, suggests:
    (a) Temporal structure isn't critical, or
    (b) Need better LSTM architecture/features
    
    Args:
        X_train: Training sequences (N, lookback, features)
        y_train: Training labels
        X_test: Test sequences
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        random_state: Random seed
    
    Returns:
        Predictions on test set
    """
    if not XGBOOST_AVAILABLE:
        print("XGBoost not available, skipping...")
        return None
    
    # Flatten sequences
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Compute class weights
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    sample_weights = np.array([class_weights[y] for y in y_train])
    
    # Train XGBoost
    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        tree_method='hist',  # Faster training
        n_jobs=-1,
    )
    
    print(f"Training XGBoost ({n_estimators} rounds, max_depth={max_depth})...")
    xgb.fit(X_train_flat, y_train, sample_weight=sample_weights)
    
    y_pred = xgb.predict(X_test_flat)
    return y_pred


def run_all_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    random_state: int = 42
) -> dict[str, np.ndarray]:
    """
    Run all baseline models and return predictions.
    
    Returns:
        Dictionary mapping baseline name to test predictions
    """
    baselines = {}
    
    # 1. Majority class
    baselines["majority_class"] = majority_class_baseline(y_train, n=len(X_test))
    
    # 2. Last return heuristic
    baselines["last_return"] = last_return_baseline(X_test, hold_band=0.0)
    
    # 3. Logistic Regression
    baselines["logistic_regression"] = logistic_regression_baseline(
        X_train, y_train, X_test, random_state=random_state
    )
    
    # 4. Random Forest
    baselines["random_forest"] = random_forest_baseline(
        X_train, y_train, X_test,
        n_estimators=100,
        max_depth=10,
        random_state=random_state
    )
    
    # 5. XGBoost (if available)
    if XGBOOST_AVAILABLE:
        xgb_pred = xgboost_baseline(
            X_train, y_train, X_test,
            n_estimators=100,
            max_depth=6,
            random_state=random_state
        )
        if xgb_pred is not None:
            baselines["xgboost"] = xgb_pred
    
    return baselines
