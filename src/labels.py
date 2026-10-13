import numpy as np
import pandas as pd


LABEL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}


def make_labels(df: pd.DataFrame, horizon: int, threshold: float) -> np.ndarray:
    """
    Create Buy/Hold/Sell labels based on future H-day return.

    r_{t,H} = (Close_{t+H} - Close_t) / Close_t

    BUY  if r_{t,H} > +threshold
    SELL if r_{t,H} < -threshold
    HOLD otherwise
    
    Rationale for threshold selection:
    - Too low (0.1-0.5%): Creates noisy labels, hard to learn
    - Too high (3-5%): Too few BUY/SELL signals, mostly HOLD
    - Sweet spot (1-2%): Balance between signal and noise
    
    Default recommendation: threshold = 0.01 (1%) for 5-day horizon
    """
    future_close = df["Close"].shift(-horizon)
    future_ret = (future_close / df["Close"]) - 1.0

    y = np.full(len(df), 1, dtype=np.int64)  # HOLD
    y[future_ret > threshold] = 2            # BUY
    y[future_ret < -threshold] = 0           # SELL

    # Last horizon rows have unknown future; drop them
    y = y[:-horizon]
    
    return y


def analyze_label_distribution(df: pd.DataFrame, horizon: int, thresholds: list[float]) -> None:
    """
    Utility function to analyze how different thresholds affect label distribution.
    Useful for threshold selection in your report.
    
    Usage:
        analyze_label_distribution(df, horizon=5, thresholds=[0.005, 0.01, 0.015, 0.02])
    """
    print(f"\nLabel Distribution Analysis (horizon={horizon} days)")
    print("=" * 70)
    
    for thresh in thresholds:
        y = make_labels(df, horizon=horizon, threshold=thresh)
        
        sell_pct = (y == 0).mean() * 100
        hold_pct = (y == 1).mean() * 100
        buy_pct = (y == 2).mean() * 100
        
        print(f"Threshold {thresh*100:.1f}%: "
              f"SELL {sell_pct:.1f}% | HOLD {hold_pct:.1f}% | BUY {buy_pct:.1f}%")
    
    print("=" * 70)
