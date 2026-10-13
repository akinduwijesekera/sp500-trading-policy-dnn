import pandas as pd
import numpy as np


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive feature engineering for daily OHLCV + external data.
    
    Design Principles:
    - Use only information available up to time t (no look-ahead bias)
    - Mix of momentum, volatility, and trend indicators
    - Incorporate external market indicators (VIX, Treasury, Gold)
    - Create relative features (ratios, differences) for stationarity
    
    Feature Categories:
    1. Returns (momentum)
    2. Moving averages (trend)
    3. Volatility indicators
    4. Volume indicators
    5. External market indicators
    6. Technical oscillators
    
    Returns:
        DataFrame with ~20-25 engineered features
    """
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    # ======================================================================
    # 1. RETURNS (Momentum indicators)
    # ======================================================================
    df["ret_1d"] = df["Close"].pct_change()
    df["ret_5d"] = df["Close"].pct_change(5)
    df["ret_20d"] = df["Close"].pct_change(20)
    
    # Log returns (more stable for extreme moves)
    df["log_ret_1d"] = np.log(df["Close"] / df["Close"].shift(1))

    # ======================================================================
    # 2. MOVING AVERAGES (Trend indicators)
    # ======================================================================
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()
    
    # MA Crossovers (Golden Cross / Death Cross signals)
    df["MA_5_20_cross"] = (df["MA_5"] - df["MA_20"]) / df["MA_20"]  # Normalized
    df["MA_20_50_cross"] = (df["MA_20"] - df["MA_50"]) / df["MA_50"]  # Normalized
    
    # Price relative to MA (overbought/oversold)
    df["price_to_MA20"] = (df["Close"] - df["MA_20"]) / df["MA_20"]
    df["price_to_MA50"] = (df["Close"] - df["MA_50"]) / df["MA_50"]

    # ======================================================================
    # 3. VOLATILITY INDICATORS
    # ======================================================================
    # Rolling volatility of returns
    df["volatility_5"] = df["ret_1d"].rolling(5).std()
    df["volatility_20"] = df["ret_1d"].rolling(20).std()
    
    # Bollinger Bands
    df["BB_upper"] = df["MA_20"] + 2 * df["Close"].rolling(20).std()
    df["BB_lower"] = df["MA_20"] - 2 * df["Close"].rolling(20).std()
    df["BB_position"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"] + 1e-8)
    
    # High-Low range (intraday volatility)
    df["HL_range"] = (df["High"] - df["Low"]) / df["Close"]

    # ======================================================================
    # 4. VOLUME INDICATORS
    # ======================================================================
    df["volume_MA_20"] = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / (df["volume_MA_20"] + 1e-8)
    
    # On-Balance Volume (OBV) - cumulative volume with direction
    df["OBV"] = (np.sign(df["ret_1d"]) * df["Volume"]).cumsum()
    df["OBV_MA_20"] = df["OBV"].rolling(20).mean()
    df["OBV_signal"] = (df["OBV"] - df["OBV_MA_20"]) / (df["OBV_MA_20"].abs() + 1e-8)

    # ======================================================================
    # 5. TECHNICAL OSCILLATORS
    # ======================================================================
    # RSI (Relative Strength Index) - 0 to 100
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # Normalize RSI to [-1, 1] for better neural network training
    df["RSI_norm"] = (df["RSI"] - 50) / 50
    
    # Rate of Change (ROC) - momentum
    df["ROC_5"] = (df["Close"] - df["Close"].shift(5)) / (df["Close"].shift(5) + 1e-8)
    df["ROC_20"] = (df["Close"] - df["Close"].shift(20)) / (df["Close"].shift(20) + 1e-8)
    
    # MACD (Moving Average Convergence Divergence)
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    
    # Normalize MACD by price for scale-independence
    df["MACD_norm"] = df["MACD"] / (df["Close"] + 1e-8)

    # ======================================================================
    # 6. EXTERNAL MARKET INDICATORS
    # ======================================================================
    # VIX (Fear index)
    if "VIX_Close" in df.columns:
        df["VIX_ret_1d"] = df["VIX_Close"].pct_change()
        df["VIX_MA_5"] = df["VIX_Close"].rolling(5).mean()
        df["VIX_zscore"] = (df["VIX_Close"] - df["VIX_MA_5"]) / (df["VIX_Close"].rolling(5).std() + 1e-8)
        
        # VIX level indicators (regime detection)
        df["VIX_high"] = (df["VIX_Close"] > 25).astype(float)  # High fear
        df["VIX_low"] = (df["VIX_Close"] < 15).astype(float)   # Low fear
    
    # Treasury Yields (Interest rate proxy)
    if "Treasury_Close" in df.columns:
        df["Treasury_ret_1d"] = df["Treasury_Close"].pct_change()
        df["Treasury_MA_20"] = df["Treasury_Close"].rolling(20).mean()
        df["Treasury_trend"] = (df["Treasury_Close"] - df["Treasury_MA_20"]) / (df["Treasury_MA_20"] + 1e-8)
    
    # Gold (Safe-haven indicator)
    if "Gold_Close" in df.columns:
        df["Gold_ret_1d"] = df["Gold_Close"].pct_change()
        df["Gold_MA_20"] = df["Gold_Close"].rolling(20).mean()
        df["Gold_trend"] = (df["Gold_Close"] - df["Gold_MA_20"]) / (df["Gold_MA_20"] + 1e-8)
        
        # Gold-SPX correlation (risk-on/risk-off)
        df["Gold_SPX_corr"] = df["Gold_ret_1d"].rolling(20).corr(df["ret_1d"])

    # ======================================================================
    # 7. CALENDAR FEATURES (Market seasonality)
    # ======================================================================
    df["day_of_week"] = pd.to_datetime(df["Date"]).dt.dayofweek / 4.0  # Normalize to [0,1]
    df["month"] = pd.to_datetime(df["Date"]).dt.month / 11.0  # Normalize to [0,1]

    # Drop rows with NaNs introduced by rolling windows
    initial_rows = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped_rows = initial_rows - len(df)
    
    if dropped_rows > 0:
        print(f"Feature engineering: Dropped {dropped_rows} rows due to rolling window NaNs")

    return df


def get_feature_columns() -> list[str]:
    """
    Define which columns are used as model inputs.
    
    Selection Rationale:
    - Excluded: Date, raw OHLCV prices (non-stationary)
    - Included: Returns, ratios, normalized indicators (more stationary)
    - Included: External market indicators
    - Total: ~25 features
    
    This creates a rich feature set balancing:
    - Momentum (returns, ROC, MACD)
    - Trend (MA crossovers, price-to-MA)
    - Volatility (Bollinger Bands, vol measures)
    - Volume (OBV, volume ratio)
    - Market regime (VIX, Treasury, Gold)
    - Technical oscillators (RSI, MACD)
    """
    base_features = [
        # Returns
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "log_ret_1d",
        
        # Moving average signals
        "MA_5_20_cross",
        "MA_20_50_cross",
        "price_to_MA20",
        "price_to_MA50",
        
        # Volatility
        "volatility_5",
        "volatility_20",
        "BB_position",
        "HL_range",
        
        # Volume
        "volume_ratio",
        "OBV_signal",
        
        # Technical oscillators
        "RSI_norm",
        "ROC_5",
        "ROC_20",
        "MACD_norm",
        "MACD_hist",
        
        # Calendar
        "day_of_week",
        "month",
    ]
    
    # External indicators (conditional on availability)
    external_features = [
        "VIX_ret_1d",
        "VIX_zscore",
        "VIX_high",
        "VIX_low",
        "Treasury_ret_1d",
        "Treasury_trend",
        "Gold_ret_1d",
        "Gold_trend",
        "Gold_SPX_corr",
    ]
    
    return base_features + external_features


def get_feature_descriptions() -> dict[str, str]:
    """
    Return human-readable descriptions of each feature.
    Useful for your report's feature analysis section.
    """
    return {
        "ret_1d": "1-day return (momentum)",
        "ret_5d": "5-day return (short-term momentum)",
        "ret_20d": "20-day return (medium-term momentum)",
        "log_ret_1d": "Log return (handles extreme moves better)",
        "MA_5_20_cross": "5-day MA vs 20-day MA (short-term trend)",
        "MA_20_50_cross": "20-day MA vs 50-day MA (medium-term trend)",
        "price_to_MA20": "Price deviation from 20-day MA (overbought/oversold)",
        "price_to_MA50": "Price deviation from 50-day MA (trend strength)",
        "volatility_5": "5-day return volatility (short-term risk)",
        "volatility_20": "20-day return volatility (medium-term risk)",
        "BB_position": "Position within Bollinger Bands (0=lower, 1=upper)",
        "HL_range": "High-Low range normalized by close (intraday volatility)",
        "volume_ratio": "Current volume vs 20-day average (volume surge/drop)",
        "OBV_signal": "On-Balance Volume signal (buying/selling pressure)",
        "RSI_norm": "RSI normalized to [-1, 1] (overbought/oversold)",
        "ROC_5": "5-day rate of change (momentum)",
        "ROC_20": "20-day rate of change (trend)",
        "MACD_norm": "MACD normalized by price (momentum)",
        "MACD_hist": "MACD histogram (momentum acceleration)",
        "day_of_week": "Day of week normalized (market seasonality)",
        "month": "Month normalized (calendar effects)",
        "VIX_ret_1d": "VIX 1-day return (fear change)",
        "VIX_zscore": "VIX Z-score (fear spike detection)",
        "VIX_high": "VIX > 25 indicator (high fear regime)",
        "VIX_low": "VIX < 15 indicator (low fear regime)",
        "Treasury_ret_1d": "10Y Treasury yield change (rate direction)",
        "Treasury_trend": "Treasury yield trend (rate regime)",
        "Gold_ret_1d": "Gold 1-day return (safe-haven flows)",
        "Gold_trend": "Gold trend (risk-on/risk-off)",
        "Gold_SPX_corr": "20-day rolling correlation Gold-SPX (regime)",
    }