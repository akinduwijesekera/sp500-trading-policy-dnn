from pathlib import Path
import pandas as pd
import yfinance as yf


def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Helper to normalize Date column after loading CSV or downloading."""
    if "Date" not in df.columns:
        if "date" in df.columns:
            df = df.rename(columns={"date": "Date"})
        elif "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
        elif "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "Date"})
        elif "index" in df.columns:
            df = df.rename(columns={"index": "Date"})
    
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    
    return df


def download_sp500(
    ticker: str,
    start_date: str,
    end_date: str,
    out_path: Path
) -> pd.DataFrame:
    """
    Download daily S&P 500 OHLCV data.
    
    Data Source: Yahoo Finance via yfinance library
    Ticker: ^GSPC (S&P 500 Index)
    
    Hypothesis: Primary target for trading decisions.
    The S&P 500 represents the broader US market and serves as our prediction target.
    
    If the file already exists, load it from disk instead of re-downloading.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"Loading cached S&P 500 data from {out_path}")
        df = pd.read_csv(out_path)
        df = _ensure_date_column(df)
        return df

    print(f"Downloading S&P 500 data ({ticker}) from {start_date} to {end_date}...")
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=True,
        group_by="column",
    )

    if df is None or df.empty:
        raise RuntimeError("Downloaded S&P 500 data is empty. Check ticker or dates.")

    df = df.reset_index()
    
    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if len(col) > 1 else col for col in df.columns.values]

    df = _ensure_date_column(df)
    
    if "Date" not in df.columns:
        raise RuntimeError(f"Could not create 'Date' column. Columns: {list(df.columns)}")

    df.to_csv(out_path, index=False)
    print(f"Saved S&P 500 data to {out_path}")
    return df


def download_vix(
    start_date: str,
    end_date: str,
    out_path: Path
) -> pd.DataFrame:
    """
    Download VIX (CBOE Volatility Index) data.
    
    Data Source: Yahoo Finance
    Ticker: ^VIX
    
    Hypothesis: Market Fear Indicator
    - VIX measures expected 30-day volatility implied by S&P 500 options
    - High VIX (>30) → Fear/uncertainty → Often precedes market declines → SELL signal
    - Low VIX (<15) → Complacency → Market stability or potential reversal → BUY/HOLD signal
    - Rising VIX → Increasing uncertainty → Risk-off sentiment
    
    Rationale: VIX is a leading indicator that can signal market regime changes
    before they appear in price data alone.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"Loading cached VIX data from {out_path}")
        df = pd.read_csv(out_path)
        df = _ensure_date_column(df)
        return df

    print(f"Downloading VIX data from {start_date} to {end_date}...")
    df = yf.download(
        "^VIX",
        start=start_date,
        end=end_date,
        interval="1d",
        progress=False,
        auto_adjust=False,
    )

    if df is None or df.empty:
        raise RuntimeError("Downloaded VIX data is empty.")

    df = df.reset_index()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if len(col) > 1 else col for col in df.columns.values]

    df = _ensure_date_column(df)
    df.to_csv(out_path, index=False)
    print(f"Saved VIX data to {out_path}")
    return df


def download_treasury_yields(
    start_date: str,
    end_date: str,
    out_path: Path
) -> pd.DataFrame:
    """
    Download 10-Year US Treasury Yields.
    
    Data Source: Yahoo Finance
    Ticker: ^TNX (10-Year Treasury Note Yield)
    
    Hypothesis: Interest Rate / Risk-Free Rate Indicator
    - Rising yields → Higher discount rates → Lower stock valuations → SELL signal
    - Rising yields → Capital flows from stocks to bonds → Market headwind
    - Falling yields → Cheaper borrowing → Growth stocks benefit → BUY signal
    - Yield curve inversions (when added 2Y yields) → Recession indicator
    
    Rationale: Treasury yields represent the risk-free rate and opportunity cost
    of equity investing. Changes in yields affect stock valuations through
    discounted cash flow models and investor allocation decisions.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"Loading cached Treasury yields from {out_path}")
        df = pd.read_csv(out_path)
        df = _ensure_date_column(df)
        return df

    print(f"Downloading Treasury yields from {start_date} to {end_date}...")
    df = yf.download(
        "^TNX",
        start=start_date,
        end=end_date,
        interval="1d",
        progress=False,
        auto_adjust=False,
    )

    if df is None or df.empty:
        raise RuntimeError("Downloaded Treasury data is empty.")

    df = df.reset_index()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if len(col) > 1 else col for col in df.columns.values]

    df = _ensure_date_column(df)
    df.to_csv(out_path, index=False)
    print(f"Saved Treasury yields to {out_path}")
    return df


def download_gold(
    start_date: str,
    end_date: str,
    out_path: Path
) -> pd.DataFrame:
    """
    Download Gold futures prices.
    
    Data Source: Yahoo Finance
    Ticker: GC=F (Gold Futures)
    
    Hypothesis: Safe-Haven Asset / Risk-Off Indicator
    - Gold typically exhibits negative correlation with equities during crises
    - Rising gold → Flight to safety → Risk-off environment → SELL stocks
    - Falling gold → Risk-on environment → Capital flows to equities → BUY stocks
    - Gold also inversely correlated with real interest rates
    
    Rationale: Gold serves as a hedge against market stress, inflation, and
    currency debasement. Strong gold performance often coincides with equity
    market weakness, making it a useful contrarian indicator.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"Loading cached Gold data from {out_path}")
        df = pd.read_csv(out_path)
        df = _ensure_date_column(df)
        return df

    print(f"Downloading Gold prices from {start_date} to {end_date}...")
    df = yf.download(
        "GC=F",
        start=start_date,
        end=end_date,
        interval="1d",
        progress=False,
        auto_adjust=False,
    )

    if df is None or df.empty:
        raise RuntimeError("Downloaded Gold data is empty.")

    df = df.reset_index()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if len(col) > 1 else col for col in df.columns.values]

    df = _ensure_date_column(df)
    df.to_csv(out_path, index=False)
    print(f"Saved Gold data to {out_path}")
    return df


def merge_external_data(
    sp500_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    treasury_df: pd.DataFrame,
    gold_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge all external data sources onto S&P 500 dataframe by Date.
    
    Strategy: Left join on S&P 500 dates (only keep trading days)
    Forward-fill missing values for external data (reasonable assumption:
    if VIX/Treasury/Gold not trading on a given day, use last known value)
    
    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume (S&P 500)
                                 VIX_Close, Treasury_Close, Gold_Close
    """
    df = sp500_df.copy()
    
    # Prepare external datasets (keep only Date and Close price)
    if "Close" in vix_df.columns:
        vix_df = vix_df[["Date", "Close"]].rename(columns={"Close": "VIX_Close"})
        df = df.merge(vix_df, on="Date", how="left")
    
    if "Close" in treasury_df.columns:
        treasury_df = treasury_df[["Date", "Close"]].rename(columns={"Close": "Treasury_Close"})
        df = df.merge(treasury_df, on="Date", how="left")
    
    if "Close" in gold_df.columns:
        gold_df = gold_df[["Date", "Close"]].rename(columns={"Close": "Gold_Close"})
        df = df.merge(gold_df, on="Date", how="left")
    
    # Forward-fill missing external data
    external_cols = ["VIX_Close", "Treasury_Close", "Gold_Close"]
    for col in external_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    
    return df


def acquire_all_data(
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    data_dir: Path = Path("data/raw")
) -> pd.DataFrame:
    """
    Master function to acquire all data sources.
    
    Temporal Scope Rationale:
    - Start: January 2020
        * Captures COVID-19 crash and recovery (major regime change)
        * Includes pandemic-era monetary policy (QE, near-zero rates)
        * ~5 years of data is sufficient for LSTM training
        
    - End: December 2024
        * Includes recent rate hike cycle (2022-2023)
        * Captures post-pandemic normalization
        * Recent enough to be relevant for current market conditions
    
    Why NOT longer history?
    - Market regime changes: Pre-2020 data may be less relevant
    - Computational efficiency: Longer history = more data, diminishing returns
    - Stationarity: Financial markets are non-stationary; distant past less predictive
    
    Why NOT shorter history?
    - Need sufficient data for sequence model (LSTM) training
    - Want to capture at least one full market cycle
    - Need enough data for meaningful train/val/test splits
    
    Returns:
        Merged dataframe with S&P 500 OHLCV + external indicators
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("DATA ACQUISITION")
    print("="*70)
    print(f"Temporal scope: {start_date} to {end_date}")
    print(f"Rationale: 5 years captures recent market regimes while maintaining relevance")
    print()
    
    # Download all datasets
    sp500_df = download_sp500("^GSPC", start_date, end_date, data_dir / "sp500_raw.csv")
    vix_df = download_vix(start_date, end_date, data_dir / "vix_raw.csv")
    treasury_df = download_treasury_yields(start_date, end_date, data_dir / "treasury_raw.csv")
    gold_df = download_gold(start_date, end_date, data_dir / "gold_raw.csv")
    
    # Merge all data
    print("\nMerging external data sources...")
    merged_df = merge_external_data(sp500_df, vix_df, treasury_df, gold_df)
    
    print(f"Merged dataset: {len(merged_df)} rows, {len(merged_df.columns)} columns")
    print(f"Columns: {list(merged_df.columns)}")
    print("="*70)
    
    return merged_df
