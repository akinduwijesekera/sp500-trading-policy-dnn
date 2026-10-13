import re
from typing import Any
import pandas as pd


# Object model (schema) for a valid daily price record
REQUIRED_COLS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def _flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If DataFrame has MultiIndex columns (e.g. yfinance when downloading multiple tickers),
    flatten them to a single-level by joining with an underscore.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join([str(c) for c in col if c not in (None, "")]).strip("_")
            for col in df.columns.values
        ]
    return df


def _find_date_like_column(columns: pd.Index) -> str | None:
    """
    Heuristically find a date-like column among column names.
    Return the column name if found, else None.
    """
    date_patterns = [r"(?i)^date$", r"(?i)date", r"(?i)datetime", r"(?i)^time$"]
    for pattern in date_patterns:
        for col in columns:
            if re.search(pattern, str(col)):
                return col
    return None


def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee a 'Date' column exists.

    Strategies:
    - Flatten MultiIndex columns
    - If 'Date' exists (case sensitive) -> done
    - If 'date' lowercase exists -> rename
    - If index is DatetimeIndex -> reset_index() and normalise name to 'Date'
    - If a column matches common date-like names -> rename to 'Date'
    - If first column looks parseable to datetime (>=90% non-na after parse) -> rename it to 'Date'
    - If file was saved with 'Unnamed: 0' -> rename to 'Date'
    """

    if df is None:
        raise ValueError("_ensure_date_column received None")

    df = _flatten_multiindex_columns(df)
    df = df.copy()

    # Already good
    if "Date" in df.columns:
        return df

    # Lowercase date
    if "date" in df.columns:
        df = df.rename(columns={"date": "Date"})
        return df

    # If index is a DatetimeIndex, make it a column
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        # after reset, try canonical names
        if "Date" in df.columns:
            return df
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
            return df
        if "index" in df.columns and "Date" not in df.columns:
            df = df.rename(columns={"index": "Date"})
            return df

    # Detect date-like column by name
    candidate = _find_date_like_column(df.columns)
    if candidate is not None and candidate != "Date":
        df = df.rename(columns={candidate: "Date"})
        return df

    # CSV quirks (e.g. saved df.reset_index() may produce Unnamed: 0)
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "Date"})
        return df

    # Another common fallback column name
    if "index" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"index": "Date"})
        return df

    # Fallback: first column parseable as datetime
    if len(df.columns) > 0:
        first = df.columns[0]
        try:
            parsed = pd.to_datetime(df[first], errors="coerce")
            if parsed.notna().mean() > 0.9:
                df = df.rename(columns={first: "Date"})
                return df
        except Exception:
            pass

    # If we still cannot find a Date column, raise an informative error
    raise KeyError(
        "Could not infer a 'Date' column from input dataframe. "
        f"Available columns: {list(df.columns)}, Index type: {type(df.index).__name__}, Index name: {df.index.name}"
    )


def _normalize_ticker_suffixed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect columns that are suffixed by a ticker (e.g. 'Open_^GSPC', 'Close_^GSPC')
    and, if exactly one ticker suffix is found across the OHLCV candidates,
    rename them to the canonical column names: 'Open','High','Low','Close','Volume'.

    If multiple tickers are present, return df unchanged (consumer should explicitly handle multi-ticker inputs).
    """
    df = _flatten_multiindex_columns(df).copy()
    cols = list(df.columns)

    # Pattern to capture base name and optional suffix separated by underscore/hyphen/space
    # Examples matched: "Open_^GSPC", "Close-^GSPC", "High ^GSPC"
    pattern = re.compile(r"^(Open|High|Low|Close|Adj Close|Volume)(?:[_\-\s](.*))?$", flags=re.IGNORECASE)

    suffix_map: dict[str, set] = {}
    col_matches: dict[str, tuple[str, str | None]] = {}

    for col in cols:
        m = pattern.match(str(col))
        if m:
            base = m.group(1)
            suf = m.group(2) if m.group(2) not in (None, "") else None
            # normalize 'Adj Close' -> 'Close'
            base_norm = "Close" if base.lower() == "adj close" else base.capitalize()
            col_matches[col] = (base_norm, suf)
            key = suf or "__no_suffix__"
            suffix_map.setdefault(key, set()).add(base_norm)

    # consider only columns that matched OHLCV-like names
    if not col_matches:
        return df

    # Determine non-empty suffixes present
    non_empty_suffixes = {s: cols for s, cols in suffix_map.items() if s != "__no_suffix__"}

    # If there's exactly one non-empty suffix (i.e. single ticker suffix) and it covers OHLCV bases, rename
    if len(non_empty_suffixes) == 1:
        suffix = next(iter(non_empty_suffixes))
        # Build rename mapping for columns that end with that suffix
        rename_map: dict[str, str] = {}
        for col, (base_norm, suf) in col_matches.items():
            if suf == suffix:
                canonical = base_norm
                # Avoid overwriting an existing canonical column
                if canonical not in df.columns:
                    rename_map[col] = canonical
        if rename_map:
            df = df.rename(columns=rename_map)
    # If no non-empty suffixes but there are columns like 'Open__no_suffix__' nothing to do
    return df


def _ensure_required_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the numeric columns exist or attempt sensible fallbacks:
    - If 'Close' is missing but 'Adj Close' exists, copy it to 'Close'
    - After normalization, if required numeric columns are still missing, raise ValueError with diagnostics
    """
    df = df.copy()
    df = _flatten_multiindex_columns(df)

    # Fallback: map 'Adj Close' -> 'Close' if needed
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        # don't fail silently; give helpful diagnostics
        raise ValueError(
            f"Missing required columns after normalization: {missing}. "
            f"Columns present: {list(df.columns)}"
        )
    return df


def validate_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validation + cleaning for raw OHLCV prices.

    Accepts either:
    - Date as a column, OR
    - Date as the index (common for yfinance)

    Enforces object model:
    - Date: valid datetime
    - Open/High/Low/Close/Volume: numeric, non-missing
    - One row per trading day (drop duplicate dates)
    - Sorted in time order
    """

    if df is None:
        raise ValueError("validate_prices received None instead of a DataFrame.")
    if len(df) == 0:
        raise ValueError("validate_prices received an empty dataframe.")

    # Work on a copy
    df = df.copy()

    # --- Normalise columns and Date column ---
    try:
        df = _ensure_date_column(df)
    except KeyError as e:
        # Provide helpful debug info for CI logs
        preview_cols = list(df.columns)
        preview_head = df.head(5).to_dict(orient="list")
        raise KeyError(
            "Failed to infer 'Date' column during normalization. "
            f"Original error: {e}. Columns present: {preview_cols}. "
            f"Sample data (first rows): {preview_head}"
        ) from e

    # Try to strip a single ticker suffix if present (e.g. "Open_^GSPC" -> "Open")
    df = _normalize_ticker_suffixed_columns(df)

    # Final fallback: if index is datetime and Date still absent, reset
    if "Date" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        if "index" in df.columns and "Date" not in df.columns:
            df = df.rename(columns={"index": "Date"})

    # Ensure 'Date' now exists
    if "Date" not in df.columns:
        raise ValueError(f"Critical error: Date column missing after normalization. Columns: {list(df.columns)}")

    # Attempt to ensure required numeric columns exist (with fallbacks)
    try:
        df = _ensure_required_numeric_columns(df)
    except ValueError as e:
        # surface clear error for CI
        raise ValueError(f"Required numeric columns missing: {e}") from e

    # 2) Parse Date safely
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Drop rows with invalid dates (log count)
    invalid_dates = int(df["Date"].isna().sum())
    if invalid_dates > 0:
        print(f"Warning: Dropping {invalid_dates} rows with invalid dates")
    df = df.dropna(subset=["Date"])

    # Check if we have any data left
    if len(df) == 0:
        raise ValueError("No valid data remaining after date parsing")

    # 3) Sort + deduplicate (one record per day)
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

    # 4) Coerce numeric columns, drop invalid rows
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Count invalid rows before dropping
    invalid_rows = int(df[numeric_cols].isna().any(axis=1).sum())
    if invalid_rows > 0:
        print(f"Warning: Dropping {invalid_rows} rows with invalid OHLCV data")

    df = df.dropna(subset=numeric_cols)

    # Check if we have any data left
    if len(df) == 0:
        raise ValueError("No valid data remaining after numeric validation")

    # 5) Basic sanity checks
    df = df[df["Volume"] >= 0].reset_index(drop=True)

    # Extra: remove rows with nonsensical OHLC
    bad_ohlc = df["High"] < df["Low"]
    if bad_ohlc.any():
        num_bad = int(bad_ohlc.sum())
        print(f"Warning: Found {num_bad} rows where High < Low. Removing these rows.")
        df = df[~bad_ohlc].reset_index(drop=True)

    # Additional sanity: Close should be between Low and High
    bad_close = (df["Close"] < df["Low"]) | (df["Close"] > df["High"])
    if bad_close.any():
        num_bad = int(bad_close.sum())
        print(f"Warning: Found {num_bad} rows where Close is outside Low/High range. Removing these rows.")
        df = df[~bad_close].reset_index(drop=True)

    # Final check
    if len(df) == 0:
        raise ValueError("No valid data remaining after all validation checks")

    # Keep deterministic ordering
    df = df.sort_values("Date").reset_index(drop=True)

    return df