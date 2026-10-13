# src/storage.py
import sqlite3
from pathlib import Path
from typing import Optional, Any
import json

import pandas as pd


class TradingDatabase:
    """
    SQLite database for storing trading data pipeline artifacts.

    Tables:
    - raw_data: Unprocessed data from APIs (per-source rerun-safe)
    - validated_data: Cleaned data passing validation checks (overwritten per run)
    - features: Engineered features (stored as JSON per date; overwritten per run)
    - metadata: Pipeline execution metadata and statistics (key/value JSON)
    """

    def __init__(self, db_path: Path = Path("data/trading.db")):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        cur = self.conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                vix_close REAL,
                treasury_close REAL,
                gold_close REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source, date)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS validated_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                vix_close REAL,
                treasury_close REAL,
                gold_close REAL,
                validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                feature_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL UNIQUE,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        self.conn.commit()
        cur.close()

    # ----------------------------
    # Helpers
    # ----------------------------

    @staticmethod
    def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if "date" in out.columns:
            return out

        if "Date" in out.columns:
            return out.rename(columns={"Date": "date"})

        if out.index.name in ("Date", "date"):
            idx = out.index.name
            out = out.reset_index()
            return out.rename(columns={idx: "date"})

        raise ValueError("No date column found (expected 'date' or 'Date', or a Date/date index).")

    @staticmethod
    def _clean_dates(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"])
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        return out

    # ----------------------------
    # Raw data
    # ----------------------------

    def save_raw_data(self, df: pd.DataFrame, source: str) -> None:
        """
        Save raw data rerun-safely PER SOURCE.
        Strategy: DELETE rows for this source, then append fresh.
        """
        df0 = df.copy()
        df0["source"] = source

        df0 = df0.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "VIX_Close": "vix_close",
                "Treasury_Close": "treasury_close",
                "Gold_Close": "gold_close",
                # Intentionally ignore extra columns like "Adj Close"
            }
        )

        df0 = self._ensure_date_column(df0)
        df0 = self._clean_dates(df0)

        cols = [
            "source",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vix_close",
            "treasury_close",
            "gold_close",
        ]
        df_insert = df0[[c for c in cols if c in df0.columns]].copy()
        df_insert = df_insert.drop_duplicates(subset=["source", "date"], keep="last")

        # Rerun-safe per source
        cur = self.conn.cursor()
        cur.execute("DELETE FROM raw_data WHERE source = ?", (source,))
        self.conn.commit()
        cur.close()

        df_insert.to_sql("raw_data", self.conn, if_exists="append", index=False)
        self.conn.commit()

        print(f"Upserted {len(df_insert)} raw rows (source={source})")

    # ----------------------------
    # Validated data
    # ----------------------------

    def save_validated_data(self, df: pd.DataFrame) -> None:
        """
        Save validated/cleaned data (overwrites table each run).

        CRITICAL: only writes columns that exist in the validated_data schema.
        This prevents CI failures when the input includes extra cols like 'Adj Close'.
        """
        df0 = df.copy()

        df0 = df0.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "VIX_Close": "vix_close",
                "Treasury_Close": "treasury_close",
                "Gold_Close": "gold_close",
                # DO NOT store "Adj Close" (ignored by design)
            }
        )

        df0 = self._ensure_date_column(df0)
        df0 = self._clean_dates(df0)

        schema_cols = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vix_close",
            "treasury_close",
            "gold_close",
        ]
        df_insert = df0[[c for c in schema_cols if c in df0.columns]].copy()

        # Make this table rerun-safe: clear then insert
        cur = self.conn.cursor()
        cur.execute("DELETE FROM validated_data")
        self.conn.commit()
        cur.close()

        df_insert.to_sql("validated_data", self.conn, if_exists="append", index=False)
        self.conn.commit()

        print(f"Saved {len(df_insert)} rows of validated data to database")

    # ----------------------------
    # Features
    # ----------------------------

    def save_features(self, df_feat: pd.DataFrame) -> None:
        """
        Save engineered features per date as JSON (overwrites each run).
        Expects df_feat to include 'Date' or 'date' plus feature columns.
        """
        df0 = df_feat.copy()
        df0 = self._ensure_date_column(df0)
        df0 = self._clean_dates(df0)

        # Build JSON dict per row (excluding date)
        feature_cols = [c for c in df0.columns if c != "date"]
        records = []
        for _, row in df0.iterrows():
            feat_dict = {c: (None if pd.isna(row[c]) else float(row[c])) for c in feature_cols}
            records.append({"date": row["date"], "feature_json": json.dumps(feat_dict)})

        df_insert = pd.DataFrame(records)

        # Rerun-safe: clear table then insert
        cur = self.conn.cursor()
        cur.execute("DELETE FROM features")
        self.conn.commit()
        cur.close()

        df_insert.to_sql("features", self.conn, if_exists="append", index=False)
        self.conn.commit()

        print(f"Saved {len(df_insert)} rows of features to database")

    # ----------------------------
    # Housekeeping
    # ----------------------------

    def close(self) -> None:
        try:
            self.conn.commit()
        finally:
            self.conn.close()