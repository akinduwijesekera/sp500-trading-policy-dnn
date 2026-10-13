# S&P 500 Trading Policy via Deep Sequence Modelling

An end-to-end AI system that learns a discrete trading policy (Buy / Hold / Sell) for the S&P 500 index from historical price, volatility, interest-rate, and gold-market data. Built as the final assignment for UCL's ELEC0136 — AI System Design module.

## Overview

The system treats trading as a supervised classification problem: given a 20-day window of engineered features, predict one of three actions over the following 5-day horizon. It covers the full pipeline end to end — data acquisition, validation, storage, feature engineering, dataset construction, model training, baseline comparison, and evaluation — rather than just the model itself.

## Data

- **Sources:** S&P 500 OHLCV, CBOE VIX, US Treasury yields, and gold spot prices, all pulled via the `yfinance` API.
- **Period:** January 2020 – December 2025.
- **Storage:** Raw and processed data persisted to a local SQLite database (`data/trading.db`), with a validation stage checking for missing dates, gaps, and anomalous values before features are built.

## Dataset Construction

- **Lookback window:** 20 trading days of history per sample.
- **Features:** 30 engineered features per sample, grouped into categories (price/return-based, moving averages, volatility, external indicators, etc.).
- **Labelling:** Forward-looking 5-day return thresholded at ±0.5% to assign Buy / Hold / Sell labels.
- **Split:** Chronological train/validation/test split (841 / 180 / 181 samples) to avoid look-ahead bias — no shuffling across time.

Feature relevance was checked with PCA and a correlation analysis before committing to the final feature set, and a group-wise ablation study was run post-hoc to quantify which feature groups actually mattered (see Results).

## Model
Linear(128) → LSTM(128) → LSTM(128) → Linear(128) → Policy(3)


A two-layer LSTM sequence model over the 20-day feature window, with linear projection layers at the input and output, producing a 3-way softmax over the action space.

- **Optimizer:** Adam
- **Loss:** Categorical cross-entropy with class weighting (to counter label imbalance)
- **Model selection:** Early stopping on validation macro-F1
- **Training budget:** 15-minute cap per run
- **Seeds:** 3 random seeds, results reported as mean ± std

## Baselines

The DNN is compared against five baselines: majority-class, last-return, logistic regression, random forest, and XGBoost. Confusion matrices for every model/seed combination are included in the full report.

## Results

| Model | Test Accuracy | Test Macro-F1 |
|---|---|---|
| Best baseline (XGBoost) | — | 0.389 |
| LSTM policy (mean, 3 seeds) | 0.407 ± 0.012 | 0.401 ± 0.021 |

The LSTM policy outperforms the strongest baseline by roughly 3% relative improvement in macro-F1.

**Ablation findings:** removing the Moving Averages feature group caused the largest drop in performance, followed by the External Indicators group (VIX/yields/gold), indicating the model relies most heavily on trend and macro-context signals rather than raw price action alone.

## Project Structure
├── main.py # Pipeline entry point (acquisition → validation → training → evaluation)
├── src/
│ ├── features.py # Feature engineering
│ └── ... # Data acquisition, validation, model, training, evaluation modules
├── data/ # Raw/processed data, SQLite DB (gitignored)
├── outputs/ # Trained model checkpoints, metrics, ablation results (gitignored)
├── figures/ # Generated plots (gitignored)
├── environment.yml
└── requirements.txt


## Running

```bash
conda env create -f environment.yml
conda activate <env-name>
python main.py
```

`main.py` runs the full staged pipeline: data acquisition → validation → storage → feature engineering → dataset construction → baseline training → feature/ablation analysis → DNN training → evaluation → results summary.

## Notes

- Individual coursework for UCL's ELEC0136 — AI System Design module.
- AI assistance was used for code structuring and documentation only (UCL Category 2 compliant), not for the modelling methodology or analysis.
