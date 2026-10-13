"""
ELEC0136 Final Assessment: S&P 500 Trading Policy System
Student: 22008430

This pipeline implements an end-to-end AI decision system for S&P 500 trading.

Pipeline stages:
1. Data Acquisition (multiple sources with hypotheses)
2. Data Validation & Storage (SQLite database)
3. Feature Engineering (20+ technical indicators)
4. Dataset Construction (windowed sequences for LSTM)
5. Baseline Models (5 different approaches)
6. Deep Neural Network Training (required architecture)
7. Feature Analysis (correlation, PCA, ablation)
8. Evaluation & Results

Design Principles:
- Reproducibility: Fixed seeds, deterministic operations
- Auditability: Clear logging of all decisions
- Leak-free: Chronological splits, no future information
- Documented: Every choice justified with rationale
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Import all modules
from src.seed import set_global_seed
from src.acquire import acquire_all_data
from src.validate import validate_prices
from src.storage import TradingDatabase
from src.features import add_features, get_feature_columns
from src.dataset import make_windows, time_split, class_distribution
from src.baselines import run_all_baselines
from src.metrics import compute_metrics
from src.plotting import save_confusion_matrix
from src.model import PolicyNetwork
from src.train import TrainConfig, train_model, _predict
from src.analysis import (
    plot_feature_correlations,
    plot_pca_visualization,
    plot_training_history,
    create_analysis_report,
    analyze_feature_groups,
    run_group_ablation_inference,
    plot_group_ablation,
)


def main():
    """Main pipeline execution."""
    
    print("\n" + "="*80)
    print(" " * 20 + "ELEC0136 AI SYSTEM DESIGN")
    print(" " * 15 + "S&P 500 Trading Policy System")
    print("="*80)
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    # Acquisition config
    START_DATE = "2020-01-01"
    END_DATE = "2025-12-31"
    
    # Dataset config
    LOOKBACK = 20      # Use 20 days of history
    HORIZON = 5        # Predict 5-day future
    THRESHOLD = 0.005   # 0.5% threshold for BUY/SELL labels (reduced noise)
    
    # Training config
    SEEDS = [0, 1, 2]  # Multiple seeds for statistical significance
    
    # Paths
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    outputs_dir = Path("outputs")
    figures_dir = Path("figures")
    
    # Create directories
    for dir_path in [raw_dir, processed_dir, outputs_dir, figures_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # STAGE 1: DATA ACQUISITION
    # =========================================================================
    print("\n" + "="*80)
    print("STAGE 1: DATA ACQUISITION")
    print("="*80)
    
    df_merged = acquire_all_data(
        start_date=START_DATE,
        end_date=END_DATE,
        data_dir=raw_dir
    )
    
    print(f"\nAcquired {len(df_merged)} days of data")
    print(f"Date range: {df_merged['Date'].min()} to {df_merged['Date'].max()}")
    print(f"Columns: {list(df_merged.columns)}")
    
    # =========================================================================
    # STAGE 2: DATA VALIDATION & STORAGE
    # =========================================================================
    print("\n" + "="*80)
    print("STAGE 2: DATA VALIDATION & STORAGE")
    print("="*80)
    
    print("\nValidating data...")
    df_clean = validate_prices(df_merged)
    
    print(f"Validated {len(df_clean)} rows (dropped {len(df_merged) - len(df_clean)} invalid rows)")
    
    # Save to SQLite database
    print("\nStoring data in SQLite database...")
    db = TradingDatabase(db_path=data_dir / "trading.db")
    
    db.save_raw_data(df_merged, source="sp500_merged")
    db.save_validated_data(df_clean)
    
    # db_stats = db.get_statistics()
    print(f"Database saved successfully")
    
    # Also save CSV for manual inspection
    df_clean.to_csv(processed_dir / "sp500_validated.csv", index=False)
    
    # =========================================================================
    # STAGE 3: FEATURE ENGINEERING
    # =========================================================================
    print("\n" + "="*80)
    print("STAGE 3: FEATURE ENGINEERING")
    print("="*80)
    
    print("\nEngineering features...")
    df_feat = add_features(df_clean)
    
    feature_cols = get_feature_columns()
    print(f"Created {len(feature_cols)} features")
    print(f"Feature columns: {feature_cols}")
    
    # Save features to database and CSV
    db.save_features(df_feat)
    df_feat.to_csv(processed_dir / "sp500_features.csv", index=False)
    
    print(f"Final feature dataset: {len(df_feat)} rows")
    
    # =========================================================================
    # STAGE 4: DATASET CONSTRUCTION
    # =========================================================================
    print("\n" + "="*80)
    print("STAGE 4: DATASET CONSTRUCTION")
    print("="*80)
    
    print(f"\nCreating windowed dataset...")
    print(f"  Lookback: {LOOKBACK} days")
    print(f"  Horizon: {HORIZON} days")
    print(f"  Threshold: {THRESHOLD*100:.1f}%")
    
    X, y = make_windows(
        df=df_feat,
        feature_cols=feature_cols,
        lookback=LOOKBACK,
        horizon=HORIZON,
        threshold=THRESHOLD,
    )
    
    print(f"X shape: {X.shape} (samples, lookback, features)")
    print(f"y shape: {y.shape}")
    
    # Time-based split (no shuffling to prevent leakage)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = time_split(
        X, y, train_frac=0.70, val_frac=0.15
    )
    
    print(f"\nTrain/Val/Test split:")
    print(f"  Train: {len(y_train)} samples ({len(y_train)/len(y)*100:.1f}%)")
    print(f"  Val:   {len(y_val)} samples ({len(y_val)/len(y)*100:.1f}%)")
    print(f"  Test:  {len(y_test)} samples ({len(y_test)/len(y)*100:.1f}%)")
    
    print(f"\nClass distribution (0=SELL, 1=HOLD, 2=BUY):")
    print(f"  Train: {class_distribution(y_train)}")
    print(f"  Val:   {class_distribution(y_val)}")
    print(f"  Test:  {class_distribution(y_test)}")
    
    # Save arrays for reproducibility
    np.save(processed_dir / "X_train.npy", X_train)
    np.save(processed_dir / "y_train.npy", y_train)
    np.save(processed_dir / "X_val.npy", X_val)
    np.save(processed_dir / "y_val.npy", y_val)
    np.save(processed_dir / "X_test.npy", X_test)
    np.save(processed_dir / "y_test.npy", y_test)
    
    # =========================================================================
    # STAGE 5: BASELINE MODELS
    # =========================================================================
    print("\n" + "="*80)
    print("STAGE 5: BASELINE MODELS")
    print("="*80)
    
    print("\nTraining baseline models...")
    baseline_predictions = run_all_baselines(
        X_train, y_train, X_test, random_state=42
    )
    
    baseline_results = {}
    
    for name, y_pred in baseline_predictions.items():
        m = compute_metrics(y_test, y_pred)
        baseline_results[name] = {
            "accuracy": m.accuracy,
            "macro_f1": m.macro_f1,
            "confusion": m.confusion,
        }
        
        # Save confusion matrix
        save_confusion_matrix(
            m.confusion,
            title=f"Confusion Matrix - {name.replace('_', ' ').title()}",
            out_path=figures_dir / f"confusion_baseline_{name}.png",
        )
        
        print(f"  {name:20s}: acc={m.accuracy:.3f}, macro_f1={m.macro_f1:.3f}")
    
    # Save baseline results
    with open(outputs_dir / "baselines_metrics.json", "w") as f:
        json.dump(baseline_results, f, indent=2)
    
    # =========================================================================
    # STAGE 6: FEATURE ANALYSIS
    # =========================================================================
    print("\n" + "="*80)
    print("STAGE 6: FEATURE ANALYSIS")
    print("="*80)
    
    create_analysis_report(
        df_features=df_feat,
        feature_cols=feature_cols,
        X_train=X_train,
        y_train=y_train,
        figures_dir=figures_dir,
    )
    
    # =========================================================================
    # STAGE 7: DEEP NEURAL NETWORK TRAINING
    # =========================================================================
    print("\n" + "="*80)
    print("STAGE 7: DEEP NEURAL NETWORK TRAINING")
    print("="*80)
    
    print(f"\nTraining DNN with {len(SEEDS)} different seeds...")
    print(f"Architecture: Linear(128) → LSTM(128) → LSTM(128) → Linear(128) → Policy(3)")
    
    dnn_results = []
    
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        set_global_seed(seed)
        
        # Initialize model
        model = PolicyNetwork(
            input_dim=X_train.shape[2],
            hidden_dim=128,
            num_actions=3,
            dropout=0.1
        )
        
        # Training configuration
        cfg = TrainConfig(
            max_epochs=100,
            batch_size=32,
            lr=5e-4,
            weight_decay=1e-4,
            patience=30,
            time_limit_seconds=14 * 60,  # 14 minutes (safety margin)
        )
        
        # Train
        save_path = outputs_dir / f"best_policy_seed{seed}.pt"
        train_info = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            cfg=cfg,
            save_path=save_path,
        )
        
        # Load best weights and evaluate on test
        state = torch.load(save_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        
        y_test_pred = _predict(model, X_test)
        m_test = compute_metrics(y_test, y_test_pred)
        
        # Store results
        result = {
            "seed": seed,
            "ran_seconds": train_info["ran_seconds"],
            "best_epoch": train_info["best_epoch"],
            "val_best_macro_f1": train_info["best_val_macro_f1"],
            "test_accuracy": m_test.accuracy,
            "test_macro_f1": m_test.macro_f1,
            "confusion": m_test.confusion,
            "history": train_info["history"],
        }
        dnn_results.append(result)
        
        # Save confusion matrix
        save_confusion_matrix(
            m_test.confusion,
            title=f"Confusion Matrix - DNN Policy (Seed {seed})",
            out_path=figures_dir / f"confusion_dnn_seed{seed}.png",
        )
        
        # Save training curves
        plot_training_history(
            train_info["history"],
            output_path=figures_dir / f"training_curves_seed{seed}.png",
        )
        
        print(f"\nResults:")
        print(f"  Best val F1: {train_info['best_val_macro_f1']:.3f}")
        print(f"  Test acc:    {m_test.accuracy:.3f}")
        print(f"  Test F1:     {m_test.macro_f1:.3f}")
        print(f"  Time:        {train_info['ran_seconds']:.1f}s")
    
    # =========================================================================
    # STAGE 7.5: GROUP ABLATION (INFERENCE-TIME)  [for T7.3]
    # =========================================================================
    print("\n" + "-"*80)
    print("STAGE 7.5: GROUP ABLATION (INFERENCE-TIME)")
    print("-"*80)

    # Choose one trained seed to ablate (seed 0 is fine; keep it deterministic)
    ablation_seed = SEEDS[0]
    ablation_ckpt = outputs_dir / f"best_policy_seed{ablation_seed}.pt"

    # Recreate model and load checkpoint
    ablation_model = PolicyNetwork(
        input_dim=X_train.shape[2],
        hidden_dim=128,
        num_actions=3,
        dropout=0.1
    )
    state = torch.load(ablation_ckpt, map_location="cpu", weights_only=True)
    ablation_model.load_state_dict(state)

    # Build feature groups
    groups = analyze_feature_groups(df_feat, feature_cols)

    # Predict wrapper so analysis doesn't depend on torch
    def predict_wrapper(X_np: np.ndarray) -> np.ndarray:
        return _predict(ablation_model, X_np)

    # Run inference-time group ablation on test set
    df_ab = run_group_ablation_inference(
        X_test=X_test,
        y_test=y_test,
        feature_cols=feature_cols,
        groups=groups,
        predict_fn=predict_wrapper,
    )

    # Save artefacts
    df_ab.to_csv(outputs_dir / "ablation_groups.csv", index=False)
    plot_group_ablation(df_ab, figures_dir / "ablation_groups.png")

    print("Saved ablation table to outputs/ablation_groups.csv")
    print("Saved ablation plot  to figures/ablation_groups.png")
    print(df_ab)

    # =========================================================================
    # STAGE 8: RESULTS SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("STAGE 8: RESULTS SUMMARY")
    print("="*80)
    
    # Compute DNN statistics across seeds
    test_accs = [r["test_accuracy"] for r in dnn_results]
    test_f1s = [r["test_macro_f1"] for r in dnn_results]
    
    mean_acc = np.mean(test_accs)
    std_acc = np.std(test_accs, ddof=1)
    mean_f1 = np.mean(test_f1s)
    std_f1 = np.std(test_f1s, ddof=1)
    
    # 95% confidence intervals across seeds (n=3 -> df=2)
    n = len(test_f1s)
    tcrit = 4.303  # t_{0.975, df=2}
    ci_acc = tcrit * (std_acc / np.sqrt(n))
    ci_f1 = tcrit * (std_f1 / np.sqrt(n))

    print("\n=== BASELINE RESULTS ===")
    for name, res in baseline_results.items():
        print(f"{name:20s}: acc={res['accuracy']:.3f}, f1={res['macro_f1']:.3f}")
    
    print("\n=== DNN RESULTS (per seed) ===")
    for r in dnn_results:
        print(f"Seed {r['seed']}: acc={r['test_accuracy']:.3f}, f1={r['test_macro_f1']:.3f}")
    
    print("\n=== DNN AGGREGATE ===")
    print(f"Mean ± Std: acc={mean_acc:.3f}±{std_acc:.3f}, f1={mean_f1:.3f}±{std_f1:.3f}")
    print(f"95% CI:     acc=±{ci_acc:.3f}, f1=±{ci_f1:.3f}")
    

    # Best baseline for comparison
    best_baseline_f1 = max(res["macro_f1"] for res in baseline_results.values())
    improvement = mean_f1 - best_baseline_f1
    
    print(f"\n=== IMPROVEMENT OVER BEST BASELINE ===")
    print(f"Best baseline F1: {best_baseline_f1:.3f}")
    print(f"DNN mean F1:      {mean_f1:.3f}")
    print(f"Improvement:      {improvement:+.3f} ({improvement/best_baseline_f1*100:+.1f}%)")
    
    # Save all results
    with open(outputs_dir / "dnn_results.json", "w") as f:
        json.dump(dnn_results, f, indent=2)
    
    # Save summary
    summary = {
        "dataset": {
            "temporal_scope": f"{START_DATE} to {END_DATE}",
            "total_samples": int(len(y)),
            "train_samples": int(len(y_train)),
            "val_samples": int(len(y_val)),
            "test_samples": int(len(y_test)),
            "num_features": len(feature_cols),
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "threshold": THRESHOLD,
        },
        "baselines": baseline_results,
        "dnn": {
            "mean_test_accuracy": float(mean_acc),
            "std_test_accuracy": float(std_acc),
            "ci95_test_accuracy": float(ci_acc),
            "mean_test_f1": float(mean_f1),
            "std_test_f1": float(std_f1),
            "ci95_test_f1": float(ci_f1),
            "seeds": SEEDS,
            "per_seed_results": [
                {k: v for k, v in r.items() if k != "history"}
                for r in dnn_results
            ],
        },
        "improvement_over_baseline": {
            "best_baseline_f1": float(best_baseline_f1),
            "dnn_mean_f1": float(mean_f1),
            "absolute_improvement": float(improvement),
            "relative_improvement_pct": float(improvement / best_baseline_f1 * 100),
        },
    }
    
    with open(outputs_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Close database
    db.close()
    
    print("\n" + "="*80)
    print("PIPELINE FINISHED SUCCESSFULLY")
    print("="*80)
    print(f"\nOutputs saved to:")
    print(f"  - Metrics: {outputs_dir}")
    print(f"  - Figures: {figures_dir}")
    print(f"  - Database: {data_dir / 'trading.db'}")
    print(f"  - Processed data: {processed_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()