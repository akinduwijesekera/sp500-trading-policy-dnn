from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_feature_correlations(
    df: pd.DataFrame,
    feature_cols: list[str],
    output_path: Path,
    figsize: tuple = (16, 14),
) -> None:
    """
    Create correlation heatmap of features.

    Purpose: Identify redundant/correlated features and feature relationships.

    Interpretation:
    - High correlation (>0.8): Features may be redundant
    - Negative correlation: Inverse relationships (e.g., VIX vs SPX)
    - Orthogonal features (correlation ~0): Independent information sources

    This analysis helps answer T7.2:
    "How are features related to each other? Are there correlated features?"
    """
    plt.figure(figsize=figsize)

    # Compute correlation matrix
    corr_matrix = df[feature_cols].corr()

    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=False,  # Too many features to annotate
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )

    plt.title("Feature Correlation Matrix", fontsize=16, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved feature correlation heatmap to {output_path}")

    # Print highly correlated pairs for report
    print("\nHighly correlated feature pairs (|r| > 0.7):")
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                print(f"  {corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {corr_val:.3f}")


def analyze_feature_groups(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, list[str]]:
    """
    Group features by category for structured analysis.

    Returns dictionary mapping category -> list of features
    """
    groups: dict[str, list[str]] = {
        "Returns": [],
        "Moving Averages": [],
        "Volatility": [],
        "Volume": [],
        "Technical Oscillators": [],
        "External Indicators": [],
        "Calendar": [],
    }

    for feat in feature_cols:
        if "ret" in feat or "ROC" in feat:
            groups["Returns"].append(feat)
        elif "MA" in feat or "price_to" in feat:
            groups["Moving Averages"].append(feat)
        elif "vol" in feat or "BB" in feat or "HL_range" in feat:
            groups["Volatility"].append(feat)
        elif "volume" in feat or "OBV" in feat:
            groups["Volume"].append(feat)
        elif "RSI" in feat or "MACD" in feat:
            groups["Technical Oscillators"].append(feat)
        elif "VIX" in feat or "Treasury" in feat or "Gold" in feat:
            groups["External Indicators"].append(feat)
        elif "day" in feat or "month" in feat:
            groups["Calendar"].append(feat)

    return groups


def plot_pca_visualization(
    X: np.ndarray,
    y: np.ndarray,
    output_path: Path,
    n_components: int = 2,
    figsize: tuple = (10, 8),
) -> None:
    """
    Create 2D PCA visualization of feature space colored by label.

    Purpose: Visualize if classes are separable in feature representation.

    This addresses T7.1: "Is there any pattern in the representation?"

    Interpretation:
    - Clear clusters → Features capture class differences well
    - Overlapping clusters → Classes not linearly separable, need non-linear model
    - Outliers → Potential data quality issues or rare events

    Args:
        X: Feature array (N, lookback, features) - will be flattened
        y: Labels (N,)
        output_path: Where to save figure
    """
    # Flatten sequences for PCA
    if X.ndim == 3:
        X_flat = X.reshape(X.shape[0], -1)
    else:
        X_flat = X

    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_flat)

    # Plot
    plt.figure(figsize=figsize)

    colors = {0: "red", 1: "gray", 2: "green"}
    labels_map = {0: "SELL", 1: "HOLD", 2: "BUY"}

    for label_val in [0, 1, 2]:
        mask = (y == label_val)
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=colors[label_val],
            label=labels_map[label_val],
            alpha=0.6,
            s=20,
        )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
    plt.title("PCA: 2D Feature Space Visualization", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved PCA visualization to {output_path}")
    print(
        f"Explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, "
        f"PC2={pca.explained_variance_ratio_[1]:.1%}"
    )


def plot_tsne_visualization(
    X: np.ndarray,
    y: np.ndarray,
    output_path: Path,
    perplexity: int = 30,
    figsize: tuple = (10, 8),
) -> None:
    """
    Create 2D t-SNE visualization of feature space.

    Purpose: Non-linear dimensionality reduction to reveal cluster structure.
    t-SNE is better than PCA at revealing non-linear separability.

    Args:
        X: Feature array (N, lookback, features)
        y: Labels (N,)
        output_path: Where to save figure
        perplexity: t-SNE perplexity parameter (typically 5-50)
    """
    # Flatten sequences
    if X.ndim == 3:
        X_flat = X.reshape(X.shape[0], -1)
    else:
        X_flat = X

    # Sample if too large (t-SNE is slow)
    if len(X_flat) > 2000:
        indices = np.random.choice(len(X_flat), 2000, replace=False)
        X_flat = X_flat[indices]
        y = y[indices]
        print(f"Sampled 2000 points for t-SNE (original: {len(X_flat)})")

    # Apply t-SNE
    print("Running t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_flat)

    # Plot
    plt.figure(figsize=figsize)

    colors = {0: "red", 1: "gray", 2: "green"}
    labels_map = {0: "SELL", 1: "HOLD", 2: "BUY"}

    for label_val in [0, 1, 2]:
        mask = (y == label_val)
        plt.scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            c=colors[label_val],
            label=labels_map[label_val],
            alpha=0.6,
            s=20,
        )

    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.title("t-SNE: 2D Feature Space Visualization", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved t-SNE visualization to {output_path}")


def plot_training_history(
    history: list[dict],
    output_path: Path,
    figsize: tuple = (12, 5),
) -> None:
    """
    Plot training curves (loss and F1 over epochs).

    Purpose: Diagnose training issues (overfitting, underfitting, convergence).

    Interpretation:
    - Train loss decreasing, val loss increasing → Overfitting
    - Both losses high and flat → Underfitting / not learning
    - Both decreasing → Good training
    - Val F1 improving then plateau → Early stopping working correctly
    """
    if not history:
        print("No training history to plot")
        return

    df_hist = pd.DataFrame(history)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Loss curves
    axes[0].plot(df_hist["epoch"], df_hist["train_loss"], label="Train Loss", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training Loss Over Time", fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot 2: F1 scores
    axes[1].plot(df_hist["epoch"], df_hist["train_macro_f1"], label="Train F1", linewidth=2)
    axes[1].plot(df_hist["epoch"], df_hist["val_macro_f1"], label="Val F1", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Macro F1", fontsize=12)
    axes[1].set_title("F1 Score Over Time", fontsize=14, fontweight="bold")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved training curves to {output_path}")


# =============================================================================
# REAL (RUBRIC-VALID) ABLATION: inference-time feature group ablation
# =============================================================================

def ablate_feature_group(
    X: np.ndarray,
    feature_cols: list[str],
    group_feats: list[str],
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Return a copy of X with selected feature columns set to fill_value
    across all timesteps.

    X expected shape: (N, L, F)
    """
    X2 = X.copy()
    idx = [feature_cols.index(f) for f in group_feats if f in feature_cols]
    if not idx:
        return X2
    X2[:, :, idx] = fill_value
    return X2


def run_group_ablation_inference(
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list[str],
    groups: dict[str, list[str]],
    predict_fn,
) -> pd.DataFrame:
    """
    Inference-time group ablation on the test set.

    Baseline: evaluate on X_test as-is.
    For each group: zero-out that group's columns and re-evaluate.

    predict_fn should be a callable: predict_fn(X_np) -> y_pred (N,)
    """
    # Local import to avoid circular dependencies
    from src.metrics import compute_metrics

    # Baseline
    y_pred_base = predict_fn(X_test)
    base = compute_metrics(y_test, y_pred_base)

    rows: list[dict] = []
    for gname, gfeats in groups.items():
        X_ab = ablate_feature_group(X_test, feature_cols, gfeats, fill_value=0.0)
        y_pred_ab = predict_fn(X_ab)
        m_ab = compute_metrics(y_test, y_pred_ab)

        rows.append(
            {
                "group": gname,
                "n_features": int(len(gfeats)),
                "baseline_accuracy": base.accuracy,
                "baseline_macro_f1": base.macro_f1,
                "ablated_accuracy": m_ab.accuracy,
                "ablated_macro_f1": m_ab.macro_f1,
                "delta_accuracy": base.accuracy - m_ab.accuracy,
                "delta_macro_f1": base.macro_f1 - m_ab.macro_f1,
            }
        )

    df_ab = pd.DataFrame(rows).sort_values("delta_macro_f1", ascending=False)
    return df_ab


def plot_group_ablation(
    df_ab: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (10, 4),
) -> None:
    """
    Plot macro-F1 drop per ablated group.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=figsize)
    plt.bar(df_ab["group"], df_ab["delta_macro_f1"])
    plt.ylabel("Macro-F1 drop (higher = more important)")
    plt.xticks(rotation=30, ha="right")
    plt.title("Inference-time Group Ablation (Test Set)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved group ablation plot to {output_path}")


def create_analysis_report(
    df_features: pd.DataFrame,
    feature_cols: list[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    figures_dir: Path,
) -> None:
    """
    Create comprehensive feature analysis report with key visualizations.

    This generates the core figures needed for T7 of the report:
    - feature correlation heatmap
    - PCA 2D plot

    (Optional) t-SNE can be enabled but is slow and not required.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("GENERATING FEATURE ANALYSIS REPORT")
    print("=" * 70)

    # 1. Feature correlation heatmap
    plot_feature_correlations(
        df_features,
        feature_cols,
        output_path=figures_dir / "feature_correlations.png",
    )

    # 2. PCA visualization
    plot_pca_visualization(
        X_train,
        y_train,
        output_path=figures_dir / "pca_2d.png",
    )

    # 3. t-SNE visualization (optional, slow)
    # plot_tsne_visualization(
    #     X_train,
    #     y_train,
    #     output_path=figures_dir / "tsne_2d.png"
    # )

    print("=" * 70)
    print("Feature analysis complete!")
    print("=" * 70)
