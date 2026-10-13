from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.metrics import compute_metrics


# -------------------------------------------------
# Training configuration
# -------------------------------------------------
@dataclass
class TrainConfig:
    max_epochs: int = 100  # UPDATED: Was 50, now 100 for more training
    batch_size: int = 64
    lr: float = 1e-3  
    weight_decay: float = 1e-5
    patience: int = 30  
    time_limit_seconds: int = 14 * 60  # 14 minutes (safety margin)
    eval_batch_size: int = 256
    print_every: int = 5  # Print training progress every N epochs


# -------------------------------------------------
# Inference helper
# -------------------------------------------------
def _predict(model: nn.Module, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Run batched inference and return argmax class predictions."""
    model.eval()
    device = next(model.parameters()).device

    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    preds: list[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            logits = model(xb)
            yhat = torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(yhat)

    return np.concatenate(preds, axis=0)


# -------------------------------------------------
# Loss helper (class imbalance)
# -------------------------------------------------
def _make_class_weighted_loss(
    y_train: np.ndarray, num_classes: int = 3
) -> nn.Module:
    """
    Create inverse-frequency class-weighted cross entropy.
    
    This addresses class imbalance by penalizing errors on minority classes more.
    Formula: weight[c] = total_samples / (num_classes * count[c])
    """
    counts = np.bincount(y_train.astype(int), minlength=num_classes).astype(np.float32)
    
    # Prevent division by zero
    counts = np.maximum(counts, 1.0)
    
    # Inverse frequency weighting
    total = counts.sum()
    weights = total / (num_classes * counts)
    
    print(f"Class counts: {counts}")
    print(f"Class weights: {weights}")
    
    return nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32)
    )


# -------------------------------------------------
# Training loop with detailed diagnostics
# -------------------------------------------------
def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
    save_path: Path,
) -> dict[str, Any]:
    """
    Train a policy network with:
      - class-weighted cross-entropy
      - early stopping on validation macro-F1
      - hard time cutoff (< 15 minutes)
      - detailed training diagnostics

    Best model weights are saved to `save_path`.
    Returns training history and metrics.
    """
    device = torch.device("cpu")
    model.to(device)

    # Dataset
    ds_train = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    dl_train = DataLoader(
        ds_train, batch_size=cfg.batch_size, shuffle=True
    )

    # Optimizer and loss
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    loss_fn = _make_class_weighted_loss(y_train).to(device)

    # Learning rate scheduler - reduces LR when validation plateaus
    scheduler = ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=5
    )

    best_val_f1 = -1.0
    best_epoch = -1
    no_improve = 0
    start_time = time.time()

    history: list[dict[str, float]] = []

    print("\nStarting training...")
    print(f"Train samples: {len(y_train)}, Val samples: {len(y_val)}")
    print(f"Max epochs: {cfg.max_epochs}, Patience: {cfg.patience}")
    print(f"Batch size: {cfg.batch_size}, Learning rate: {cfg.lr}")

    for epoch in range(1, cfg.max_epochs + 1):
        # Hard time cutoff
        elapsed = time.time() - start_time
        if elapsed > cfg.time_limit_seconds:
            print(f"\nTime limit reached ({elapsed:.1f}s). Stopping training.")
            break

        # ---------------------
        # Train epoch
        # ---------------------
        model.train()
        running_loss = 0.0
        num_batches = 0

        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            opt.step()

            running_loss += float(loss.item()) * len(yb)
            num_batches += 1

        train_loss = running_loss / len(ds_train)

        # ---------------------
        # Validation
        # ---------------------
        y_val_pred = _predict(model, X_val, batch_size=cfg.eval_batch_size)
        m_val = compute_metrics(y_val, y_val_pred)

        # Also get training metrics to check for overfitting
        y_train_pred = _predict(model, X_train, batch_size=cfg.eval_batch_size)
        m_train = compute_metrics(y_train, y_train_pred)

        history.append({
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "train_accuracy": float(m_train.accuracy),
            "train_macro_f1": float(m_train.macro_f1),
            "val_accuracy": float(m_val.accuracy),
            "val_macro_f1": float(m_val.macro_f1),
        })

        # Print progress
        if epoch % cfg.print_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{cfg.max_epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Train F1: {m_train.macro_f1:.3f} | "
                f"Val F1: {m_val.macro_f1:.3f} | "
                f"Time: {elapsed:.1f}s"
            )

        # ---------------------
        # Early stopping
        # ---------------------
        if m_val.macro_f1 > best_val_f1 + 1e-6:
            best_val_f1 = float(m_val.macro_f1)
            best_epoch = epoch
            no_improve = 0

            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={cfg.patience})")
                break
        
        # Update learning rate based on validation performance
        scheduler.step(m_val.macro_f1)

    total_time = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"Best val F1: {best_val_f1:.3f} at epoch {best_epoch}")
    print(f"Total time: {total_time:.1f}s")

    return {
        "best_val_macro_f1": best_val_f1,
        "best_epoch": best_epoch,
        "history": history,
        "ran_seconds": float(total_time),
    }