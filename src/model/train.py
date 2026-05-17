"""Training loop with early stopping and MLflow experiment tracking."""

import logging
from typing import Any

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from src.data.preprocess import load_split
from src.model.dataset import StockDataset
from src.model.lstm import AttentionLSTM, StackedLSTM

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "stacked_lstm": StackedLSTM,
    "attention_lstm": AttentionLSTM,
}


def _build_loaders(
    tickers: list[str],
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Concatenate all ticker splits into two DataLoaders (train, val)."""
    train_sets, val_sets = [], []
    for ticker in tickers:
        X_tr, y_tr = load_split(ticker, "train")
        X_val, y_val = load_split(ticker, "val")
        train_sets.append(StockDataset(X_tr, y_tr))
        val_sets.append(StockDataset(X_val, y_val))

    train_loader = DataLoader(ConcatDataset(train_sets), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ConcatDataset(val_sets), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """Run one epoch; returns mean loss. Pass optimizer=None for eval mode."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0

    with torch.set_grad_enabled(training):
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            loss = criterion(pred, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * len(X)

    return total_loss / len(loader.dataset)


def train(
    model_name: str,
    tickers: list[str],
    model_kwargs: dict[str, Any],
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 15,
) -> nn.Module:
    """Train a model, log to the active MLflow run, return the best model.

    The model is trained on the concatenated training set of all tickers and
    validated on the concatenated validation set. Early stopping monitors
    validation MSE loss with the given patience.
    """
    logger.info("Training %s on device=%s tickers=%s", model_name, DEVICE, tickers)

    model = MODEL_REGISTRY[model_name](**model_kwargs).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("%s has %d trainable parameters", model_name, n_params)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    train_loader, val_loader = _build_loaders(tickers, batch_size)

    mlflow.log_params(
        {
            "model": model_name,
            "n_params": n_params,
            "tickers": ",".join(tickers),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "patience": patience,
            **{f"arch_{k}": v for k, v in model_kwargs.items()},
        }
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_weights: dict = {}

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer)
        val_loss = _run_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            logger.info(
                "epoch %3d | train=%.6f | val=%.6f | best=%.6f | patience=%d/%d",
                epoch, train_loss, val_loss, best_val_loss, patience_counter, patience,
            )

        if patience_counter >= patience:
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    model.load_state_dict(best_weights)
    mlflow.log_metric("best_val_loss", best_val_loss)
    mlflow.pytorch.log_model(model, name="model")
    logger.info("Done. best_val_loss=%.6f", best_val_loss)
    return model
