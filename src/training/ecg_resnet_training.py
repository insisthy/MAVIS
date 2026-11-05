"""Original ResNet1D training implementation; portable caller supplies input paths."""
from __future__ import annotations

import argparse

import json

import sys

from dataclasses import asdict, dataclass

from pathlib import Path

from typing import Any

import numpy as np

import pandas as pd

import torch

import wfdb

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from torch import nn

from torch.utils.data import DataLoader

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]

from src.data_processing.waveform_dataset import WaveformConfig, preprocess_waveform

from src.models.resnet1d_small import ResNet1DSmall

OUTPUT_ROOT = ROOT / "outputs"

MODEL_DIR = OUTPUT_ROOT / "ecg_model"

PREDICTION_DIR = OUTPUT_ROOT / "predictions"

SELECTION_DIR = OUTPUT_ROOT / "selection"

LOG_DIR = OUTPUT_ROOT / "logs"

INDEX_PATH = ROOT / "data" / "ecg_index.csv"

LABEL_PATH = ROOT / "data" / "label_split_table.csv"

EXPECTED_SPLIT_COUNTS = {"train": 1628, "val": 465, "test": 233}

@dataclass
class ScratchWaveformConfig:
    fs_hz: int = 500
    n_leads: int = 12
    n_samples: int = 5000
    batch_size: int = 16
    max_epochs: int = 30
    early_stopping_patience: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.3
    num_workers: int = 0
    use_amp: bool = True
    random_seed: int = 42
    waveform_base_dir: str = "data/raw"
    grad_clip_norm: float = 1.0
    threshold_grid_size: int = 199


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))

def resolve_wave_path(value: object) -> Path:
    raw = Path(str(value))
    candidates = [raw] if raw.is_absolute() else [ROOT / "data/raw" / raw, ROOT / raw, raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()

def build_manifest() -> pd.DataFrame:
    labels = pd.read_csv(LABEL_PATH, encoding="utf-8-sig", low_memory=False)
    index = pd.read_csv(INDEX_PATH, encoding="utf-8-sig", low_memory=False)
    for frame in (labels, index):
        frame["admission_id"] = frame["admission_id"].astype(str)
        frame["waveform_id"] = frame["waveform_id"].astype(str)
    manifest = labels.merge(
        index,
        on=["admission_id", "waveform_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_index"),
    )
    counts = manifest["split"].value_counts().to_dict()
    if counts != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(f"Split counts do not match Dataset-v5: {counts}")
    manifest["hea_path_abs"] = manifest["hea_path"].map(resolve_wave_path)
    manifest["dat_path_abs"] = manifest["dat_path"].map(resolve_wave_path)
    return manifest

class V29WaveformDataset(torch.utils.data.Dataset):
    def __init__(self, manifest: pd.DataFrame, split: str, config: ScratchWaveformConfig):
        self.frame = manifest[manifest["split"].eq(split)].reset_index(drop=True)
        self.config = config
        self.model_input_columns = ["waveform_tensor"]

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        record_base = Path(str(row["hea_path_abs"])).with_suffix("")
        record = wfdb.rdrecord(str(record_base))
        signal = record.p_signal if record.p_signal is not None else record.d_signal
        if signal is None:
            raise ValueError(f"No WFDB signal found: {record_base}")
        signal = np.asarray(signal, dtype=np.float32).T
        signal, _ = preprocess_waveform(
            signal,
            WaveformConfig(
                fs_hz=self.config.fs_hz,
                n_leads=self.config.n_leads,
                n_samples=self.config.n_samples,
                waveform_base_dir=self.config.waveform_base_dir,
            ),
        )
        return {
            "waveform": torch.from_numpy(signal),
            "label": torch.tensor(float(row["Target_CAG_Binary"]), dtype=torch.float32),
            "admission_id": str(row["admission_id"]),
            "waveform_id": str(row["waveform_id"]),
            "split": str(row["split"]),
        }

def build_loaders(
    manifest: pd.DataFrame,
    config: ScratchWaveformConfig,
) -> dict[str, DataLoader]:
    return {
        split: DataLoader(
            V29WaveformDataset(manifest, split, config),
            batch_size=config.batch_size,
            shuffle=split == "train",
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split in ("train", "val")
    }

def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"v2.9 Predict {split}", total=len(loader)):
            waveforms = batch["waveform"].to(device, non_blocking=True)
            logits = model(waveforms).detach().cpu().numpy()
            scores = sigmoid(logits)
            labels = batch["label"].detach().cpu().numpy().astype(int)
            for index in range(len(logits)):
                rows.append(
                    {
                        "admission_id": batch["admission_id"][index],
                        "waveform_id": batch["waveform_id"][index],
                        "split": split,
                        "y_true": int(labels[index]),
                        "y_score": float(scores[index]),
                        "logit": float(logits[index]),
                        "model_name": "ResNet1D_raw_waveform",
                    }
                )
    return pd.DataFrame(rows)

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
    epoch: int,
    total_epochs: int,
    grad_clip_norm: float,
) -> float:
    model.train()
    losses = []
    progress = tqdm(loader, desc=f"Train epoch {epoch}/{total_epochs}", total=len(loader))
    for batch in progress:
        waveforms = batch["waveform"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(waveforms)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
        progress.set_postfix(train_loss=np.mean(losses), lr=optimizer.param_groups[0]["lr"])
    return float(np.mean(losses))

def select_balanced_accuracy_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    grid_size: int,
) -> float:
    candidates = np.unique(np.concatenate([np.linspace(0.01, 0.99, grid_size), y_score]))
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in candidates:
        prediction = (y_score >= threshold).astype(int)
        score = balanced_accuracy_score(y_true, prediction)
        if score > best_score or (
            np.isclose(score, best_score)
            and abs(threshold - 0.5) < abs(best_threshold - 0.5)
        ):
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold

def ece_score(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (y_score >= lower) & (y_score < upper if upper < 1.0 else y_score <= upper)
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - y_score[mask].mean())
    return float(ece)

def compute_metrics(y_true: np.ndarray, logits: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = y_true.astype(int)
    scores = sigmoid(logits.astype(float))
    prediction = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "split": "val",
        "n": int(len(y_true)),
        "auc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "sensitivity": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "ppv": float(precision_score(y_true, prediction, zero_division=0)),
        "npv": float(tn / (tn + fn)) if (tn + fn) else 0.0,
        "brier": float(brier_score_loss(y_true, scores)),
        "ece": ece_score(y_true, scores),
        "threshold": float(threshold),
        "threshold_source": "val_balanced_accuracy",
    }


def train(config: ScratchWaveformConfig, class_weight_strategy: str) -> dict[str, Any]:
    for directory in (MODEL_DIR, PREDICTION_DIR, SELECTION_DIR, LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    loaders = build_loaders(manifest, config)
    set_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet1DSmall(in_channels=config.n_leads, dropout=config.dropout).to(device)
    if class_weight_strategy == "pos_weight":
        train_frame = manifest[manifest["split"].eq("train")]
        positives = int(train_frame["Target_CAG_Binary"].sum())
        negatives = int(len(train_frame) - positives)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([negatives / positives], dtype=torch.float32, device=device)
        )
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    use_amp = bool(config.use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_auc = -np.inf
    best_epoch = 0
    early_counter = 0
    history: list[dict[str, Any]] = []
    stem = f"v2_9_resnet1d_waveform_{class_weight_strategy}_ResNet1D_raw_waveform"
    best_path = MODEL_DIR / f"{stem}_best.pt"

    for epoch in range(1, config.max_epochs + 1):
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            device,
            criterion,
            optimizer,
            scaler,
            use_amp,
            epoch,
            config.max_epochs,
            config.grad_clip_norm,
        )
        validation = predict(model, loaders["val"], device, "val")
        auc = float(roc_auc_score(validation["y_true"], validation["y_score"]))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_auc": auc})
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            early_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "class_weight_strategy": class_weight_strategy,
                    "best_epoch": best_epoch,
                },
                best_path,
            )
        else:
            early_counter += 1
        pd.DataFrame(history).to_csv(
            LOG_DIR / f"{stem}_training_history.csv",
            index=False,
            encoding="utf-8-sig",
        )
        if early_counter >= config.early_stopping_patience:
            break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation = predict(model, loaders["val"], device, "val")
    threshold = select_balanced_accuracy_threshold(
        validation["y_true"].to_numpy(dtype=int),
        validation["y_score"].to_numpy(dtype=float),
        config.threshold_grid_size,
    )
    metrics = compute_metrics(
        validation["y_true"].to_numpy(dtype=int),
        validation["logit"].to_numpy(dtype=float),
        threshold,
    )
    metrics.update(
        {
            "dataset_version": "v2_9",
            "task": "resnet1d_waveform",
            "model_name": "ResNet1D_raw_waveform",
            "class_weight_strategy": class_weight_strategy,
            "best_epoch": best_epoch,
            "test_used": False,
            "test_metrics_computed": False,
        }
    )
    validation["selected_threshold"] = threshold
    prediction_path = PREDICTION_DIR / f"{stem}_val_predictions.csv"
    metrics_path = SELECTION_DIR / f"{stem}_validation_metrics.csv"
    registry_path = SELECTION_DIR / f"{stem}_training_registry.csv"
    validation.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "dataset_version": "v2_9",
                "task": "resnet1d_waveform",
                "model_name": "ResNet1D_raw_waveform",
                "class_weight_strategy": class_weight_strategy,
                "model_artifact": str(best_path),
                "prediction_file": str(prediction_path),
                "validation_metrics_file": str(metrics_path),
                "test_used": False,
                "test_metrics_computed": False,
            }
        ]
    ).to_csv(registry_path, index=False, encoding="utf-8-sig")
    return {
        "dataset_version": "v2_9",
        "model_training_run": True,
        "train_rows": EXPECTED_SPLIT_COUNTS["train"],
        "validation_rows": len(validation),
        "best_epoch": best_epoch,
        "best_validation_auc": best_auc,
        "validation_selected_threshold": threshold,
        "model_artifact": str(best_path),
        "validation_predictions": str(prediction_path),
        "test_used": False,
        "test_metrics_computed": False,
    }

