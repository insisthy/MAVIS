from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.io import read_csv_auto


ADMISSION_COL = "admission_id"
EXPECTED_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FORBIDDEN_WAVEFORM_INPUT_COLUMNS = {
    ADMISSION_COL,
    "waveform_id",
    "record_id",
    "patient_id",
    "inpat_id",
    "source_record_id",
    "CAG_label",
    "Target_CAG_Binary",
    "group_label",
    "fp_task_label",
    "fn_task_label",
    "split",
    "final_ecg_ischemia_label",
    "ECG_ischemia_label",
    "ECG_Ischemic_Phenotype",
}


@dataclass(frozen=True)
class WaveformConfig:
    fs_hz: int = 500
    n_leads: int = 12
    n_samples: int = 5000
    normalize: str = "per_record_per_lead_zscore"
    waveform_base_dir: str = "data/raw"


def build_waveform_manifest(
    ecg_index_path: str | Path,
    label_path: str | Path,
    root_dir: str | Path,
    waveform_base_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Join labels with ECG waveform index without modifying source tables."""
    root = Path(root_dir)
    base_dir = Path(waveform_base_dir)
    index = read_csv_auto(ecg_index_path)
    labels = read_csv_auto(label_path)
    if "waveform_id" not in labels.columns or "waveform_id" not in index.columns:
        raise ValueError("Both label and ECG index tables must contain waveform_id.")
    required_label = [ADMISSION_COL, "waveform_id", "Target_CAG_Binary", "split", "group_label"]
    missing = [c for c in required_label if c not in labels.columns]
    if missing:
        raise ValueError(f"Missing required label columns: {missing}")
    index_cols = [
        c
        for c in [
            ADMISSION_COL,
            "waveform_id",
            "record_name",
            "hea_path",
            "dat_path",
            "fs_hz",
            "n_leads",
            "sig_len",
            "lead_order_json",
            "basic_qc_pass_flag",
            "large_amplitude_warning_flag",
            "uncommon_sig_len_flag",
            "conversion_status",
        ]
        if c in index.columns
    ]
    label_cols = [c for c in required_label if c in labels.columns]
    merged = labels[label_cols].merge(index[index_cols], on="waveform_id", how="left", suffixes=("", "_index"))
    if f"{ADMISSION_COL}_index" in merged.columns:
        merged = merged.drop(columns=[f"{ADMISSION_COL}_index"])
    def resolve_wave_path(value: object) -> tuple[str, str]:
        if pd.isna(value):
            return "", "missing"
        raw = Path(str(value))
        candidates = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            # Primary convention: ECG_index stores paths like WFDB/foo.hea,
            # and waveform_base_dir points to data/raw, so resolve to
            # project_root/data/raw/WFDB/foo.hea.
            candidates.extend(
                [
                    root / base_dir / raw,
                    root / raw,
                    raw,
                ]
            )
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve()), str(candidate)
        return (str(candidates[0].resolve()), "not_found:" + " | ".join(str(c) for c in candidates)) if candidates else ("", "missing")

    for col in ["hea_path", "dat_path"]:
        if col in merged.columns:
            resolved = merged[col].map(resolve_wave_path)
            merged[f"{col}_abs"] = resolved.map(lambda item: item[0])
            merged[f"{col}_resolution_source"] = resolved.map(lambda item: item[1])
    merged["waveform_base_dir"] = str(base_dir)
    return merged


def wfdb_record_base(row: pd.Series) -> Path:
    hea = Path(str(row.get("hea_path_abs") or row.get("hea_path") or ""))
    if hea.suffix.lower() == ".hea":
        return hea.with_suffix("")
    return hea


def read_wfdb_signal(row: pd.Series) -> tuple[np.ndarray, dict[str, Any]]:
    import wfdb

    record_base = wfdb_record_base(row)
    record = wfdb.rdrecord(str(record_base))
    signal = record.p_signal if record.p_signal is not None else record.d_signal
    if signal is None:
        raise ValueError(f"No signal array in WFDB record: {record_base}")
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D WFDB signal, got shape={arr.shape}")
    # WFDB returns [samples, leads]; model expects [leads, samples].
    arr = arr.T
    meta = {
        "fs_hz_read": getattr(record, "fs", None),
        "n_leads_read": arr.shape[0],
        "sig_len_read": arr.shape[1],
        "sig_name_read": list(getattr(record, "sig_name", []) or []),
    }
    return arr, meta


def preprocess_waveform(signal: np.ndarray, config: WaveformConfig = WaveformConfig()) -> tuple[np.ndarray, list[str]]:
    warnings: list[str] = []
    x = np.asarray(signal, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected [leads, samples], got shape={x.shape}")
    if x.shape[0] != config.n_leads:
        warnings.append(f"n_leads_adjusted:{x.shape[0]}->{config.n_leads}")
        if x.shape[0] > config.n_leads:
            x = x[: config.n_leads, :]
        else:
            pad = np.zeros((config.n_leads - x.shape[0], x.shape[1]), dtype=np.float32)
            x = np.concatenate([x, pad], axis=0)
    if x.shape[1] != config.n_samples:
        warnings.append(f"sig_len_adjusted:{x.shape[1]}->{config.n_samples}")
        if x.shape[1] > config.n_samples:
            start = max((x.shape[1] - config.n_samples) // 2, 0)
            x = x[:, start : start + config.n_samples]
        else:
            pad = np.zeros((x.shape[0], config.n_samples - x.shape[1]), dtype=np.float32)
            x = np.concatenate([x, pad], axis=1)
    if not np.isfinite(x).all():
        warnings.append("nan_or_inf_replaced")
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if config.normalize == "per_record_per_lead_zscore":
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        x = (x - mean) / std
    return x.astype(np.float32), warnings


class ECGWaveformDataset(Dataset):
    """Dataset returning waveform tensor and labels; labels are never part of model input."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str | None = None,
        config: WaveformConfig = WaveformConfig(),
        reader: Callable[[pd.Series], tuple[np.ndarray, dict[str, Any]]] = read_wfdb_signal,
    ) -> None:
        self.config = config
        self.reader = reader
        frame = manifest.copy()
        if split is not None:
            frame = frame[frame["split"] == split].copy()
        self.frame = frame.reset_index(drop=True)
        self.model_input_columns = ["waveform_tensor"]

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        signal, meta = self.reader(row)
        signal, warnings = preprocess_waveform(signal, self.config)
        return {
            "waveform": torch.from_numpy(signal),
            "label": torch.tensor(float(row["Target_CAG_Binary"]), dtype=torch.float32),
            "admission_id": str(row.get(ADMISSION_COL, "")),
            "waveform_id": str(row.get("waveform_id", "")),
            "split": str(row.get("split", "")),
            "warning_text": ";".join(warnings),
            "read_shape": str((meta.get("n_leads_read"), meta.get("sig_len_read"))),
        }

    def forbidden_columns_in_model_input(self) -> list[str]:
        return sorted(set(self.model_input_columns) & FORBIDDEN_WAVEFORM_INPUT_COLUMNS)
