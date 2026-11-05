"""Method-only extraction; study-specific commands and fixtures removed."""
from __future__ import annotations

import random

import os

import tempfile

from pathlib import Path

import numpy as np

import pandas as pd

import torch

import wfdb

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedGroupKFold

from torch.utils.data import DataLoader, Dataset

from src.data_processing.waveform_dataset import EXPECTED_LEADS, WaveformConfig, preprocess_waveform

from src.models.resnet1d_small import ResNet1DSmall

from .contracts import TAU_ECG, require, validate_method

from .runtime import Transaction, file_hash

from .oof_engineering import (Progress, build_binding, fold_binding, match,
                              preflight_fold, preflight_run, task, task_inputs, validate_predictions)

def snapshot(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

def save_checkpoint(path, value):
    require(not path.exists(), "CHECKPOINT_EXISTS")
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)

def inner_split(outer_train):
    require(outer_train.admission_id.is_unique, "DUPLICATE_TRAIN_PATIENT")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260703)
    fit, selection = next(splitter.split(outer_train, outer_train.group_label, groups=outer_train.admission_id))
    require(not set(fit) & set(selection) and set(fit) | set(selection) == set(range(len(outer_train))), "INNER_PARTITION_INVALID")
    return outer_train.iloc[fit].copy(), outer_train.iloc[selection].copy()

def strict_fold(train, holdout_features, select_epoch, refit, predict):
    """The callback boundary makes holdout outcomes inaccessible during all fitting."""
    require("Target_CAG_Binary" not in holdout_features and "y_true" not in holdout_features, "HOLDOUT_LABEL_VISIBLE")
    require(not set(train.admission_id) & set(holdout_features.admission_id), "OUTER_HOLDOUT_OVERLAP")
    fit, selection = inner_split(train)
    epoch = select_epoch(fit, selection)
    require(type(epoch) is int and epoch >= 1, "INVALID_SELECTED_EPOCH")
    model = refit(train.copy(), epoch)
    prediction = predict(model, holdout_features.copy())
    return prediction, epoch

class Waves(Dataset):
    def __init__(self, frame, root, labels):
        self.frame, self.root, self.labels = frame.reset_index(drop=True), Path(root), labels
        if not labels:
            require("Target_CAG_Binary" not in frame and "y_true" not in frame, "PREDICT_DATASET_HAS_LABELS")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = Path(row.hea_path)
        if not path.is_absolute():
            path = self.root / "data/raw" / path
        record = wfdb.rdrecord(str(path.with_suffix("")))
        require(record.fs == 500 and record.n_sig == 12 and record.sig_name == EXPECTED_LEADS, "ECG_SHAPE_OR_LEAD_ORDER")
        array, _ = preprocess_waveform(np.asarray(record.p_signal, dtype=np.float32).T, WaveformConfig())
        result = {"waveform": torch.from_numpy(array), "admission_id": str(row.admission_id)}
        if self.labels:
            result["label"] = torch.tensor(float(row.Target_CAG_Binary))
        return result

class Trainer:
    def __init__(self, root, config, workers=0, progress=None):
        self.root, self.config, self.workers = root, config, workers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.progress = progress

    def initialize(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = ResNet1DSmall(in_channels=12, dropout=self.config["dropout"]).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])
        return model, optimizer

    def loader(self, frame, labels, shuffle=False):
        return DataLoader(Waves(frame, self.root, labels), batch_size=self.config["batch_size"], shuffle=shuffle, num_workers=self.workers)

    def epoch(self, model, optimizer, loader):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(batch["waveform"].to(self.device)), batch["label"].to(self.device))
            loss.backward()
            optimizer.step()
            if self.progress is not None:
                self.progress.report()

    def predict(self, model, frame):
        model.eval()
        result = []
        with torch.no_grad():
            for batch in self.loader(frame, labels=False):
                scores = torch.sigmoid(model(batch["waveform"].to(self.device))).cpu().numpy()
                result.extend({"admission_id": patient, "ecg_oof_score": float(score)} for patient, score in zip(batch["admission_id"], scores))
                if self.progress is not None:
                    self.progress.report()
        return result

    def select_epoch(self, train, selection):
        model, optimizer = self.initialize(self.config["seed"])
        loader = self.loader(train, labels=True, shuffle=True)
        require(selection.Target_CAG_Binary.nunique() == 2, "INNER_SELECTION_ONE_CLASS")
        best, best_epoch, best_state, stale = -1.0, 0, None, 0
        for epoch in range(1, self.config["epochs"] + 1):
            self.epoch(model, optimizer, loader)
            predictions = self.predict(model, selection.drop(columns=["Target_CAG_Binary"]))
            score_by_id = {row["admission_id"]: row["ecg_oof_score"] for row in predictions}
            auc = roc_auc_score(selection.Target_CAG_Binary, [score_by_id[str(i)] for i in selection.admission_id])
            if auc > best:
                best, best_epoch, stale = auc, epoch, 0
                best_state = snapshot(model)
            else:
                stale += 1
            print(f"inner epoch={epoch} auroc={auc:.6f} selected_epoch={best_epoch}", flush=True)
            if stale >= self.config["patience"]:
                break
        require(best_state is not None, "NO_VALID_EPOCH")
        # Selection weights never initialize the outer refit.
        return best_epoch

    def refit(self, train, epoch):
        model, optimizer = self.initialize(self.config["refit_seed"])
        loader = self.loader(train, labels=True, shuffle=True)
        for step in range(epoch):
            self.epoch(model, optimizer, loader)
            print(f"outer refit epoch={step + 1}/{epoch}", flush=True)
        return model

    def load_refit(self, checkpoint):
        saved = torch.load(checkpoint, map_location=self.device, weights_only=True)
        model = ResNet1DSmall(in_channels=12, dropout=self.config["dropout"]).to(self.device)
        model.load_state_dict(saved["model_state_dict"])
        return model

def run_fold(output, binding, fold, train, holdout, trainer, progress, resume):
    directory = Path(output) / f"fold{fold}"
    preflight_fold(directory, binding, trainer.config["epochs"])
    with Transaction(directory, binding, resume) as tx:
        model_holder, epoch_holder = {}, {}

        def select_epoch(fit, selection):
            progress.start(fold, "epoch_selection")
            path = tx.directory / "epoch_selection.json"
            reused = path.exists()
            def operation():
                epoch = trainer.select_epoch(fit, selection)
                require(type(epoch) is int and 1 <= epoch <= trainer.config["epochs"], "INVALID_SELECTED_EPOCH")
                return {"selected_epoch": epoch}
            receipt = task(path, task_inputs(binding, "epoch_selection"), operation)
            epoch_holder["epoch"] = receipt["output"]["selected_epoch"]
            progress.finish(fold, "epoch_selection", receipt["duration_seconds"], reused)
            return epoch_holder["epoch"]

        def refit(data, epoch):
            progress.start(fold, "refit")
            path = tx.directory / "refit_task.json"
            checkpoint = tx.directory / "refit.pt"
            reused = path.exists()
            def operation():
                model_holder["model"] = trainer.refit(data, epoch)
                save_checkpoint(checkpoint, {"model_state_dict": snapshot(model_holder["model"]), "fixed_epoch": epoch, "outer_fold": fold})
                return {"selected_epoch": epoch, "checkpoint_sha256": file_hash(checkpoint)}
            receipt = task(path, task_inputs(binding, "refit", selected_epoch=epoch), operation)
            require(file_hash(checkpoint) == receipt["output"]["checkpoint_sha256"], "RESUME_REFUSED_CHECKPOINT_HASH:" + str(checkpoint))
            progress.finish(fold, "refit", receipt["duration_seconds"], reused)
            # A complete holdout receipt needs no model load or extra forward pass.
            if "model" not in model_holder and not (tx.directory / "predictions.json").exists():
                model_holder["model"] = trainer.load_refit(checkpoint)
            return model_holder.get("model")

        def predict(model, features):
            progress.start(fold, "holdout_prediction")
            checkpoint_hash = file_hash(tx.directory / "refit.pt")
            def operation():
                rows = trainer.predict(model, features)
                target = holdout.set_index("admission_id")
                for row in rows:
                    row.update({"outer_fold": fold, "oof_fold_id": fold, "split": "train", "tau_ecg": TAU_ECG,
                                "y_true": int(target.loc[row["admission_id"], "Target_CAG_Binary"]),
                                "waveform_id": str(target.loc[row["admission_id"], "waveform_id"]),
                                "ecg_oof_state": int(row["ecg_oof_score"] >= TAU_ECG)})
                validate_predictions(rows, binding, str(tx.directory / "predictions.json"))
                return {"predictions": rows, "selected_epoch": epoch_holder["epoch"], "checkpoint_sha256": checkpoint_hash}
            receipt = task(tx.directory / "predictions.json", task_inputs(binding, "holdout_prediction", selected_epoch=epoch_holder["epoch"], checkpoint_sha256=checkpoint_hash), operation)
            progress.report(force=True, detail="HOLDOUT_COMPLETE (not an extra training task)")
            return receipt["output"]

        # Keep the frozen selection -> fresh refit -> one-shot holdout orchestration.
        saved, _ = strict_fold(train, holdout[["admission_id", "hea_path"]], select_epoch, refit, predict)
        return saved["predictions"]

def run(root, method, output, workers=0, resume=False):
    validate_method(method)
    root, output = Path(root), Path(output)
    source = root / method["oof_manifest"]
    index_path = root / method["ecg_index"]
    # This is a training-only historical manifest, not the combined label table.
    frame = pd.read_csv(source, dtype={"admission_id": str, "waveform_id": str})
    require(len(frame) == method["expected_train_count"] and frame.admission_id.is_unique and set(frame.split) == {"train"}, "OOF_TRAIN_COHORT")
    require(set(frame.oof_fold_id) == {0, 1, 2}, "OOF_ASSIGNMENT")
    index = pd.read_csv(index_path, dtype={"admission_id": str, "waveform_id": str}, usecols=["admission_id", "waveform_id", "hea_path", "dat_path"])
    frame = frame.merge(index, on=["admission_id", "waveform_id"], validate="one_to_one", how="left")
    require(frame[["hea_path", "dat_path"]].notna().all().all(), "MISSING_WAVEFORM")
    progress = Progress()
    trainer = Trainer(root, method["ecg"], workers, progress)
    binding = build_binding(root, method, frame, workers, trainer.device)
    partitions, folds = [], []
    for fold in (0, 1, 2):
        train, holdout = frame[frame.oof_fold_id != fold].copy(), frame[frame.oof_fold_id == fold].copy()
        fit, selection = inner_split(train)
        partitions.append((train, holdout))
        folds.append(fold_binding(binding, fold, train, holdout, fit, selection))
    preflight_run(output, binding, folds, method["ecg"]["epochs"], resume)
    all_rows = []
    with Transaction(output, binding, resume):
        for fold, (train, holdout) in enumerate(partitions):
            all_rows.extend(run_fold(output, folds[fold], fold, train, holdout, trainer, progress, resume))
        require(len(all_rows) == method["expected_train_count"] and len({r["admission_id"] for r in all_rows}) == method["expected_train_count"], "OOF_COMPLETENESS")
        with Transaction(output / "complete", binding, resume):
            result = {"predictions": all_rows, "strict_epoch_selection": True, "holdout_inference_passes_per_fold": 1, "tau_ecg": TAU_ECG}
            receipt = task(output / "complete/strict_ecg_oof.json", binding, lambda: result)
            match(receipt["output"], result, "complete.output")
    return all_rows
