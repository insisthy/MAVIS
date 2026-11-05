"""Method-only extraction; study-specific commands and fixtures removed."""
from __future__ import annotations

import json

import math

import time

from datetime import datetime, timedelta, timezone

from pathlib import Path

import torch

from .contracts import TAU_ECG, digest, require

from .runtime import atomic_json, file_hash, software_versions

def mismatch_paths(saved, expected, prefix="binding"):
    if isinstance(saved, dict) and isinstance(expected, dict):
        result = []
        for key in sorted(set(saved) | set(expected)):
            field = prefix + "." + key
            if key not in saved or key not in expected:
                result.append(field)
            else:
                result.extend(mismatch_paths(saved[key], expected[key], field))
        return result
    return [] if type(saved) is type(expected) and saved == expected else [prefix]

def match(saved, expected, prefix="binding"):
    differences = mismatch_paths(saved, expected, prefix)
    require(not differences, "RESUME_REFUSED_FIELD_MISMATCH:" + ",".join(differences))

def membership(frame):
    columns = ["admission_id", "waveform_id"]
    require(frame.admission_id.is_unique, "MEMBERSHIP_DUPLICATE_PATIENT")
    return frame[columns].astype(str).sort_values(columns).to_dict("records")

def build_binding(root, method, frame, workers, device):
    root = Path(root)
    # Package-wide inclusion also covers the entry-point Full-freeze guard imports.
    paths = set((root / "src/training/strict_oof").glob("*.py"))
    paths.update(root / p for p in (
        "scripts/generate_strict_ecg_oof.py",
        "src/__init__.py", "src/models/__init__.py", "src/data_processing/__init__.py", "src/utils/__init__.py",
        "src/models/resnet1d_small.py", "src/data_processing/waveform_dataset.py", "src/utils/io.py"))
    code = {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(paths)}
    raw_inputs = {}
    for column in ("hea_path", "dat_path"):
        for value in sorted(set(frame[column])):
            path = Path(value)
            if not path.is_absolute():
                path = root / "data/raw" / path
            raw_inputs[str(path.resolve())] = file_hash(path)
    return {
        "contract": "strict_oof_engineering_v2", "code_sha256": code,
        "config_sha256": file_hash(root / "configs/strict_oof.json"), "method": method,
        "fold_assignment_sha256": file_hash(root / method["oof_manifest"]),
        "index_sha256": file_hash(root / method["ecg_index"]), "training_waveform_sha256": raw_inputs,
        "tau_ecg": TAU_ECG, "training_hyperparameters": method["ecg"],
        "seed_config": {"selection": method["ecg"]["seed"], "refit": method["ecg"]["refit_seed"]},
        "software_versions": software_versions(),
        "execution": {"num_workers": workers, "device": str(device), "cuda_version": torch.version.cuda,
                      "cudnn_version": torch.backends.cudnn.version(), "cudnn_benchmark": torch.backends.cudnn.benchmark,
                      "cudnn_deterministic": torch.backends.cudnn.deterministic,
                      "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                      "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                      "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                      "device_name": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else "cpu"},
        "training_membership": membership(frame),
    }

def fold_binding(binding, fold, train, holdout, inner_train, inner_selection):
    return {"run_binding": binding, "outer_fold": int(fold), "tau_ecg": TAU_ECG,
            "fitting_membership": membership(train), "holdout_membership": membership(holdout),
            "inner_train_membership": membership(inner_train), "inner_selection_membership": membership(inner_selection)}

def task_inputs(binding, phase, **extra):
    return {"fold_binding_sha256": digest(binding), "phase": phase, **extra}

def load_complete(path, inputs):
    path = Path(path)
    receipt = json.loads(path.read_text("utf-8"))
    require(receipt.get("status") == "COMPLETE", "RESUME_REFUSED_TASK_NOT_COMPLETE:" + str(path))
    match(receipt.get("input_binding"), inputs, str(path) + ".input_binding")
    require(receipt.get("input_sha256") == digest(inputs), "RESUME_REFUSED_INPUT_HASH:" + str(path))
    require(receipt.get("output_sha256") == digest(receipt["output"]), "RESUME_REFUSED_OUTPUT_HASH:" + str(path))
    seconds = receipt.get("duration_seconds")
    require(type(seconds) in (float, int) and math.isfinite(seconds) and seconds >= 0, "INVALID_TASK_DURATION:" + str(path))
    return receipt

def task(path, inputs, operation, clock=time.monotonic):
    path = Path(path)
    if path.exists():
        return load_complete(path, inputs)
    started = clock()
    atomic_json(path, {"status": "RUNNING", "input_binding": inputs, "input_sha256": digest(inputs)})
    try:
        output = operation()
    except BaseException as error:
        atomic_json(path, {"status": "FAILED", "input_binding": inputs, "input_sha256": digest(inputs), "error_type": type(error).__name__, "error": str(error)})
        raise
    receipt = {"status": "COMPLETE", "input_binding": inputs, "input_sha256": digest(inputs),
               "output": output, "output_sha256": digest(output), "duration_seconds": clock() - started}
    atomic_json(path, receipt)
    return receipt

def validate_predictions(rows, binding, context):
    expected = binding["holdout_membership"]
    actual = sorted(({"admission_id": str(r["admission_id"]), "waveform_id": str(r["waveform_id"])} for r in rows), key=lambda r: (r["admission_id"], r["waveform_id"]))
    match(actual, expected, context + ".holdout_membership")
    for row in rows:
        match(row["outer_fold"], binding["outer_fold"], context + ".outer_fold")
        match(row["oof_fold_id"], binding["outer_fold"], context + ".oof_fold_id")
        match(row["tau_ecg"], TAU_ECG, context + ".tau_ecg")
        require(row["split"] == "train", "RESUME_REFUSED_NONTRAIN_PREDICTION:" + context)
        score = row["ecg_oof_score"]
        require(type(score) in (float, int) and math.isfinite(score) and 0 <= score <= 1, "INVALID_OOF_SCORE:" + context)
        match(row["ecg_oof_state"], int(score >= TAU_ECG), context + ".ecg_oof_state")
        require(type(row["y_true"]) is int and row["y_true"] in (0, 1), "INVALID_OOF_LABEL_METADATA:" + context)

def preflight_fold(directory, binding, max_epochs):
    """Check every reusable artifact before any new gradient work is possible."""
    directory = Path(directory)
    if not directory.exists():
        return
    require(not (directory / "active.lock").exists(), "ACTIVE_OR_INTERRUPTED_LOCK_REQUIRES_REVIEW:" + str(directory))
    require(not (directory / "technical_failure.json").exists(), "TERMINAL_TECHNICAL_FAILURE_REQUIRES_REVIEW:" + str(directory))
    require((directory / "binding.json").exists(), "RESUME_REFUSED_MISSING_FOLD_BINDING:" + str(directory))
    match(json.loads((directory / "binding.json").read_text("utf-8")), binding, str(directory) + ".binding")
    selection, refit, predictions = directory / "epoch_selection.json", directory / "refit_task.json", directory / "predictions.json"
    if not selection.exists():
        require(not any(p.name != "binding.json" for p in directory.iterdir()), "RESUME_REFUSED_PARTIAL_FOLD:" + str(directory))
        return
    selected = load_complete(selection, task_inputs(binding, "epoch_selection"))["output"]["selected_epoch"]
    require(type(selected) is int and 1 <= selected <= max_epochs, "RESUME_REFUSED_SELECTED_EPOCH:" + str(selection))
    if not refit.exists():
        require(not (directory / "refit.pt").exists() and not predictions.exists(), "RESUME_REFUSED_ORPHAN_REFIT:" + str(directory))
        return
    fitted = load_complete(refit, task_inputs(binding, "refit", selected_epoch=selected))["output"]
    match(fitted["selected_epoch"], selected, str(refit) + ".selected_epoch")
    checkpoint = directory / "refit.pt"
    require(checkpoint.is_file(), "RESUME_REFUSED_CHECKPOINT_MISSING:" + str(checkpoint))
    require(file_hash(checkpoint) == fitted["checkpoint_sha256"], "RESUME_REFUSED_CHECKPOINT_HASH:" + str(checkpoint))
    if predictions.exists():
        saved = load_complete(predictions, task_inputs(binding, "holdout_prediction", selected_epoch=selected, checkpoint_sha256=fitted["checkpoint_sha256"]))["output"]
        match(saved["checkpoint_sha256"], fitted["checkpoint_sha256"], str(predictions) + ".checkpoint_sha256")
        match(saved["selected_epoch"], selected, str(predictions) + ".selected_epoch")
        validate_predictions(saved["predictions"], binding, str(predictions))

def preflight_run(output, binding, folds, max_epochs, resume):
    output = Path(output)
    if not output.exists():
        return
    require(resume, "OOF_OUTPUT_EXISTS_USE_RESUME:" + str(output))
    path = output / "binding.json"
    require(path.exists(), "RESUME_REFUSED_LEGACY_OR_UNBOUND_RUN:" + str(output))
    match(json.loads(path.read_text("utf-8")), binding, "run.binding")
    require(not (output / "active.lock").exists(), "ACTIVE_OR_INTERRUPTED_RUN_REQUIRES_REVIEW")
    require(not (output / "technical_failure.json").exists(), "TERMINAL_RUN_FAILURE_REQUIRES_REVIEW")
    for number, expected in enumerate(folds):
        preflight_fold(output / f"fold{number}", expected, max_epochs)
    complete = output / "complete"
    if complete.exists():
        require(not (complete / "active.lock").exists() and not (complete / "technical_failure.json").exists(), "INCOMPLETE_AGGREGATION_REQUIRES_REVIEW")
        require((complete / "binding.json").exists(), "AGGREGATION_BINDING_MISSING")
        match(json.loads((complete / "binding.json").read_text("utf-8")), binding, "complete.binding")
        path = complete / "strict_ecg_oof.json"
        if path.exists():
            saved = load_complete(path, binding)["output"]
            require(all((output / f"fold{k}/predictions.json").exists() for k in range(3)), "AGGREGATION_WITHOUT_COMPLETE_FOLDS")
            rows = []
            for k in range(3):
                rows.extend(json.loads((output / f"fold{k}/predictions.json").read_text("utf-8"))["output"]["predictions"])
            match(saved["predictions"], rows, "complete.predictions")

class Progress:
    total = 6

    def __init__(self, clock=time.monotonic, emit=print):
        self.clock, self.emit = clock, emit
        self.started = clock()
        self.task_started = self.started
        self.last_report = -float("inf")
        self.completed = {}
        self.fold, self.phase, self.current = 0, "epoch_selection", 1

    def start(self, fold, phase):
        self.fold, self.phase = fold, phase
        self.current = fold * 2 + (1 if phase == "epoch_selection" else 2)
        self.task_started = self.clock()
        self.report(force=True)

    def finish(self, fold, phase, seconds, reused=False):
        require(phase in ("epoch_selection", "refit"), "HOLDOUT_IS_NOT_A_TRAINING_TASK")
        self.completed[(fold, phase)] = seconds
        self.report(force=True, detail="COMPLETE_REUSED" if reused else "COMPLETE")

    @staticmethod
    def duration(seconds):
        seconds = max(0, int(seconds))
        return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"

    def report(self, detail="", force=False):
        now = self.clock()
        if not force and now - self.last_report < 5:
            return
        self.last_report = now
        task_elapsed = now - self.task_started
        if self.completed:
            remaining = sum(self.completed.values()) / len(self.completed) * (self.total - len(self.completed))
            if (self.fold, self.phase) not in self.completed and self.phase != "holdout_prediction":
                remaining = max(0, remaining - task_elapsed)
            eta = (datetime.now(timezone(timedelta(hours=8))) + timedelta(seconds=remaining)).isoformat(timespec="seconds")
            estimate = self.duration(remaining)
        else:
            estimate = eta = "estimating_until_first_training_task_completes"
        self.emit(f"outer fold = {self.fold + 1}/3 | phase = {self.phase} | task = {self.current}/6 | completed tasks = {len(self.completed)}/6 | elapsed = {self.duration(now - self.started)} | current task elapsed = {self.duration(task_elapsed)} | estimated remaining = {estimate} | ETA = {eta} (training-task estimate) | {detail}", flush=True)
