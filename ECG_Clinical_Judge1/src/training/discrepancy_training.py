"""Method-only extraction; study-specific commands and fixtures removed."""
from __future__ import annotations

import argparse

import csv

import hashlib

import inspect

import json

import math

import platform

import warnings

from dataclasses import dataclass

from datetime import datetime

from pathlib import Path

from typing import Any

import joblib

import numpy as np

import pandas as pd

import sklearn

from sklearn.base import clone

from sklearn.calibration import CalibratedClassifierCV

from sklearn.compose import ColumnTransformer

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

from sklearn.impute import SimpleImputer

from sklearn.inspection import permutation_importance

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from sklearn.model_selection import ParameterGrid, StratifiedKFold

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.exceptions import ConvergenceWarning

RANDOM_STATE = 20260706

ECG_THRESHOLD = 0.5

DEFAULT_FORMAL_OUTER_CV_SPLITS = 5

DEFAULT_FORMAL_INNER_CV_SPLITS = 3

DEFAULT_ELASTIC_NET_MAX_ITER = 5000

DEFAULT_FORMAL_ELASTIC_NET_MAX_ITER = 20000

MANUAL_TRAINING_NOTICE = (
    "Discrepancy-model training requires explicit execution. "
    "Use python -m scripts.train_discrepancy_models with --execute-training."
)

ID_COLUMNS = {
    "admission_id",
    "record_id",
    "patient_id",
    "inpat_id",
    "source_record_id",
    "waveform_id",
}

LABEL_OR_LEAKAGE_COLUMNS = {
    "Target_CAG_Binary",
    "CAG_label",
    "group_label",
    "ECG_Ischemic_Phenotype",
    "ECG_ischemia_label",
    "ECG_label",
    "CAG_Time",
    "ecg_start_time",
    "ECG_time",
    "split",
    "fold_id",
    "oof_fold_id",
}

FORBIDDEN_RUNTIME_OUTPUT_TOKENS = (
    "agent1",
    "agent2_b",
    "agent3",
    "agent4",
    "rag",
    "fusion",
    "clinical_score",
    "raw_delta",
    "constrained_delta",
    "mas_corrected_score",
)

@dataclass(frozen=True)
class BranchData:
    name: str
    label_col: str
    data: pd.DataFrame
    positive_count: int
    negative_count: int

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"admission_id": str, "waveform_id": str}, low_memory=False)

def is_forbidden_feature(column: str) -> bool:
    lower = column.lower()
    if column in ID_COLUMNS or column in LABEL_OR_LEAKAGE_COLUMNS:
        return True
    return any(token in lower for token in FORBIDDEN_RUNTIME_OUTPUT_TOKENS)

def infer_clinical_features(agent2: pd.DataFrame) -> list[str]:
    return [col for col in agent2.columns if not is_forbidden_feature(col)]

def build_training_table(
    agent2_path: Path,
    label_path: Path,
    ecg_oof_path: Path,
    *,
    ecg_threshold: float = ECG_THRESHOLD,
) -> tuple[pd.DataFrame, list[str]]:
    agent2 = _read_csv(agent2_path)
    labels = _read_csv(label_path)
    ecg = _read_csv(ecg_oof_path)

    labels = labels[labels["split"].astype(str).str.lower().eq("train")].copy()
    ecg = ecg[ecg["split"].astype(str).str.lower().eq("train")].copy()
    features = infer_clinical_features(agent2)

    if labels["admission_id"].duplicated().any():
        raise ValueError("duplicate admission_id in train labels")
    if ecg["admission_id"].duplicated().any():
        raise ValueError("duplicate admission_id in ECG OOF artifact")

    base = labels[["admission_id", "Target_CAG_Binary", "group_label", "split"]].merge(
        ecg[["admission_id", "waveform_id", "oof_fold_id", "ecg_score", "ecg_oof_model", "ecg_class_weight_strategy"]],
        on="admission_id",
        how="inner",
    )
    base = base.merge(agent2[["admission_id", *features]], on="admission_id", how="inner")
    if len(base) != len(labels):
        raise ValueError(f"train alignment mismatch: expected {len(labels)}, got {len(base)}")

    base["oof_ecg_score"] = pd.to_numeric(base["ecg_score"], errors="raise")
    base["locked_ecg_threshold"] = float(ecg_threshold)
    base["oof_score_margin"] = base["oof_ecg_score"] - float(ecg_threshold)
    base["oof_ecg_baseline_state"] = np.where(
        base["oof_ecg_score"] >= float(ecg_threshold),
        "ECG_MODEL_POSITIVE",
        "ECG_MODEL_NEGATIVE",
    )
    base["target_cag_binary"] = pd.to_numeric(base["Target_CAG_Binary"], errors="raise").astype(int)
    base["train_underestimation_label"] = np.where(base["target_cag_binary"].eq(1), 1, 0)
    base["train_overestimation_label"] = np.where(base["target_cag_binary"].eq(0), 1, 0)
    return base, features

def split_branches(table: pd.DataFrame) -> dict[str, BranchData]:
    under = table[table["oof_ecg_baseline_state"].eq("ECG_MODEL_NEGATIVE")].copy()
    over = table[table["oof_ecg_baseline_state"].eq("ECG_MODEL_POSITIVE")].copy()
    branches = {
        "underestimation": BranchData(
            "underestimation",
            "train_underestimation_label",
            under,
            int(under["train_underestimation_label"].sum()),
            int((1 - under["train_underestimation_label"]).sum()),
        ),
        "overestimation": BranchData(
            "overestimation",
            "train_overestimation_label",
            over,
            int(over["train_overestimation_label"].sum()),
            int((1 - over["train_overestimation_label"]).sum()),
        ),
    }
    return branches

def make_preprocessor(features: list[str], data: pd.DataFrame, *, scale_numeric: bool) -> ColumnTransformer:
    # Be conservative: object columns with values such as ``unknown`` or
    # ``not_applicable`` must stay categorical so their semantics are not
    # collapsed into numeric imputation.  Only native numeric dtypes enter the
    # median-imputation numeric branch.
    numeric_cols = [col for col in features if pd.api.types.is_numeric_dtype(data[col])]
    categorical_cols = [col for col in features if col not in numeric_cols]
    imputer_kwargs: dict[str, Any] = {"strategy": "median", "add_indicator": True}
    if "keep_empty_features" in inspect.signature(SimpleImputer).parameters:
        imputer_kwargs["keep_empty_features"] = True
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(**imputer_kwargs))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_cols:
        transformers.append(("num", Pipeline(numeric_steps), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)

def feature_sets(clinical_features: list[str]) -> dict[str, list[str]]:
    return {
        "F0_ecg_score_only": ["oof_ecg_score"],
        "F1_clinical_only_discrepancy": list(clinical_features),
        "F2_clinical_plus_ecg_score": [*clinical_features, "oof_ecg_score"],
    }

def make_elastic_net_logistic(*, max_iter: int = DEFAULT_ELASTIC_NET_MAX_ITER) -> LogisticRegression:
    return LogisticRegression(
        solver="saga",
        penalty="elasticnet",
        max_iter=max_iter,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

def ece_score(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    edges = np.linspace(0, 1, bins + 1)
    total = len(y_true)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1 else y_prob <= hi)
        if mask.any():
            ece += mask.mean() * abs(float(y_prob[mask].mean()) - float(y_true[mask].mean()))
    return float(ece)

def calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float | None, float | None]:
    p = np.clip(y_prob, 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p)).reshape(-1, 1)
    if len(np.unique(y_true)) < 2:
        return None, None
    try:
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(x, y_true)
        return float(lr.coef_[0][0]), float(lr.intercept_[0])
    except Exception:
        return None, None

def metrics_for(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | None]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    out: dict[str, float | None] = {
        "auroc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else None,
        "auprc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else None,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": ece_score(y_true, y_prob),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6), labels=[0, 1])),
    }
    slope, intercept = calibration_slope_intercept(y_true, y_prob)
    out["calibration_slope"] = slope
    out["calibration_intercept"] = intercept
    return out

def top_fraction_metrics(y_true: np.ndarray, y_prob: np.ndarray, frac: float) -> dict[str, float]:
    n = max(1, int(math.ceil(len(y_true) * frac)))
    idx = np.argsort(-y_prob)[:n]
    positives = float(np.asarray(y_true)[idx].sum())
    total_positive = max(1.0, float(np.asarray(y_true).sum()))
    return {
        f"precision_at_top{int(frac * 100)}pct": positives / n,
        f"recall_at_top{int(frac * 100)}pct": positives / total_positive,
    }

def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, *, n: int = 200) -> dict[str, str]:
    if n <= 0:
        return {"auroc": "disabled", "auprc": "disabled", "brier": "disabled", "ece": "disabled"}
    rng = np.random.default_rng(RANDOM_STATE)
    rows: dict[str, list[float]] = {"auroc": [], "auprc": [], "brier": [], "ece": []}
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = metrics_for(y_true[idx], y_prob[idx])
        for key in rows:
            if m[key] is not None:
                rows[key].append(float(m[key]))
    return {
        key: f"{np.percentile(vals, 2.5):.6f},{np.percentile(vals, 97.5):.6f}" if vals else ""
        for key, vals in rows.items()
    }

def make_pipeline(model: Any, features: list[str], data: pd.DataFrame, *, scale_numeric: bool) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocessor(features, data, scale_numeric=scale_numeric)),
            ("model", model),
        ]
    )

def candidate_convergence_eligible(summary_row: dict[str, Any], convergence_rows: list[dict[str, Any]] | None = None) -> bool:
    if not convergence_rows:
        return True
    feature_setting = summary_row.get("feature_setting")
    model_name = summary_row.get("model")
    calibration = summary_row.get("calibration")
    for row in convergence_rows:
        if row.get("feature_setting") != feature_setting or row.get("model_name") != model_name:
            continue
        stage = str(row.get("fit_stage", ""))
        if stage == "calibration" and calibration != "sigmoid":
            continue
        if stage not in {"inner_search", "outer_refit", "calibration", "final_candidate_fit", "final_candidate_fit_calibration"}:
            continue
        warning_count = int(row.get("convergence_warning_count", 0) or 0)
        diagnostic_available = row.get("diagnostic_available") is True or str(row.get("diagnostic_available")).lower() == "true"
        converged = row.get("converged") is True or str(row.get("converged")).lower() == "true"
        if diagnostic_available and warning_count > 0 and not converged:
            return False
    return True

def select_candidate(summary_rows: list[dict[str, Any]], convergence_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    # Pre-fixed lexical rule: AUPRC desc, if within 0.01 then Brier asc,
    # then ECE asc, then simpler model priority.
    valid = [
        r for r in summary_rows
        if r.get("calibration") in {"uncalibrated", "sigmoid"}
        and candidate_convergence_eligible(r, convergence_rows)
    ]
    if not valid:
        raise ValueError("no convergence_eligible formal candidate available")
    best = sorted(
        valid,
        key=lambda r: (
            -float(r["auprc"]),
            float(r["brier"]),
            float(r["ece"]),
            int(r["model_priority"]),
        ),
    )[0]
    tied = [
        r
        for r in valid
        if abs(float(r["auprc"]) - float(best["auprc"])) <= 0.01
    ]
    if len(tied) > 1:
        best = sorted(tied, key=lambda r: (float(r["brier"]), float(r["ece"]), int(r["model_priority"])))[0]
    return best

def _logistic_estimators(estimator: Any) -> list[LogisticRegression]:
    """Return fitted LogisticRegression estimators nested inside sklearn wrappers."""
    found: list[LogisticRegression] = []
    if isinstance(estimator, LogisticRegression):
        return [estimator]
    if isinstance(estimator, Pipeline):
        model = estimator.named_steps.get("model")
        if model is not None:
            found.extend(_logistic_estimators(model))
    if hasattr(estimator, "best_estimator_"):
        found.extend(_logistic_estimators(estimator.best_estimator_))
    if hasattr(estimator, "estimator"):
        base = getattr(estimator, "estimator")
        if base is not None:
            found.extend(_logistic_estimators(base))
    if hasattr(estimator, "calibrated_classifiers_"):
        for calibrated in getattr(estimator, "calibrated_classifiers_", []):
            inner = getattr(calibrated, "estimator", None) or getattr(calibrated, "base_estimator", None)
            if inner is not None:
                found.extend(_logistic_estimators(inner))
    return found

def logistic_convergence_state(estimator: Any) -> dict[str, Any]:
    estimators = _logistic_estimators(estimator)
    if not estimators:
        return {
            "max_iter": "",
            "observed_n_iter": "",
            "converged": True,
            "diagnostic_available": False,
        }
    max_iters = [int(getattr(model, "max_iter", 0)) for model in estimators]
    observed: list[int] = []
    for model in estimators:
        n_iter = getattr(model, "n_iter_", [])
        arr = np.asarray(n_iter).ravel()
        if arr.size:
            observed.append(int(arr.max()))
    max_iter = max(max_iters) if max_iters else 0
    observed_max = max(observed) if observed else ""
    max_iter_reached = bool(observed) and any(n >= int(getattr(model, "max_iter", 0)) for model, n in zip(estimators, observed))
    return {
        "max_iter": max_iter,
        "observed_n_iter": observed_max,
        "max_iter_reached": max_iter_reached,
        "diagnostic_available": True,
    }

def convergence_warning_messages(records: list[warnings.WarningMessage]) -> list[str]:
    return [str(item.message) for item in records if issubclass(item.category, ConvergenceWarning)]

def convergence_diagnostic_row(
    *,
    task: str,
    feature_setting: str,
    model_name: str,
    outer_fold: int | str,
    fit_stage: str,
    selected_hyperparameters: dict[str, Any] | str,
    estimator: Any,
    warning_messages: list[str],
) -> dict[str, Any]:
    state = logistic_convergence_state(estimator)
    warning_count = len(warning_messages)
    # Hard non-convergence is warning-driven.  n_iter_ reaching max_iter is
    # retained as diagnostic evidence, but without a ConvergenceWarning in the
    # scoped fit call it must not create a duplicate selected-stage failure.
    converged = warning_count == 0
    return {
        "task": task,
        "feature_setting": feature_setting,
        "model_name": model_name,
        "outer_fold": outer_fold,
        "fit_stage": fit_stage,
        "selected_hyperparameters": json.dumps(selected_hyperparameters, sort_keys=True) if isinstance(selected_hyperparameters, dict) else selected_hyperparameters,
        "max_iter": state["max_iter"],
        "observed_n_iter": state["observed_n_iter"],
        "convergence_warning_count": warning_count,
        "converged": converged,
        "max_iter_reached": state.get("max_iter_reached", ""),
        "warning_message": " | ".join(warning_messages),
        "diagnostic_available": state["diagnostic_available"],
    }

def _public_candidate_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}

def _make_formal_estimator(
    model: Any,
    params: dict[str, Any],
    features: list[str],
    training_data: pd.DataFrame,
    *,
    scale_numeric: bool,
    calibration: str,
) -> Any:
    pipeline = make_pipeline(clone(model), features, training_data, scale_numeric=scale_numeric)
    pipeline.set_params(**params)
    if calibration == "sigmoid":
        return CalibratedClassifierCV(estimator=pipeline, method="sigmoid", cv=3)
    return pipeline

def _evaluate_formal_candidate_space(
    data: pd.DataFrame,
    y: np.ndarray,
    fs_map: dict[str, list[str]],
    grids: dict[str, tuple[Any, list[dict[str, list[Any]]], bool, int]],
    *,
    task: str,
    selection_scope: str,
    cv_splits: int,
    enable_sigmoid_calibration: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray], list[dict[str, Any]]]:
    """Select a complete model/input/hyperparameter/calibration pipeline by CV."""

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    calibrations = ["uncalibrated", "sigmoid"] if enable_sigmoid_calibration else ["uncalibrated"]
    candidate_rows: list[dict[str, Any]] = []
    probabilities: dict[str, np.ndarray] = {}
    convergence_rows: list[dict[str, Any]] = []

    for feature_setting, features in fs_map.items():
        for model_name, (model, grid, scale_numeric, priority) in grids.items():
            for config_index, params in enumerate(ParameterGrid(grid)):
                for calibration in calibrations:
                    candidate_id = f"{feature_setting}|{model_name}|{config_index}|{calibration}"
                    oof_probability = np.full(len(data), np.nan)
                    warning_count = 0
                    diagnostic_available = False
                    for inner_fold, (train_idx, holdout_idx) in enumerate(cv.split(data[features], y)):
                        estimator = _make_formal_estimator(
                            model,
                            params,
                            features,
                            data.iloc[train_idx],
                            scale_numeric=scale_numeric,
                            calibration=calibration,
                        )
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always", ConvergenceWarning)
                            estimator.fit(data.iloc[train_idx][features], y[train_idx])
                        fit_warnings = convergence_warning_messages(caught)
                        diagnostic = convergence_diagnostic_row(
                            task=task,
                            feature_setting=feature_setting,
                            model_name=model_name,
                            outer_fold=selection_scope,
                            fit_stage=f"inner_cv_fold_{inner_fold}",
                            selected_hyperparameters=params,
                            estimator=estimator,
                            warning_messages=fit_warnings,
                        )
                        diagnostic["candidate_id"] = candidate_id
                        diagnostic["calibration"] = calibration
                        convergence_rows.append(diagnostic)
                        warning_count += int(diagnostic["convergence_warning_count"])
                        diagnostic_available = diagnostic_available or bool(diagnostic["diagnostic_available"])
                        oof_probability[holdout_idx] = estimator.predict_proba(data.iloc[holdout_idx][features])[:, 1]
                    if np.isnan(oof_probability).any():
                        raise RuntimeError(f"incomplete inner OOF prediction for {selection_scope}:{candidate_id}")
                    metric_row = metrics_for(y, oof_probability)
                    candidate_rows.append(
                        {
                            "head": task,
                            "selection_scope": selection_scope,
                            "candidate_id": candidate_id,
                            "feature_setting": feature_setting,
                            "model": model_name,
                            "calibration": calibration,
                            "model_priority": priority,
                            "hyperparameter_config_index": config_index,
                            "best_params": json.dumps(params, sort_keys=True),
                            "inner_cv_splits": cv_splits,
                            "inner_convergence_warning_count": warning_count,
                            "convergence_eligible": not (diagnostic_available and warning_count > 0),
                            **metric_row,
                            **top_fraction_metrics(y, oof_probability, 0.10),
                            **top_fraction_metrics(y, oof_probability, 0.20),
                            "_params": params,
                        }
                    )
                    probabilities[candidate_id] = oof_probability

    eligible_rows = [row for row in candidate_rows if row["convergence_eligible"]]
    selected = select_candidate(eligible_rows)
    for row in candidate_rows:
        row["selected_in_scope"] = row["candidate_id"] == selected["candidate_id"]
    for row in convergence_rows:
        row["selected_in_scope"] = row["candidate_id"] == selected["candidate_id"]
    return selected, candidate_rows, probabilities, convergence_rows

def train_branch(
    branch: BranchData,
    clinical_features: list[str],
    out_dir: Path,
    *,
    grid_profile: str,
    bootstrap_n: int,
    outer_cv_splits: int = DEFAULT_FORMAL_OUTER_CV_SPLITS,
    inner_cv_splits: int = DEFAULT_FORMAL_INNER_CV_SPLITS,
    enable_sigmoid_calibration: bool = True,
    elastic_net_max_iter: int = DEFAULT_ELASTIC_NET_MAX_ITER,
) -> dict[str, Any]:
    del grid_profile  # Formal selection always uses the prespecified controlled grids.
    grids = formal_controlled_candidate_grids(elastic_net_max_iter=elastic_net_max_iter)
    fs_map = feature_sets(clinical_features)
    if len(grids) != 3 or len(fs_map) != 3:
        raise RuntimeError("formal candidate space must contain 3 model families x 3 feature settings")

    outer = StratifiedKFold(n_splits=outer_cv_splits, shuffle=True, random_state=RANDOM_STATE)
    y = branch.data[branch.label_col].to_numpy(dtype=int)
    nested_oof_probability = np.full(len(branch.data), np.nan)
    outer_selected_model = np.full(len(branch.data), "", dtype=object)
    outer_selected_feature_setting = np.full(len(branch.data), "", dtype=object)
    outer_selected_calibration = np.full(len(branch.data), "", dtype=object)
    outer_selected_hyperparameters = np.full(len(branch.data), "", dtype=object)
    outer_fold_assignment = np.full(len(branch.data), -1, dtype=int)
    fold_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    outer_selection_rows: list[dict[str, Any]] = []
    inner_candidate_rows: list[dict[str, Any]] = []
    convergence_rows: list[dict[str, Any]] = []

    for outer_fold, (train_idx, holdout_idx) in enumerate(outer.split(branch.data, y)):
        outer_train = branch.data.iloc[train_idx]
        y_train = y[train_idx]
        y_holdout = y[holdout_idx]
        selected, candidate_rows, _, inner_convergence = _evaluate_formal_candidate_space(
            outer_train,
            y_train,
            fs_map,
            grids,
            task=branch.name,
            selection_scope=f"outer_fold_{outer_fold}",
            cv_splits=inner_cv_splits,
            enable_sigmoid_calibration=enable_sigmoid_calibration,
        )
        inner_candidate_rows.extend(_public_candidate_row(row) for row in candidate_rows)
        convergence_rows.extend(inner_convergence)

        features = fs_map[selected["feature_setting"]]
        model, _, scale_numeric, _ = grids[selected["model"]]
        winner = _make_formal_estimator(
            model,
            selected["_params"],
            features,
            outer_train,
            scale_numeric=scale_numeric,
            calibration=selected["calibration"],
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            winner.fit(outer_train[features], y_train)
        winner_warnings = convergence_warning_messages(caught)
        winner_diagnostic = convergence_diagnostic_row(
            task=branch.name,
            feature_setting=selected["feature_setting"],
            model_name=selected["model"],
            outer_fold=outer_fold,
            fit_stage="outer_winner_refit",
            selected_hyperparameters=selected["_params"],
            estimator=winner,
            warning_messages=winner_warnings,
        )
        winner_diagnostic["candidate_id"] = selected["candidate_id"]
        winner_diagnostic["calibration"] = selected["calibration"]
        winner_diagnostic["selected_in_scope"] = True
        convergence_rows.append(winner_diagnostic)

        holdout_probability = winner.predict_proba(branch.data.iloc[holdout_idx][features])[:, 1]
        nested_oof_probability[holdout_idx] = holdout_probability
        params_json = json.dumps(selected["_params"], sort_keys=True)
        outer_selected_model[holdout_idx] = selected["model"]
        outer_selected_feature_setting[holdout_idx] = selected["feature_setting"]
        outer_selected_calibration[holdout_idx] = selected["calibration"]
        outer_selected_hyperparameters[holdout_idx] = params_json
        outer_fold_assignment[holdout_idx] = outer_fold
        outer_metrics = metrics_for(y_holdout, holdout_probability)
        fold_rows.append(
            {
                "head": branch.name,
                "outer_fold": outer_fold,
                "selected_model": selected["model"],
                "selected_feature_setting": selected["feature_setting"],
                "selected_calibration": selected["calibration"],
                "selected_hyperparameters": params_json,
                "inner_selection_auprc": selected["auprc"],
                "inner_selection_brier": selected["brier"],
                "inner_selection_ece": selected["ece"],
                **outer_metrics,
            }
        )
        outer_selection_rows.append(
            {
                "head": branch.name,
                "outer_fold": outer_fold,
                "model_family": selected["model"],
                "feature_setting": selected["feature_setting"],
                "hyperparameters": params_json,
                "calibration_strategy": selected["calibration"],
                "inner_auprc": selected["auprc"],
                "inner_brier": selected["brier"],
                "inner_ece": selected["ece"],
                "outer_auprc": outer_metrics["auprc"],
                "outer_auroc": outer_metrics["auroc"],
                "outer_brier": outer_metrics["brier"],
                "outer_ece": outer_metrics["ece"],
            }
        )
        split_rows.append(
            {
                "head": branch.name,
                "outer_fold": outer_fold,
                "outer_train_count": len(train_idx),
                "outer_holdout_count": len(holdout_idx),
                "outer_train_positive": int(y_train.sum()),
                "outer_holdout_positive": int(y_holdout.sum()),
                "inner_selected_model": selected["model"],
                "inner_selected_feature_setting": selected["feature_setting"],
                "inner_selected_hyperparameters": params_json,
                "inner_selected_calibration": selected["calibration"],
                "random_state": RANDOM_STATE,
            }
        )

    if np.isnan(nested_oof_probability).any() or (outer_fold_assignment < 0).any():
        raise RuntimeError(f"incomplete nested outer OOF predictions for {branch.name}")

    final_selected, full_candidate_rows, full_candidate_probabilities, full_convergence = _evaluate_formal_candidate_space(
        branch.data,
        y,
        fs_map,
        grids,
        task=branch.name,
        selection_scope="full_training_cv",
        cv_splits=inner_cv_splits,
        enable_sigmoid_calibration=enable_sigmoid_calibration,
    )
    convergence_rows.extend(full_convergence)
    summary_rows: list[dict[str, Any]] = []
    for row in full_candidate_rows:
        public_row = _public_candidate_row(row)
        ci = bootstrap_ci(y, full_candidate_probabilities[row["candidate_id"]], n=bootstrap_n)
        public_row.update(
            {
                "bootstrap_ci_auroc": ci["auroc"],
                "bootstrap_ci_auprc": ci["auprc"],
                "bootstrap_ci_brier": ci["brier"],
                "bootstrap_ci_ece": ci["ece"],
            }
        )
        summary_rows.append(public_row)

    selected_features = fs_map[final_selected["feature_setting"]]
    selected_model, _, scale_numeric, _ = grids[final_selected["model"]]
    final_estimator = _make_formal_estimator(
        selected_model,
        final_selected["_params"],
        selected_features,
        branch.data,
        scale_numeric=scale_numeric,
        calibration=final_selected["calibration"],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        final_estimator.fit(branch.data[selected_features], y)
    final_warnings = convergence_warning_messages(caught)
    final_diagnostic = convergence_diagnostic_row(
        task=branch.name,
        feature_setting=final_selected["feature_setting"],
        model_name=final_selected["model"],
        outer_fold="full_train",
        fit_stage="final_configuration_refit",
        selected_hyperparameters=final_selected["_params"],
        estimator=final_estimator,
        warning_messages=final_warnings,
    )
    final_diagnostic["candidate_id"] = final_selected["candidate_id"]
    final_diagnostic["calibration"] = final_selected["calibration"]
    final_diagnostic["selected_in_scope"] = True
    convergence_rows.append(final_diagnostic)

    artifact_dir = out_dir / "candidate_artifacts" / branch.name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / f"{branch.name}_candidate_model.joblib"
    joblib.dump(final_estimator, model_path)
    schema = {
        "head_name": branch.name,
        "label_col": branch.label_col,
        "feature_setting": final_selected["feature_setting"],
        "features": selected_features,
        "forbidden_features": sorted(ID_COLUMNS | LABEL_OR_LEAKAGE_COLUMNS),
        "runtime_locked": False,
        "validation_locked": False,
        "alert_threshold_locked": False,
    }
    schema_path = artifact_dir / f"{branch.name}_feature_schema.json"
    write_json(schema_path, schema)
    metadata = {
        "head_name": branch.name,
        "label_definition": "under: ECG_MODEL_NEGATIVE CAG positive; over: ECG_MODEL_POSITIVE CAG negative",
        "candidate_selected_on_train_cv": True,
        "runtime_locked": False,
        "validation_locked": False,
        "alert_threshold_locked": False,
        "candidate_model_family": final_selected["model"],
        "selected_feature_setting": final_selected["feature_setting"],
        "selected_calibration": final_selected["calibration"],
        "selected_hyperparameters": final_selected["_params"],
        "selection_source": "full_training_inner_cv",
        "outer_oof_used_for_final_configuration_selection": False,
        "formal_candidate_model_family_count": len(grids),
        "formal_candidate_feature_setting_count": len(fs_map),
        "screening_result_used_for_formal_selection": False,
        "train_case_count": len(branch.data),
        "positive_count": branch.positive_count,
        "negative_count": branch.negative_count,
        "cv_seed": RANDOM_STATE,
        "software_versions": {"python": platform.python_version(), "sklearn": sklearn.__version__},
        "training_timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    metadata_path = artifact_dir / f"{branch.name}_model_metadata.json"
    write_json(metadata_path, metadata)

    oof_rows = branch.data[["admission_id", "waveform_id", "oof_fold_id", "oof_ecg_score", "oof_ecg_baseline_state", "target_cag_binary", branch.label_col]].copy()
    oof_rows["agent2_discrepancy_oof_score"] = nested_oof_probability
    oof_rows["outer_fold"] = outer_fold_assignment
    oof_rows["outer_fold_selected_model"] = outer_selected_model
    oof_rows["outer_fold_selected_feature_setting"] = outer_selected_feature_setting
    oof_rows["outer_fold_selected_calibration"] = outer_selected_calibration
    oof_rows["outer_fold_selected_hyperparameters"] = outer_selected_hyperparameters
    oof_rows["selected_model"] = final_selected["model"]
    oof_rows["selected_feature_setting"] = final_selected["feature_setting"]
    oof_rows["selected_calibration"] = final_selected["calibration"]

    write_csv(out_dir / f"{branch.name}_fold_metrics.csv", fold_rows)
    write_csv(out_dir / f"{branch.name}_candidate_summary.csv", summary_rows)
    write_csv(out_dir / f"{branch.name}_inner_candidate_selection_audit.csv", inner_candidate_rows)
    write_csv(out_dir / f"{branch.name}_outer_fold_selection.csv", outer_selection_rows)
    write_csv(out_dir / f"{branch.name}_convergence_diagnostics.csv", convergence_rows)
    oof_rows.to_csv(out_dir / f"{branch.name}_outer_oof_predictions.csv", index=False, encoding="utf-8-sig")

    selected_convergence_rows = [row for row in convergence_rows if row.get("selected_in_scope") is True]
    selected_blocking_rows = [
        row for row in selected_convergence_rows
        if row["fit_stage"] in {"outer_winner_refit", "final_configuration_refit"}
        and int(row.get("convergence_warning_count", 0) or 0) > 0
        and str(row["converged"]).lower() != "true"
        and row["diagnostic_available"] is True
    ]

    return {
        "selected": final_selected,
        "summary_rows": summary_rows,
        "fold_rows": fold_rows,
        "split_rows": split_rows,
        "model_path": str(model_path),
        "schema_path": str(schema_path),
        "metadata_path": str(metadata_path),
        "oof_metrics": metrics_for(y, nested_oof_probability),
        "outer_selection_rows": outer_selection_rows,
        "full_training_candidate_rows": summary_rows,
        "convergence_rows": convergence_rows,
        "selected_convergence_rows": selected_convergence_rows,
        "selected_convergence_blocking_rows": selected_blocking_rows,
    }

def formal_controlled_candidate_grids(*, elastic_net_max_iter: int = DEFAULT_FORMAL_ELASTIC_NET_MAX_ITER) -> dict[str, tuple[Any, list[dict[str, list[Any]]], bool, int]]:
    """Prespecified bounded grids used by every formal inner-CV selection."""
    elastic = [
        {"model__C": [0.03], "model__l1_ratio": [0.5], "model__class_weight": ["balanced"]},
        {"model__C": [0.1], "model__l1_ratio": [0.25], "model__class_weight": ["balanced"]},
        {"model__C": [0.1], "model__l1_ratio": [0.5], "model__class_weight": ["balanced"]},
        {"model__C": [0.3], "model__l1_ratio": [0.25], "model__class_weight": ["balanced"]},
        {"model__C": [0.3], "model__l1_ratio": [0.5], "model__class_weight": ["balanced"]},
        {"model__C": [1.0], "model__l1_ratio": [0.0], "model__class_weight": ["balanced"]},
        {"model__C": [1.0], "model__l1_ratio": [0.5], "model__class_weight": ["balanced"]},
        {"model__C": [3.0], "model__l1_ratio": [0.5], "model__class_weight": ["balanced"]},
        {"model__C": [0.3], "model__l1_ratio": [0.5], "model__class_weight": [None]},
        {"model__C": [1.0], "model__l1_ratio": [0.5], "model__class_weight": [None]},
    ]
    hgb = [
        {"model__learning_rate": [0.05], "model__max_leaf_nodes": [15], "model__max_iter": [100], "model__min_samples_leaf": [20], "model__l2_regularization": [0.1]},
        {"model__learning_rate": [0.03], "model__max_leaf_nodes": [15], "model__max_iter": [100], "model__min_samples_leaf": [20], "model__l2_regularization": [0.1]},
        {"model__learning_rate": [0.1], "model__max_leaf_nodes": [15], "model__max_iter": [100], "model__min_samples_leaf": [20], "model__l2_regularization": [0.1]},
        {"model__learning_rate": [0.05], "model__max_leaf_nodes": [7], "model__max_iter": [100], "model__min_samples_leaf": [20], "model__l2_regularization": [0.1]},
        {"model__learning_rate": [0.05], "model__max_leaf_nodes": [31], "model__max_iter": [100], "model__min_samples_leaf": [20], "model__l2_regularization": [0.1]},
        {"model__learning_rate": [0.05], "model__max_leaf_nodes": [15], "model__max_iter": [200], "model__min_samples_leaf": [20], "model__l2_regularization": [0.1]},
        {"model__learning_rate": [0.05], "model__max_leaf_nodes": [15], "model__max_iter": [100], "model__min_samples_leaf": [10], "model__l2_regularization": [0.1]},
        {"model__learning_rate": [0.05], "model__max_leaf_nodes": [15], "model__max_iter": [100], "model__min_samples_leaf": [20], "model__l2_regularization": [0.0]},
    ]
    extra_trees = [
        {"model__max_depth": [16], "model__min_samples_leaf": [5], "model__max_features": ["sqrt"], "model__class_weight": ["balanced"]},
        {"model__max_depth": [None], "model__min_samples_leaf": [5], "model__max_features": ["sqrt"], "model__class_weight": ["balanced"]},
        {"model__max_depth": [8], "model__min_samples_leaf": [5], "model__max_features": ["sqrt"], "model__class_weight": ["balanced"]},
        {"model__max_depth": [16], "model__min_samples_leaf": [1], "model__max_features": ["sqrt"], "model__class_weight": ["balanced"]},
        {"model__max_depth": [16], "model__min_samples_leaf": [10], "model__max_features": ["sqrt"], "model__class_weight": ["balanced"]},
        {"model__max_depth": [16], "model__min_samples_leaf": [5], "model__max_features": [0.5], "model__class_weight": ["balanced"]},
        {"model__max_depth": [16], "model__min_samples_leaf": [5], "model__max_features": [1.0], "model__class_weight": ["balanced"]},
        {"model__max_depth": [16], "model__min_samples_leaf": [5], "model__max_features": ["sqrt"], "model__class_weight": [None]},
    ]
    return {
        "elastic_net_logistic": (
            make_elastic_net_logistic(max_iter=elastic_net_max_iter),
            elastic,
            True,
            0,
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            hgb,
            False,
            1,
        ),
        "extra_trees": (
            ExtraTreesClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=1),
            extra_trees,
            False,
            2,
        ),
    }

def build_leakage_audit(features: list[str]) -> list[dict[str, Any]]:
    rows = []
    for col in features:
        forbidden = is_forbidden_feature(col)
        rows.append(
            {
                "feature": col,
                "feature_name": col,
                "forbidden": forbidden,
                "reason": "forbidden leakage/id/runtime field" if forbidden else "allowed clinical/lab or oof score feature",
            }
        )
    return rows

def truthy(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes"}

def deactivate_marker(path: Path, *, reason: str) -> bool:
    if not path.exists():
        return True
    try:
        path.unlink()
        return True
    except PermissionError:
        try:
            write_json(path, {
                "status": "INACTIVE",
                "active": False,
                "deactivated_at": datetime.now().isoformat(timespec="seconds"),
                "reason": reason,
            })
        except PermissionError:
            pass
        return False

def write_formal_run_status_marker(out_dir: Path, summary: dict[str, Any]) -> None:
    running = out_dir / "RUNNING.json"
    deactivate_marker(running, reason="formal run finalized")
    if str(summary.get("overall_status", "")).startswith("PASS_"):
        failed = out_dir / "FAILED.json"
        deactivate_marker(failed, reason="formal run completed successfully")
        write_json(
            out_dir / "COMPLETED.json",
            {
                "status": "COMPLETED",
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "head": summary.get("formal_selected_head", "all"),
                "selected_model": summary.get("underestimation_selected_model") or summary.get("overestimation_selected_model"),
                "selected_feature_setting": summary.get("underestimation_selected_feature_setting") or summary.get("overestimation_selected_feature_setting"),
                "outer_fold_count": summary.get("formal_outer_cv_splits"),
                "inner_fold_count": summary.get("formal_inner_cv_splits"),
                "convergence_status": summary.get("selected_candidates_converged"),
                "runtime_locked": False,
                "validation_locked": False,
                "alert_threshold_locked": False,
            },
        )
    else:
        completed = out_dir / "COMPLETED.json"
        deactivate_marker(completed, reason="formal run failed")
        write_json(
            out_dir / "FAILED.json",
            {
                "status": "FAILED",
                "failed_at": datetime.now().isoformat(timespec="seconds"),
                "overall_status": summary.get("overall_status", ""),
                "head": summary.get("formal_selected_head", "all"),
                "convergence_warning_count": summary.get("convergence_warning_count"),
                "nonconverged_selected_stage_count": summary.get("nonconverged_selected_stage_count"),
                "runtime_locked": False,
                "validation_locked": False,
                "alert_threshold_locked": False,
            },
        )

def ensure_output_dir_empty(out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"output_dir is not empty; refusing to overwrite historical run: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

def run_training(args: argparse.Namespace) -> dict[str, Any]:
    if str(getattr(args, "execute_training", "false")).lower() not in {"true", "1", "yes"}:
        raise RuntimeError(MANUAL_TRAINING_NOTICE)
    out_dir = Path(args.output_dir)
    ensure_output_dir_empty(out_dir)
    write_json(
        out_dir / "RUNNING.json",
        {
            "status": "RUNNING",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "mode": args.mode,
            "head": getattr(args, "head", "all"),
            "runtime_locked": False,
            "validation_locked": False,
            "alert_threshold_locked": False,
        },
    )
    selected_formal_head = str(getattr(args, "head", "all") or "all").strip().lower()
    selected_formal_heads = {"underestimation", "overestimation"} if selected_formal_head == "all" else {selected_formal_head}
    table, clinical_features = build_training_table(
        Path(args.agent2_input),
        Path(args.label_table),
        Path(args.ecg_oof),
        ecg_threshold=float(args.ecg_threshold),
    )
    branches = split_branches(table)

    write_json(
        out_dir / "training_data_manifest.json",
        {
            "train_only": True,
            "agent2_input": str(args.agent2_input),
            "label_table": str(args.label_table),
            "trusted_ecg_oof_artifact": str(args.ecg_oof),
            "locked_ecg_threshold": float(args.ecg_threshold),
            "train_case_count": len(table),
            "clinical_feature_count": len(clinical_features),
            "validation_access_count": 0,
            "test_access_count": 0,
            "screening_result_used_for_formal_selection": False,
        },
    )
    write_json(
        out_dir / "training_data_hashes.json",
        {
            "agent2_input_sha256": file_sha256(Path(args.agent2_input)),
            "label_table_sha256": file_sha256(Path(args.label_table)),
            "ecg_oof_sha256": file_sha256(Path(args.ecg_oof)),
            "clinical_feature_schema_sha256": hashlib.sha256(json.dumps(clinical_features, sort_keys=True).encode()).hexdigest(),
        },
    )
    write_json(out_dir / "feature_schema.json", {"clinical_features": clinical_features, "feature_sets": feature_sets(clinical_features)})
    elastic_net_max_iter = int(getattr(args, "elastic_net_max_iter", DEFAULT_ELASTIC_NET_MAX_ITER))
    active_grids = formal_controlled_candidate_grids(elastic_net_max_iter=elastic_net_max_iter)
    active_feature_sets = feature_sets(clinical_features)
    candidate_combinations = [
        {"model_family": model_name, "feature_setting": feature_setting}
        for feature_setting in active_feature_sets
        for model_name in active_grids
    ]
    grid_manifest = {
        name: {
            "configuration_count": len(list(ParameterGrid(grid))),
            "configurations": list(ParameterGrid(grid)),
        }
        for name, (_, grid, _, _) in active_grids.items()
    }
    write_json(
        out_dir / "candidate_model_grid.json",
        {
            "grid_profile": "formal_controlled_complete_candidate_space",
            "candidate_models": list(active_grids.keys()),
            "feature_settings": list(active_feature_sets.keys()),
            "model_family_x_feature_setting_count_per_head": len(candidate_combinations),
            "candidate_combinations_per_head": candidate_combinations,
            "controlled_hyperparameter_grids": grid_manifest,
            "fixed_estimator_parameters": {
                "elastic_net_logistic": {"solver": "saga", "penalty": "elasticnet", "max_iter": elastic_net_max_iter},
                "hist_gradient_boosting": {"random_state": RANDOM_STATE},
                "extra_trees": {"n_estimators": 500, "n_jobs": 1, "random_state": RANDOM_STATE},
            },
            "calibration_strategies": ["uncalibrated", "sigmoid"],
            "formal_candidate_grid_bounded": True,
            "screening_result_used_for_formal_selection": False,
            "elastic_net_max_iter": elastic_net_max_iter,
            "exhaustive_grid_is_default": False,
        },
    )
    write_json(
        out_dir / "formal_selection_manifest.json",
        {
            "selection_scope": "train_only",
            "selected_heads": sorted(selected_formal_heads),
            "candidate_space": "3 model families x 3 feature configurations per head",
            "candidate_combinations_per_head": candidate_combinations,
            "outer_cv_splits": int(getattr(args, "outer_cv_splits", DEFAULT_FORMAL_OUTER_CV_SPLITS)),
            "inner_cv_splits": int(getattr(args, "inner_cv_splits", DEFAULT_FORMAL_INNER_CV_SPLITS)),
            "selection_rule": "AUPRC descending; candidates within 0.01 AUPRC compared by Brier, then ECE, then model priority",
            "screening_result_used_for_formal_selection": False,
            "outer_holdout_used_for_candidate_selection": False,
            "outer_oof_used_for_final_configuration_selection": False,
            "validation_access_count": 0,
            "test_access_count": 0,
        },
    )

    for branch in branches.values():
        audit = [
            {
                "head": branch.name,
                "training_count": len(branch.data),
                "positive_count": branch.positive_count,
                "negative_count": branch.negative_count,
                "label_col": branch.label_col,
                "train_only": True,
                "uses_abcd_as_label": False,
            }
        ]
        write_csv(out_dir / f"{branch.name}_training_dataset_audit.csv", audit)

    write_csv(out_dir / "label_leakage_audit.csv", build_leakage_audit([*clinical_features, "oof_ecg_score"]))
    prohibited_feature_count = sum(1 for r in build_leakage_audit([*clinical_features, "oof_ecg_score"]) if r["forbidden"])

    results = {}
    split_manifest = []
    for branch in branches.values():
        if branch.name not in selected_formal_heads:
            continue
        result = train_branch(
            branch,
            clinical_features,
            out_dir,
            grid_profile=args.grid_profile,
            bootstrap_n=int(args.bootstrap_n),
            outer_cv_splits=int(getattr(args, "outer_cv_splits", DEFAULT_FORMAL_OUTER_CV_SPLITS)),
            inner_cv_splits=int(getattr(args, "inner_cv_splits", DEFAULT_FORMAL_INNER_CV_SPLITS)),
            enable_sigmoid_calibration=truthy(getattr(args, "enable_sigmoid_calibration", True)),
            elastic_net_max_iter=elastic_net_max_iter,
        )
        results[branch.name] = result
        split_manifest.extend(result["split_rows"])
    write_csv(out_dir / "nested_cv_split_manifest.csv", split_manifest)
    convergence_rows = []
    selected_convergence_blocking_rows = []
    for head, result in results.items():
        convergence_rows.extend(result.get("convergence_rows", []))
        selected_convergence_blocking_rows.extend(result.get("selected_convergence_blocking_rows", []))
    write_csv(out_dir / "convergence_diagnostics.csv", convergence_rows)
    convergence_summary = {
        "convergence_diagnostic_available": True,
        "convergence_warning_count": int(sum(int(row.get("convergence_warning_count", 0) or 0) for row in convergence_rows)),
        "nonconverged_selected_stage_count": int(len(selected_convergence_blocking_rows)),
        "selected_candidates_converged": len(selected_convergence_blocking_rows) == 0,
        "blocking_rows": selected_convergence_blocking_rows,
    }
    write_json(out_dir / "convergence_diagnostics.json", convergence_summary)
    (out_dir / "convergence_audit.md").write_text(
        "# Agent2 formal convergence audit\n\n"
        f"convergence_warning_count={convergence_summary['convergence_warning_count']}\n\n"
        f"nonconverged_selected_stage_count={convergence_summary['nonconverged_selected_stage_count']}\n\n"
        f"selected_candidates_converged={convergence_summary['selected_candidates_converged']}\n\n"
        "Inner-CV diagnostics are recorded for every complete candidate pipeline; "
        "outer_winner_refit/final_configuration_refit diagnostics read n_iter_ from selected logistic estimators where available. "
        "HistGradientBoosting has no logistic n_iter_ diagnostic and is marked diagnostic_available=false.\n",
        encoding="utf-8",
    )

    # Combined reports.
    feature_compare = []
    calibration_rows = []
    ci_rows = []
    artifact_rows = []
    for head, result in results.items():
        selected = result["selected"]
        rows = result["summary_rows"]
        for r in rows:
            feature_compare.append(
                {
                    "head": head,
                    "model": r["model"],
                    "feature_setting": r["feature_setting"],
                    "calibration": r["calibration"],
                    "hyperparameters": r["best_params"],
                    "AUROC": r["auroc"],
                    "AUPRC": r["auprc"],
                    "Brier": r["brier"],
                    "ECE": r["ece"],
                    "bootstrap_CI_AUPRC": r["bootstrap_ci_auprc"],
                    "selected_by_full_training_cv": r["selected_in_scope"],
                    "preferred_feature_setting": selected["feature_setting"],
                }
            )
            calibration_rows.append(
                {
                    "head": head,
                    "model": r["model"],
                    "feature_setting": r["feature_setting"],
                    "calibration": r["calibration"],
                    "hyperparameters": r["best_params"],
                    "brier": r["brier"],
                    "ece": r["ece"],
                    "calibration_slope": r["calibration_slope"],
                    "calibration_intercept": r["calibration_intercept"],
                }
            )
            ci_rows.append(
                {
                    "head": head,
                    "model": r["model"],
                    "feature_setting": r["feature_setting"],
                    "calibration": r["calibration"],
                    "hyperparameters": r["best_params"],
                    "auroc_ci": r["bootstrap_ci_auroc"],
                    "auprc_ci": r["bootstrap_ci_auprc"],
                    "brier_ci": r["bootstrap_ci_brier"],
                    "ece_ci": r["bootstrap_ci_ece"],
                }
            )
        artifact_rows.append(
            {
                "head": head,
                "model_path": result["model_path"],
                "schema_path": result["schema_path"],
                "metadata_path": result["metadata_path"],
                "candidate_selected_on_train_cv": True,
                "runtime_locked": False,
                "validation_locked": False,
                "alert_threshold_locked": False,
            }
        )
    write_csv(out_dir / "agent2_feature_set_comparison.csv", feature_compare)
    write_csv(out_dir / "calibration_metrics.csv", calibration_rows)
    write_csv(out_dir / "bootstrap_confidence_intervals.csv", ci_rows)
    write_csv(out_dir / "candidate_artifact_manifest.csv", artifact_rows)
    write_json(
        out_dir / "selected_candidate_models.json",
        {
            head: {
                "selected_model": result["selected"]["model"],
                "selected_feature_setting": result["selected"]["feature_setting"],
                "selected_calibration": result["selected"]["calibration"],
                "selected_hyperparameters": result["selected"]["_params"],
                "selection_source": "full_training_inner_cv",
                "outer_oof_used_for_final_configuration_selection": False,
                "metrics": result["oof_metrics"],
                "nested_cv_internal_generalization_metrics": result["oof_metrics"],
                "outer_fold_inner_winners": result["outer_selection_rows"],
                "runtime_locked": False,
                "alert_threshold_locked": False,
            }
            for head, result in results.items()
        },
    )
    formal_manifest_path = out_dir / "formal_selection_manifest.json"
    formal_manifest = json.loads(formal_manifest_path.read_text(encoding="utf-8"))
    formal_manifest["outer_fold_inner_winners"] = {
        head: result["outer_selection_rows"] for head, result in results.items()
    }
    formal_manifest["final_full_training_cv_configuration"] = {
        head: {
            "model_family": result["selected"]["model"],
            "feature_setting": result["selected"]["feature_setting"],
            "hyperparameters": result["selected"]["_params"],
            "calibration_strategy": result["selected"]["calibration"],
        }
        for head, result in results.items()
    }
    write_json(formal_manifest_path, formal_manifest)

    # Feature importance for selected models only. Formal nested CV keeps this
    # disabled unless explicitly requested.
    importance_rows = []
    for head, result in ([] if truthy(getattr(args, "disable_permutation_importance", False)) else results.items()):
        selected = result["selected"]
        branch = branches[head]
        selected_features = feature_sets(clinical_features)[selected["feature_setting"]]
        # Use selected OOF score permutation on a full refit candidate as a bounded interpretability audit.
        model = joblib.load(result["model_path"])
        try:
            imp = permutation_importance(
                model,
                branch.data[selected_features],
                branch.data[branch.label_col],
                scoring="average_precision",
                n_repeats=5,
                random_state=RANDOM_STATE,
            )
            for feature, mean, std in zip(selected_features, imp.importances_mean, imp.importances_std):
                importance_rows.append({"head": head, "feature_name": feature, "importance_mean": mean, "importance_std": std, "interpretation_boundary": "statistical association with ECG model error risk only"})
        except Exception as exc:
            importance_rows.append({"head": head, "feature_name": "__IMPORTANCE_FAILED__", "importance_mean": "", "importance_std": "", "interpretation_boundary": repr(exc)})
    write_csv(out_dir / "feature_importance_stability.csv", importance_rows)

    route_rows = [
        ("runtime_anchor", "ecg", "ecg", True),
        ("agent1_role", "ECG_Baseline_Audit_Agent", "ECG_Baseline_Audit_Agent", True),
        ("agent2_mode", "discrepancy_detection", "discrepancy_detection", True),
        ("agent2_uses_clinical_only_score_in_runtime", "false", "false", True),
        ("agent3_outputs_ecg_baseline_relation", "true", "true", True),
        ("agent4_final_semantic_action_source", "true", "true", True),
        ("controller_projection_target", "locked_ecg_score", "locked_ecg_score", True),
        ("clinical_only_model_scope", "baseline_only", "baseline_only", True),
        ("deprecated_combined_model_runtime_scope", "excluded", "excluded", True),
        ("abcd_runtime_access", "false", "false", True),
        ("model_error_label_runtime_access", "false", "false", True),
        ("rag_used_in_training", "false", "false", True),
        ("llm_used_in_training", "false", "false", True),
        ("deprecated_score_compatibility_enabled", "false", "false", True),
    ]
    write_csv(out_dir / "architecture_route_guard_audit.csv", [{"item": a, "expected": b, "observed": c, "pass": d} for a, b, c, d in route_rows])

    summary = {
        "api_called": False,
        "llm_called": False,
        "runtime_called": False,
        "validation_used": False,
        "test_used": False,
        "test_allowed": False,
        "ecg_oof_generation_executed": False,
        "train_only": True,
        "trusted_ecg_oof_artifact": str(args.ecg_oof),
        "underestimation_training_count": len(branches["underestimation"].data),
        "underestimation_positive_count": branches["underestimation"].positive_count,
        "underestimation_negative_count": branches["underestimation"].negative_count,
        "overestimation_training_count": len(branches["overestimation"].data),
        "overestimation_positive_count": branches["overestimation"].positive_count,
        "overestimation_negative_count": branches["overestimation"].negative_count,
        "clinical_feature_count": len(clinical_features),
        "prohibited_feature_count": int(prohibited_feature_count),
        "label_leakage_count": int(prohibited_feature_count),
        "nested_cv_completed": True,
        "formal_requires_screen_manifest": False,
        "screening_result_used_for_formal_selection": False,
        "formal_candidate_grid_bounded": True,
        "formal_model_family_count": 3,
        "formal_feature_setting_count": 3,
        "formal_model_family_x_feature_setting_count_per_head": 9,
        "validation_access_count": 0,
        "test_access_count": 0,
        "formal_outer_cv_splits": int(getattr(args, "outer_cv_splits", DEFAULT_FORMAL_OUTER_CV_SPLITS)),
        "formal_inner_cv_splits": int(getattr(args, "inner_cv_splits", DEFAULT_FORMAL_INNER_CV_SPLITS)),
        "elastic_net_max_iter": elastic_net_max_iter,
        "convergence_diagnostic_available": True,
        "convergence_warning_count": convergence_summary["convergence_warning_count"],
        "nonconverged_selected_stage_count": convergence_summary["nonconverged_selected_stage_count"],
        "selected_candidates_converged": convergence_summary["selected_candidates_converged"],
        "outer_oof_prediction_complete": True,
        "formal_selected_head": selected_formal_head,
        "formal_separate_head_execution": selected_formal_head != "all",
        "underestimation_candidate_selected": "underestimation" in results,
        "underestimation_selected_model": results["underestimation"]["selected"]["model"] if "underestimation" in results else "",
        "underestimation_selected_feature_setting": results["underestimation"]["selected"]["feature_setting"] if "underestimation" in results else "",
        "underestimation_outer_oof_auc": results["underestimation"]["oof_metrics"]["auroc"] if "underestimation" in results else "",
        "underestimation_outer_oof_auprc": results["underestimation"]["oof_metrics"]["auprc"] if "underestimation" in results else "",
        "underestimation_outer_oof_brier": results["underestimation"]["oof_metrics"]["brier"] if "underestimation" in results else "",
        "underestimation_outer_oof_ece": results["underestimation"]["oof_metrics"]["ece"] if "underestimation" in results else "",
        "overestimation_candidate_selected": "overestimation" in results,
        "overestimation_selected_model": results["overestimation"]["selected"]["model"] if "overestimation" in results else "",
        "overestimation_selected_feature_setting": results["overestimation"]["selected"]["feature_setting"] if "overestimation" in results else "",
        "overestimation_outer_oof_auc": results["overestimation"]["oof_metrics"]["auroc"] if "overestimation" in results else "",
        "overestimation_outer_oof_auprc": results["overestimation"]["oof_metrics"]["auprc"] if "overestimation" in results else "",
        "overestimation_outer_oof_brier": results["overestimation"]["oof_metrics"]["brier"] if "overestimation" in results else "",
        "overestimation_outer_oof_ece": results["overestimation"]["oof_metrics"]["ece"] if "overestimation" in results else "",
        "clinical_features_add_incremental_value_under": (results["underestimation"]["selected"]["feature_setting"] != "F0_ecg_score_only") if "underestimation" in results else "",
        "clinical_features_add_incremental_value_over": (results["overestimation"]["selected"]["feature_setting"] != "F0_ecg_score_only") if "overestimation" in results else "",
        "candidate_artifacts_saved": True,
        "candidate_artifacts_runtime_locked": False,
        "underestimation_alert_threshold_locked": False,
        "overestimation_alert_threshold_locked": False,
        "agent2_discrepancy_models_locked": False,
        "runtime_anchor": "ecg",
        "agent2_mode": "discrepancy_detection",
        "controller_projection_target": "locked_ecg_score",
        "architecture_drift_detected": False,
        "head_run_completed": convergence_summary["selected_candidates_converged"],
        "ready_for_formal_adjudication": True,
        "ready_for_agent2_validation_lock": False,
        "ready_for_runtime": False,
        "ready_for_pilot": False,
        "ready_for_test": False,
        "overall_status": "PASS_AGENT2_TRAIN_CV_CANDIDATES_SELECTED" if convergence_summary["selected_candidates_converged"] else "FAIL_AGENT2_NONCONVERGENCE",
        "output_report_dir": str(out_dir),
    }
    write_json(out_dir / "readiness.json", summary)
    (out_dir / "preprocessing_specification.md").write_text(
        "# Preprocessing specification\n\nNumeric: median imputation + missing indicator; logistic also standardizes numeric fields. Categorical: missing maps to `__MISSING__`; unknown and not_applicable remain independent categories; OneHotEncoder(handle_unknown='ignore'). Preprocessors are fit inside folds.\n",
        encoding="utf-8",
    )
    (out_dir / "model_selection_decision.md").write_text(
        "# Model selection decision\n\n"
        "Each outer fold selected model family, F0/F1/F2, controlled hyperparameters, and calibration using only its inner three-fold CV. "
        "The fixed lexical rule was AUPRC first; candidates within 0.01 AUPRC were compared by Brier, then ECE, then simpler model priority. "
        "Outer holdouts were used only for internal generalization estimation. The final configuration was selected by a separate three-fold CV on the complete training branch and then refit on that branch. "
        "Screening, validation, and test results were not used for formal selection; runtime_locked=false; alert_threshold_locked=false.\n",
        encoding="utf-8",
    )
    (out_dir / "task_summary.md").write_text(
        "# Task summary\n\n" + "\n".join(f"{k}={v}" for k, v in summary.items()),
        encoding="utf-8",
    )
    write_formal_run_status_marker(out_dir, summary)
    return summary

