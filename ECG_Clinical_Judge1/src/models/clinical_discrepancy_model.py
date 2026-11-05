"""Interfaces for Agent2 ECG-baseline discrepancy models.

The migration defines two deterministic, supervised heads:

* underestimation head: active only when ECG baseline is negative
* overestimation head: active only when ECG baseline is positive

This file deliberately does not train or load a fallback model.  Runtime must
fail fast until locked artifacts are supplied by a later audited training step.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


AGENT2_MODEL_NOT_LOCKED = "AGENT2_DISCREPANCY_MODEL_NOT_LOCKED"


@dataclass(frozen=True)
class ClinicalDiscrepancyPrediction:
    schema_version: str
    agent: str
    selected_discrepancy_head: str
    discrepancy_risk_score: float
    discrepancy_risk_band: str
    discrepancy_direction: str
    model_confidence: str
    feature_contributions: list[dict[str, Any]]
    missingness_reliability: str
    model_version: str


def select_discrepancy_head(ecg_baseline_state: str) -> str:
    if ecg_baseline_state == "ECG_MODEL_NEGATIVE":
        return "UNDERESTIMATION"
    if ecg_baseline_state == "ECG_MODEL_POSITIVE":
        return "OVERESTIMATION"
    raise RuntimeError(f"INVALID_ECG_BASELINE_STATE_FOR_AGENT2: {ecg_baseline_state!r}")


def assert_agent2_discrepancy_models_locked(config: dict[str, Any] | None) -> None:
    cfg = config or {}
    if cfg.get("mock_allowed") is True or cfg.get("fallback_allowed") is True:
        raise RuntimeError("AGENT2_MOCK_OR_FALLBACK_NOT_ALLOWED")
    if cfg.get("runtime_lock_manifest"):
        manifest_path = _resolve_path(cfg.get("runtime_lock_manifest"))
        if not manifest_path.exists():
            raise RuntimeError(f"AGENT2_RUNTIME_LOCK_MANIFEST_MISSING: {manifest_path}")
        return
    under = cfg.get("underestimation_model_artifact")
    over = cfg.get("overestimation_model_artifact")
    if not under or not over:
        raise RuntimeError(AGENT2_MODEL_NOT_LOCKED)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_path(path_value: Any) -> Path:
    root = Path(__file__).resolve().parents[2]
    path = Path(str(path_value))
    return path if path.is_absolute() else root / path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_manifest_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise RuntimeError(f"AGENT2_LOCKED_ARTIFACT_MANIFEST_MISSING: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {row.get("head", ""): row for row in csv.DictReader(f)}


def _positive_class_index(model: Any) -> int:
    classes = list(getattr(model, "classes_", []))
    for target in (1, "1", True):
        if target in classes:
            return classes.index(target)
    raise RuntimeError(f"AGENT2_POSITIVE_CLASS_NOT_FOUND: classes_={classes!r}")


def _numeric_or_nan(value: Any) -> Any:
    if value is None or value == "":
        return float("nan")
    try:
        number = float(value)
    except Exception:
        return value
    if not math.isfinite(number):
        return float("nan")
    return number


def _build_feature_row(context: dict[str, Any], features: list[str]) -> dict[str, Any]:
    clinical = context.get("agent2_clinical_evidence") or context.get("clinical_evidence_packet") or {}
    ecg_anchor = context.get("ecg_anchor_context") or {}
    row: dict[str, Any] = {}
    missing: list[str] = []
    for feature in features:
        if feature == "oof_ecg_score":
            if "locked_ecg_score" not in ecg_anchor:
                missing.append(feature)
            else:
                row[feature] = _numeric_or_nan(ecg_anchor.get("locked_ecg_score"))
            continue
        if feature in clinical:
            row[feature] = clinical.get(feature)
        elif feature in context:
            row[feature] = context.get(feature)
        else:
            missing.append(feature)
    if missing:
        raise RuntimeError("AGENT2_REQUIRED_FEATURES_MISSING: " + ",".join(missing))
    return row


def _verify_validation_lock_source(config: dict[str, Any]) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, dict[str, str]]]:
    lock_dir = _resolve_path(config.get("validation_lock_source", ""))
    summary = _read_json(lock_dir / "validation_lock_summary.json")
    manifest = _read_json(lock_dir / "validation_lock_manifest.json")
    rows = _read_manifest_rows(lock_dir / "locked_artifact_manifest.csv")
    if summary.get("overall_status") != "PASS_AGENT2_VALIDATION_LOCK_COMPLETED":
        raise RuntimeError("AGENT2_VALIDATION_LOCK_STATUS_NOT_PASS")
    if summary.get("test_used") is not False or summary.get("test_allowed") is not False:
        raise RuntimeError("AGENT2_VALIDATION_LOCK_TEST_GATE_NOT_FALSE")
    if manifest.get("validation_lock_completed") is not True:
        raise RuntimeError("AGENT2_VALIDATION_LOCK_MANIFEST_NOT_COMPLETED")
    if "agent2_formal_nested_cv_run003" not in str(manifest.get("source_formal_run_dir", "")):
        raise RuntimeError("AGENT2_FORMAL_SOURCE_NOT_RUN003")
    return lock_dir, summary, manifest, rows


def _verify_runtime_lock_manifest(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = _resolve_path(config.get("runtime_lock_manifest", ""))
    manifest = _read_json(manifest_path)
    if not manifest:
        raise RuntimeError("AGENT2_RUNTIME_LOCK_MANIFEST_EMPTY")
    required_true = [
        "no_mock",
        "no_fallback",
        "scientific_agent2_models_runtime_locked",
        "scientific_agent2_thresholds_locked",
        "scientific_agent2_risk_bands_locked",
    ]
    for key in required_true:
        if manifest.get(key) is not True:
            raise RuntimeError(f"AGENT2_RUNTIME_LOCK_MANIFEST_FIELD_NOT_TRUE:{key}")
    if manifest.get("test_used") is not False:
        raise RuntimeError("AGENT2_RUNTIME_LOCK_TEST_USED_NOT_FALSE")
    threshold_lock = _resolve_path(config.get("threshold_lock_config", ""))
    if not threshold_lock.exists():
        raise RuntimeError(f"AGENT2_THRESHOLD_LOCK_CONFIG_MISSING: {threshold_lock}")
    if _sha256_file(threshold_lock) != manifest.get("threshold_lock_hash"):
        raise RuntimeError("AGENT2_THRESHOLD_LOCK_HASH_MISMATCH")
    return manifest


def _risk_band(score: float, lower: float, high: float) -> str:
    if not math.isfinite(score) or score < 0 or score > 1:
        raise RuntimeError(f"AGENT2_DISCREPANCY_SCORE_OUT_OF_RANGE: {score!r}")
    if score < lower:
        return "LOW"
    if score < high:
        return "INTERMEDIATE"
    return "HIGH"


def _verify_head_artifacts(
    *,
    config: dict[str, Any],
    head: str,
    manifest_rows: dict[str, dict[str, str]],
) -> tuple[Path, Path, dict[str, Any], dict[str, Any], str, str]:
    model_path = _resolve_path(config.get(f"{head}_model_artifact", ""))
    schema_path = _resolve_path(config.get(f"{head}_feature_schema", ""))
    metadata_path = _resolve_path(config.get(f"{head}_model_metadata", ""))
    for path, role in [(model_path, "model"), (schema_path, "schema"), (metadata_path, "metadata")]:
        if not path.exists() or path.stat().st_size <= 0:
            raise RuntimeError(f"AGENT2_{head.upper()}_{role.upper()}_MISSING_OR_EMPTY: {path}")
    manifest_row = manifest_rows.get(head, {})
    model_hash = _sha256_file(model_path)
    if model_hash != manifest_row.get("locked_model_sha256"):
        raise RuntimeError(f"AGENT2_{head.upper()}_MODEL_HASH_MISMATCH")
    schema = _read_json(schema_path)
    metadata = _read_json(metadata_path)
    return model_path, schema_path, schema, metadata, model_hash, _sha256_file(schema_path)


def predict_agent2_discrepancy_from_locked_artifacts(
    context: dict[str, Any],
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Run the locked Agent2 discrepancy head for the ECG-anchor state.

    This loads audited validation-lock artifacts only.  It never fits, calibrates,
    selects thresholds, or reads label fields.
    """

    cfg = config or {}
    assert_agent2_discrepancy_models_locked(cfg)
    if cfg.get("runtime_lock_manifest"):
        return _predict_agent2_discrepancy_from_runtime_lock(context, cfg)
    lock_dir, _summary, _manifest, manifest_rows = _verify_validation_lock_source(cfg)
    ecg_anchor = context.get("ecg_anchor_context") or {}
    active_head = select_discrepancy_head(str(ecg_anchor.get("ecg_baseline_state", ""))).lower()
    outputs: dict[str, Any] = {
        "agent2_validation_lock_source": str(lock_dir),
        "agent2_inference_status": "pass",
        "selected_discrepancy_head": active_head.upper(),
    }
    active_score: float | None = None
    active_threshold: float | None = None
    for head in ("underestimation", "overestimation"):
        model_path, schema_path, schema, metadata, model_hash, schema_hash = _verify_head_artifacts(
            config=cfg,
            head=head,
            manifest_rows=manifest_rows,
        )
        threshold = float(cfg.get(f"{head}_alert_threshold"))
        features = list(schema.get("features", []))
        applicable = head == active_head
        outputs[f"{head}_threshold"] = threshold
        outputs[f"{head}_applicable"] = applicable
        outputs[f"{head}_model_artifact_hash"] = model_hash
        outputs[f"{head}_schema_hash"] = schema_hash
        outputs[f"{head}_feature_setting"] = str(metadata.get("selected_feature_setting", ""))
        if not applicable:
            outputs[f"{head}_alert"] = False
            continue
        row = _build_feature_row(context, features)
        frame = pd.DataFrame([row], columns=features)
        model = joblib.load(model_path)
        proba = model.predict_proba(frame)
        pos_idx = _positive_class_index(model)
        probability = float(proba[0][pos_idx])
        outputs[f"{head}_alert"] = bool(probability >= threshold)
        active_score = probability
        active_threshold = threshold
    if active_score is None or active_threshold is None:
        raise RuntimeError("AGENT2_ACTIVE_DISCREPANCY_SCORE_MISSING")
    direction = (
        "POSSIBLE_ECG_UNDERESTIMATION"
        if active_head == "underestimation"
        else "POSSIBLE_ECG_OVERESTIMATION"
    )
    risk_band = "HIGH" if active_score >= active_threshold else "LOW"
    outputs.update({
        "discrepancy_risk_score": active_score,
        "discrepancy_risk_band": risk_band,
        "discrepancy_direction": direction,
        "discrepancy_confidence": "HIGH" if risk_band == "HIGH" else "LOW",
        "model_confidence": "HIGH" if risk_band == "HIGH" else "LOW",
        "model_version": f"v2_8_agent2_{active_head}_validation_lock_compatibility",
        "model_hash": outputs[f"{active_head}_model_artifact_hash"],
        "schema_hash": outputs[f"{active_head}_schema_hash"],
        "threshold_lock_hash": "",
        "feature_setting": outputs[f"{active_head}_feature_setting"],
        "runtime_locked": False,
        "mock_used": False,
        "fallback_used": False,
    })
    return outputs


def _predict_agent2_discrepancy_from_runtime_lock(context: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    manifest = _verify_runtime_lock_manifest(config)
    ecg_anchor = context.get("ecg_anchor_context") or {}
    locked_ecg_score = float(ecg_anchor.get("locked_ecg_score"))
    locked_ecg_threshold = float(ecg_anchor.get("locked_ecg_threshold", manifest.get("ecg_anchor_threshold")))
    if not math.isfinite(locked_ecg_score) or locked_ecg_score < 0 or locked_ecg_score > 1:
        raise RuntimeError(f"AGENT2_LOCKED_ECG_SCORE_OUT_OF_RANGE: {locked_ecg_score!r}")
    if not math.isfinite(locked_ecg_threshold) or locked_ecg_threshold < 0 or locked_ecg_threshold > 1:
        raise RuntimeError(f"AGENT2_LOCKED_ECG_THRESHOLD_OUT_OF_RANGE: {locked_ecg_threshold!r}")
    baseline_state = "ECG_MODEL_POSITIVE" if locked_ecg_score >= locked_ecg_threshold else "ECG_MODEL_NEGATIVE"
    declared_state = str(ecg_anchor.get("ecg_baseline_state", baseline_state))
    if declared_state and declared_state != baseline_state:
        raise RuntimeError("AGENT2_ECG_BASELINE_STATE_SCORE_MISMATCH")
    active_head = select_discrepancy_head(baseline_state).lower()
    if active_head == "underestimation":
        model_path = _resolve_path(manifest["under_model_path"])
        schema_path = _resolve_path(manifest["under_schema_path"])
        threshold = float(manifest["under_alert_threshold"])
        high_threshold = float(manifest["under_high_risk_threshold"])
        direction = "POSSIBLE_ECG_UNDERESTIMATION"
        feature_setting = manifest["under_feature_setting"]
        model_hash_expected = manifest["under_model_hash"]
        schema_hash_expected = manifest["under_schema_hash"]
    else:
        model_path = _resolve_path(manifest["over_model_path"])
        schema_path = _resolve_path(manifest["over_schema_path"])
        threshold = float(manifest["over_review_threshold"])
        high_threshold = float(manifest["over_high_risk_threshold"])
        direction = "POSSIBLE_ECG_OVERESTIMATION"
        feature_setting = manifest["over_feature_setting"]
        model_hash_expected = manifest["over_model_hash"]
        schema_hash_expected = manifest["over_schema_hash"]
    if not model_path.exists() or not schema_path.exists():
        raise RuntimeError("AGENT2_RUNTIME_ARTIFACT_MISSING")
    model_hash = _sha256_file(model_path)
    schema_hash = _sha256_file(schema_path)
    if model_hash != model_hash_expected:
        raise RuntimeError("AGENT2_RUNTIME_MODEL_HASH_MISMATCH")
    if schema_hash != schema_hash_expected:
        raise RuntimeError("AGENT2_RUNTIME_SCHEMA_HASH_MISMATCH")
    schema = _read_json(schema_path)
    features = list(schema.get("features", []))
    row = _build_feature_row(context, features)
    frame = pd.DataFrame([row], columns=features)
    model = joblib.load(model_path)
    proba = model.predict_proba(frame)
    pos_idx = _positive_class_index(model)
    score = float(proba[0][pos_idx])
    risk_band = _risk_band(score, threshold, high_threshold)
    active_alert = bool(score >= threshold)
    confidence = "HIGH" if risk_band == "HIGH" else "MODERATE" if risk_band == "INTERMEDIATE" else "LOW"
    return {
        "agent2_runtime_lock_manifest": str(_resolve_path(config.get("runtime_lock_manifest"))),
        "agent2_inference_status": "pass",
        "admission_id": context.get("admission_id", ""),
        "ecg_baseline_state": baseline_state,
        "selected_discrepancy_head": active_head.upper(),
        "discrepancy_risk_score": score,
        "discrepancy_risk_band": risk_band,
        "underestimation_alert": active_alert if active_head == "underestimation" else False,
        "overestimation_alert": active_alert if active_head == "overestimation" else False,
        "discrepancy_direction": direction,
        "discrepancy_confidence": confidence,
        "model_confidence": confidence,
        "model_version": f"{manifest.get('dataset_version', 'v2_9')}_agent2_{active_head}_scientific_runtime_lock",
        "model_hash": model_hash,
        "schema_hash": schema_hash,
        "threshold_lock_hash": manifest["threshold_lock_hash"],
        "feature_setting": feature_setting,
        "missing_feature_summary": [],
        "evidence_contributions": [],
        "runtime_locked": True,
        "mock_used": False,
        "fallback_used": False,
    }


def build_agent2_oof_training_record_schema() -> list[str]:
    return [
        "admission_id",
        "fold_id",
        "oof_ecg_score",
        "oof_threshold_source",
        "oof_ecg_baseline_state",
        "oof_score_margin",
        "target_cag_binary",
        "ecg_model_version",
    ]


def build_under_over_label_spec() -> dict[str, dict[str, str]]:
    return {
        "underestimation": {
            "row_filter": "oof_ecg_baseline_state == ECG_MODEL_NEGATIVE",
            "label": "train_underestimation_label = 1 if target_cag_binary == 1 else 0",
        },
        "overestimation": {
            "row_filter": "oof_ecg_baseline_state == ECG_MODEL_POSITIVE",
            "label": "train_overestimation_label = 1 if target_cag_binary == 0 else 0",
        },
    }
