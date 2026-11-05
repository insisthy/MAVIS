"""ECG-anchor runtime contract for the Dataset-v4 correction architecture.

This module is intentionally small and deterministic.  It does not train
models, call LLMs, or infer labels; it only normalizes the active runtime
anchor and rejects deprecated non-ECG score fields.
"""

from __future__ import annotations

from typing import Any


ECG_CORRECTION_SCHEMA_VERSION = "ecg_correction_v1"

ECG_BASELINE_POSITIVE = "ECG_MODEL_POSITIVE"
ECG_BASELINE_NEGATIVE = "ECG_MODEL_NEGATIVE"
ECG_ONLY_CAG_PREDICTED_POSITIVE = "PREDICTED_CAG_POSITIVE"
ECG_ONLY_CAG_PREDICTED_NEGATIVE = "PREDICTED_CAG_NEGATIVE"

ECG_BASELINE_RELATIONS = {
    "ECG_BASELINE_SUPPORTED",
    "ECG_BASELINE_POSSIBLY_UNDERSTATED",
    "ECG_BASELINE_POSSIBLY_OVERSTATED",
    "ECG_BASELINE_RELATION_UNRESOLVED",
}

DEPRECATED_COMBINED_SCORE_RUNTIME_FIELDS = {
    "locked_fusion_score",
    "fusion_score",
    "locked_fusion_threshold",
    "fusion_threshold",
    "fusion_margin",
    "fusion_probability_margin",
    "fusion_logit_margin",
    "fusion_baseline_prediction",
    "fusion_baseline_pred",
    "fusion_relation",
    "corrected_fusion_score",
    "clinical_fusion_gap",
    "ecg_fusion_gap",
}

DEPRECATED_CLINICAL_BASELINE_SCORE_RUNTIME_FIELDS = {
    "locked_clinical_score",
    "clinical_score",
    "locked_clinical_threshold",
    "clinical_threshold",
    "clinical_margin",
    "clinical_probability_margin",
    "clinical_logit_margin",
    "clinical_score_consistency",
}

LABEL_LEAKAGE_FIELDS = {
    "Target_CAG_Binary",
    "CAG_label",
    "group_label",
    "ECG_Ischemic_Phenotype",
    "ECG_ischemia_label",
    "split",
    "y_true",
    "ground_truth",
    "MODEL_TP",
    "MODEL_FP",
    "MODEL_TN",
    "MODEL_FN",
    "TP",
    "FP",
    "TN",
    "FN",
}


def _walk_dicts(value: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(value, dict):
        out.append(value)
        for child in value.values():
            out.extend(_walk_dicts(child))
    elif isinstance(value, list):
        for child in value:
            out.extend(_walk_dicts(child))
    return out


def find_forbidden_runtime_fields(payload: dict[str, Any]) -> dict[str, list[str]]:
    """Return deprecated score fields and label leakage found in a payload."""

    found_fusion: set[str] = set()
    found_clinical: set[str] = set()
    found_labels: set[str] = set()
    for item in _walk_dicts(payload):
        keys = {str(k) for k in item}
        found_fusion.update(keys & DEPRECATED_COMBINED_SCORE_RUNTIME_FIELDS)
        found_clinical.update(keys & DEPRECATED_CLINICAL_BASELINE_SCORE_RUNTIME_FIELDS)
        found_labels.update(keys & LABEL_LEAKAGE_FIELDS)
    return {
        "deprecated_combined_score": sorted(found_fusion),
        "deprecated_clinical_baseline_score": sorted(found_clinical),
        "label_leakage": sorted(found_labels),
    }


def assert_ecg_anchor_runtime_boundary(payload: dict[str, Any]) -> None:
    found = find_forbidden_runtime_fields(payload)
    if found["deprecated_combined_score"]:
        raise RuntimeError("DEPRECATED_COMBINED_SCORE_FIELD_REJECTED: " + ",".join(found["deprecated_combined_score"]))
    if found["deprecated_clinical_baseline_score"]:
        raise RuntimeError("DEPRECATED_CLINICAL_BASELINE_SCORE_FIELD_REJECTED: " + ",".join(found["deprecated_clinical_baseline_score"]))
    if found["label_leakage"]:
        raise RuntimeError("RUNTIME_LABEL_FIELD_REJECTED: " + ",".join(found["label_leakage"]))


def build_ecg_anchor_context(payload: dict[str, Any]) -> dict[str, Any]:
    """Build or validate the canonical ECG-anchor context.

    The preferred input is ``payload["ecg_anchor_context"]``.  During static
    contract tests, a caller may provide ``locked_ecg_score`` and
    ``locked_ecg_threshold`` inside that context. No non-ECG score fallback is
    allowed here.
    """

    context = dict(payload.get("ecg_anchor_context") or {})
    if "locked_ecg_score" not in context or "locked_ecg_threshold" not in context:
        raise RuntimeError("ECG_ANCHOR_CONTEXT_REQUIRED")
    score = float(context["locked_ecg_score"])
    threshold = float(context["locked_ecg_threshold"])
    context["locked_ecg_score"] = score
    context["locked_ecg_threshold"] = threshold
    context["ecg_score_margin"] = float(score - threshold)
    context["ecg_baseline_state"] = (
        ECG_BASELINE_POSITIVE if score >= threshold else ECG_BASELINE_NEGATIVE
    )
    context["ecg_only_cag_prediction_class"] = (
        ECG_ONLY_CAG_PREDICTED_POSITIVE
        if context["ecg_baseline_state"] == ECG_BASELINE_POSITIVE
        else ECG_ONLY_CAG_PREDICTED_NEGATIVE
    )
    context.setdefault("ecg_model_version", "UNSPECIFIED_ECG_MODEL_VERSION")
    context.setdefault("ecg_score_provenance", "UNSPECIFIED_ECG_SCORE_PROVENANCE")
    return context
