"""v2.8 graded directional correction contract.

This module is deterministic and side-effect free.  It separates absolute
clinical CAG-risk context from the correction relation relative to the locked
ECG baseline.  It does not call LLMs, RAG, train models, read labels, or choose
numeric deltas.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


ABSOLUTE_CLINICAL_RISK_DIRECTIONS = {
    "HIGHER_CAG_RISK_CONTEXT",
    "LOWER_CAG_RISK_CONTEXT",
    "NEUTRAL_CAG_RISK_CONTEXT",
    "INDETERMINATE_CAG_RISK_CONTEXT",
}

CORRECTION_RELATIONS_RELATIVE_TO_ECG = {
    "ECG_BASELINE_SUPPORTED",
    "ECG_BASELINE_POSSIBLY_UNDERSTATED",
    "ECG_BASELINE_POSSIBLY_OVERSTATED",
    "ECG_BASELINE_RELATION_UNRESOLVED",
}

CORRECTION_EVIDENCE_GRADES = {"NONE", "WEAK", "MODERATE", "STRONG"}
ACTION_CONFIDENCE_BY_GRADE = {
    "NONE": "LOW",
    "WEAK": "LOW",
    "MODERATE": "MEDIUM",
    "STRONG": "HIGH",
}

UNDER_BRANCH = "UNDER_ESTIMATION_AUDIT"
OVER_BRANCH = "OVER_ESTIMATION_AUDIT"
UNKNOWN_BRANCH = "UNKNOWN_BRANCH"


@dataclass(frozen=True)
class DirectionalActionNormalization:
    agent4_raw_action: str
    controller_public_action: str
    action_direction_contract_valid: bool
    action_normalization_applied: bool
    action_normalization_reason: str
    guardrail_result: str
    applied_delta: float
    corrected_score_source: str = "locked_ecg_score"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def branch_from_ecg_baseline_state(ecg_baseline_state: str | None) -> str:
    state = str(ecg_baseline_state or "").upper()
    if state == "ECG_MODEL_NEGATIVE":
        return UNDER_BRANCH
    if state == "ECG_MODEL_POSITIVE":
        return OVER_BRANCH
    return UNKNOWN_BRANCH


def canonicalize_absolute_clinical_risk_direction(value: Any) -> str:
    direction = str(value or "").strip().upper()
    return direction if direction in ABSOLUTE_CLINICAL_RISK_DIRECTIONS else "INDETERMINATE_CAG_RISK_CONTEXT"


def canonicalize_correction_relation(value: Any) -> str:
    relation = str(value or "").strip().upper()
    return relation if relation in CORRECTION_RELATIONS_RELATIVE_TO_ECG else "ECG_BASELINE_RELATION_UNRESOLVED"


def canonicalize_evidence_grade(value: Any) -> str:
    grade = str(value or "").strip().upper()
    return grade if grade in CORRECTION_EVIDENCE_GRADES else "NONE"


def classify_negative_evidence_status(*, observed_normal: bool, missing: bool = False, imputed: bool = False) -> str:
    """Classify negative clinical evidence without treating missing/imputed as normal."""

    if missing:
        return "MISSING_OR_UNKNOWN"
    if imputed:
        return "ABSENCE_OF_POSITIVE_EVIDENCE"
    if observed_normal:
        return "VERIFIED_NEGATIVE_EVIDENCE"
    return "ABSENCE_OF_POSITIVE_EVIDENCE"


def derive_correction_relation_relative_to_ecg(
    *,
    ecg_baseline_state: str,
    absolute_clinical_risk_direction: str,
    verified_negative_evidence_present: bool = False,
    positive_high_risk_evidence_present: bool = False,
    material_direction_conflict: bool = False,
) -> str:
    """Return the only relation that may drive a correction action."""

    branch = branch_from_ecg_baseline_state(ecg_baseline_state)
    absolute = canonicalize_absolute_clinical_risk_direction(absolute_clinical_risk_direction)
    if material_direction_conflict:
        return "ECG_BASELINE_RELATION_UNRESOLVED"
    if branch == UNDER_BRANCH:
        if positive_high_risk_evidence_present or absolute == "HIGHER_CAG_RISK_CONTEXT":
            return "ECG_BASELINE_POSSIBLY_UNDERSTATED"
        return "ECG_BASELINE_SUPPORTED"
    if branch == OVER_BRANCH:
        if absolute == "HIGHER_CAG_RISK_CONTEXT" and not verified_negative_evidence_present:
            return "ECG_BASELINE_SUPPORTED"
        if verified_negative_evidence_present or absolute == "LOWER_CAG_RISK_CONTEXT":
            return "ECG_BASELINE_POSSIBLY_OVERSTATED"
        return "ECG_BASELINE_RELATION_UNRESOLVED"
    return "ECG_BASELINE_RELATION_UNRESOLVED"


def grade_directional_evidence(
    *,
    directional_signal_present: bool,
    independent_supporting_group_count: int = 0,
    agent2_risk_band: str = "LOW",
    agent1_agent2_same_direction: bool = False,
    critical_safety_conflict: bool = False,
) -> str:
    """Grade signal strength without requiring RAG success and without labels."""

    if not directional_signal_present or critical_safety_conflict:
        return "NONE"
    band = str(agent2_risk_band or "").upper()
    groups = max(0, int(independent_supporting_group_count or 0))
    if groups >= 3 and agent1_agent2_same_direction:
        return "STRONG"
    if groups >= 2 or (band in {"INTERMEDIATE", "HIGH"} and groups >= 1):
        return "MODERATE"
    return "WEAK"


def allowed_actions_for_relation(*, ecg_baseline_state: str, relation: str) -> set[str]:
    branch = branch_from_ecg_baseline_state(ecg_baseline_state)
    relation = canonicalize_correction_relation(relation)
    if relation == "ECG_BASELINE_SUPPORTED":
        return {"PRESERVE"}
    if relation == "ECG_BASELINE_POSSIBLY_UNDERSTATED" and branch == UNDER_BRANCH:
        return {"UP", "PRESERVE", "REVIEW_REQUIRED"}
    if relation == "ECG_BASELINE_POSSIBLY_OVERSTATED" and branch == OVER_BRANCH:
        return {"DOWN", "PRESERVE", "REVIEW_REQUIRED"}
    if relation == "ECG_BASELINE_RELATION_UNRESOLVED":
        return {"PRESERVE", "REVIEW_REQUIRED"}
    return {"PRESERVE", "REVIEW_REQUIRED"}


def normalize_agent4_action_for_directional_contract(
    *,
    ecg_baseline_state: str,
    raw_action: str,
    correction_relation_relative_to_ecg: str,
    correction_evidence_grade: str = "NONE",
) -> DirectionalActionNormalization:
    """Normalize impossible branch/action combinations while keeping zero delta."""

    branch = branch_from_ecg_baseline_state(ecg_baseline_state)
    raw = str(raw_action or "REVIEW_REQUIRED").strip().upper()
    relation = canonicalize_correction_relation(correction_relation_relative_to_ecg)
    grade = canonicalize_evidence_grade(correction_evidence_grade)
    allowed = allowed_actions_for_relation(ecg_baseline_state=ecg_baseline_state, relation=relation)

    if branch == OVER_BRANCH and raw == "UP":
        return DirectionalActionNormalization(raw, "PRESERVE", False, True, "OVER_BRANCH_FORBIDS_UP", "BLOCKED_DIRECTION_MISMATCH", 0.0)
    if branch == UNDER_BRANCH and raw == "DOWN":
        return DirectionalActionNormalization(raw, "PRESERVE", False, True, "UNDER_BRANCH_FORBIDS_DOWN", "BLOCKED_DIRECTION_MISMATCH", 0.0)
    if relation == "ECG_BASELINE_SUPPORTED" and raw in {"UP", "DOWN"}:
        return DirectionalActionNormalization(raw, "PRESERVE", False, True, "BASELINE_SUPPORTED_FORBIDS_DIRECTIONAL_ACTION", "BLOCKED_BASELINE_SUPPORTED_DIRECTION", 0.0)
    if relation == "ECG_BASELINE_RELATION_UNRESOLVED" and raw in {"UP", "DOWN"}:
        return DirectionalActionNormalization(raw, "REVIEW_REQUIRED", False, True, "UNRESOLVED_FORBIDS_DIRECTIONAL_ACTION", "BLOCKED_UNRESOLVED_DIRECTION", 0.0)
    if raw not in allowed:
        safe = "REVIEW_REQUIRED" if relation == "ECG_BASELINE_RELATION_UNRESOLVED" else "PRESERVE"
        return DirectionalActionNormalization(raw, safe, False, True, "ACTION_NOT_ALLOWED_FOR_RELATION", "BLOCKED_RELATION_ACTION_MISMATCH", 0.0)
    if raw in {"UP", "DOWN"} and grade == "NONE":
        return DirectionalActionNormalization(raw, "REVIEW_REQUIRED", False, True, "DIRECTIONAL_ACTION_REQUIRES_NON_NONE_GRADE", "BLOCKED_NO_DIRECTIONAL_SIGNAL", 0.0)
    return DirectionalActionNormalization(raw, raw, True, False, "NONE", "PASS_DIRECTIONAL_CONTRACT", 0.0)


def detect_overconservative_policy(
    *,
    eligible_directional_candidate_count: int,
    emitted_directional_action_count: int,
    critical_safety_conflict_count: int = 0,
) -> bool:
    """Flag systematic suppression only when eligible signals exist."""

    return (
        int(eligible_directional_candidate_count or 0) > 0
        and int(emitted_directional_action_count or 0) == 0
        and int(critical_safety_conflict_count or 0) == 0
    )
