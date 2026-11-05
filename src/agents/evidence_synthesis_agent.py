from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from .ecg_audit_agent import LLMAgentBase, stable_hash
from .clinical_assessment_agent import build_safe_clinical_packet
from .missingness import classify_dataset_v4_missingness
from .schemas import (
    V2_6_SAFE_BOUNDARY_CHECK,
    V2_6_SCHEMA_VERSION,
    find_forbidden_v2_6_fields,
    validate_v2_6_agent_output,
)


def _locked_score(payload: dict[str, Any], name: str) -> float:
    if name != "ecg":
        raise ValueError(f"locked_{name}_score_not_available_in_ecg_anchor_runtime")
    ecg_anchor = payload.get("ecg_anchor_context", {})
    value = ecg_anchor.get("locked_ecg_score")
    if value is None:
        raise ValueError(f"missing_locked_{name}_score_fail_closed")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"invalid_locked_{name}_score_fail_closed")
    return score


def _locked_threshold(payload: dict[str, Any], name: str) -> float:
    if name != "ecg":
        raise ValueError(f"locked_{name}_threshold_not_available_in_ecg_anchor_runtime")
    ecg_anchor = payload.get("ecg_anchor_context", {})
    value = ecg_anchor.get("locked_ecg_threshold")
    if value is None:
        raise ValueError(f"missing_locked_{name}_threshold_fail_closed")
    threshold = float(value)
    if not math.isfinite(threshold):
        raise ValueError(f"invalid_locked_{name}_threshold_fail_closed")
    return threshold


def _locked_risk(payload: dict[str, Any], name: str) -> str:
    score = _locked_score(payload, name)
    threshold = _locked_threshold(payload, name)
    return "above_threshold" if score >= threshold else "below_threshold"


def _score_margin(score: float, threshold: float | None = None) -> float | None:
    if threshold is None:
        return None
    return round(float(score) - float(threshold), 6)


def _uncertainty_bin(margin: float | None) -> str:
    return "threshold_unavailable" if margin is None else "continuous_distance_recorded"


EXPLICIT_ECG_MIMIC_ALLOWLIST = {
    "left_ventricular_hypertrophy_mimic",
    "lvh_strain_secondary_st_t",
}


PATIENT_PRESENT_STATUSES = {
    "present",
    "categorical_value_available",
    "continuous_value_available",
}


def patient_concept_statuses_v2_6(plan: dict[str, Any]) -> dict[str, str]:
    """Return patient concept states from the deterministic evidence trace.

    Clinical factor IDs may legitimately appear in Agent2 evidence when the
    patient state is absent (for example, absent diabetes can support the
    overestimation head).  Such IDs must remain available for arbitration but
    must not be reinterpreted as patient-present facts by RAG.
    """

    statuses: dict[str, str] = {}
    for item in plan.get("patient_evidence_trace", []):
        concept = str(
            item.get("canonical_concept_id")
            or item.get("patient_factor_id")
            or ""
        ).strip()
        if not concept:
            continue
        status = str(
            item.get("concept_status")
            or item.get("patient_factor_status")
            or item.get("status")
            or "unknown"
        ).strip().lower()
        statuses[concept] = status
    return statuses


def patient_present_concepts_v2_6(plan: dict[str, Any]) -> set[str]:
    """Return only concepts whose patient state is positively available."""

    present = set(map(str, plan.get("patient_ecg_concept_ids", [])))
    statuses = patient_concept_statuses_v2_6(plan)
    present.update(
        concept for concept, status in statuses.items()
        if status in PATIENT_PRESENT_STATUSES
    )
    # The explicit field supports compact plans that intentionally omit the
    # verbose evidence trace.  Do not fall back to patient_clinical_factor_ids:
    # that collection contains both positive and negative patient evidence.
    present.update(map(str, plan.get("patient_present_clinical_factor_ids", [])))
    return present


def _agent1_direction(agent1: dict[str, Any] | None) -> str:
    if not agent1:
        return "indeterminate"
    legacy = agent1.get("ecg_evidence_direction")
    if legacy:
        return str(legacy)
    pattern = str(agent1.get("ecg_evidence_pattern", "NOT_EVALUABLE"))
    if pattern == "ISCHEMIA_SUPPORTIVE":
        return "up"
    if pattern == "NO_CLEAR_ISCHEMIC_EVIDENCE":
        return "neutral"
    if pattern == "NON_SPECIFIC_OR_CONFOUNDED":
        mimic = (
            set(map(str, agent1.get("possible_ecg_mimic_terms", [])))
            | set(map(str, agent1.get("non_specific_or_confounding_evidence_ids", [])))
        ) & EXPLICIT_ECG_MIMIC_ALLOWLIST
        if bool(agent1.get("false_positive_mimic_warning")) and mimic:
            return "down"
        return "neutral"
    if pattern == "MIXED":
        return "neutral"
    return "indeterminate"


def _agent2_direction(agent2: dict[str, Any] | None) -> str:
    if not agent2:
        return "indeterminate"
    stance = str(agent2.get("evidence_stance", "INSUFFICIENT")).upper()
    head = str(agent2.get("selected_discrepancy_head", "")).upper()
    if stance != "SUPPORTS_DISCREPANCY":
        return "neutral" if stance in {"OPPOSES_DISCREPANCY", "MIXED", "INSUFFICIENT"} else "indeterminate"
    return "up" if head == "UNDERESTIMATION" else "down" if head == "OVERESTIMATION" else "indeterminate"


def _directional_strength_rank(value: str) -> int:
    return {"NONE": 0, "WEAK": 1, "MODERATE": 2, "STRONG": 3}.get(str(value).upper(), 0)


def _strength_from_rank(rank: int) -> str:
    if rank >= 3:
        return "STRONG"
    if rank == 2:
        return "MODERATE"
    if rank == 1:
        return "WEAK"
    return "NONE"


def assess_patient_evidence_for_rag_aware_policy_v2_7(
    agent1: dict[str, Any] | None,
    agent2: dict[str, Any] | None,
    ecg_anchor_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize patient-level evidence independently of retrieved RAG cards.

    RAG may contextualize or safety-check the conclusion, but absence of a
    supporting card must not erase patient-level ECG/clinical/model evidence.
    """

    a1_dir = _agent1_direction(agent1).lower()
    a2_dir = _agent2_direction(agent2).lower()
    a1_strength = _directional_strength_rank(str((agent1 or {}).get("ecg_directional_strength", "NONE")))
    a2_strength = _directional_strength_rank(
        "MODERATE" if str((agent2 or {}).get("evidence_stance", "")).upper() == "SUPPORTS_DISCREPANCY" else "NONE"
    )
    strength_rank = max(a1_strength, a2_strength)
    if a1_dir == "up" and a2_dir == "up":
        direction = "UP"
        basis = "CONCORDANT_MULTIMODAL"
        strength_rank = max(strength_rank, 2)
    elif a1_dir == "down" and a2_dir == "down":
        direction = "DOWN"
        basis = "CONCORDANT_MULTIMODAL"
        strength_rank = max(strength_rank, 1)
    elif a1_dir == "up" and a2_dir in {"neutral", "indeterminate"}:
        direction = "UP"
        basis = "ECG_DOMINANT"
    elif a2_dir == "up" and a1_dir in {"neutral", "indeterminate"}:
        direction = "UP"
        basis = "CLINICAL_DOMINANT"
    elif a1_dir == "down" and a2_dir in {"neutral", "indeterminate"}:
        direction = "DOWN"
        basis = "ECG_DOMINANT"
    elif a2_dir == "down" and a1_dir in {"neutral", "indeterminate"}:
        direction = "DOWN"
        basis = "CLINICAL_DOMINANT"
    elif a1_dir in {"up", "down"} and a2_dir in {"up", "down"} and a1_dir != a2_dir:
        direction = "UNCERTAIN"
        basis = "CONFLICTING_MULTIMODAL"
    else:
        direction = "PRESERVE" if (agent1 or agent2) else "UNCERTAIN"
        basis = "INSUFFICIENT"
        strength_rank = 0
    if basis == "INSUFFICIENT":
        strength_rank = 0
    ecg_baseline_state = str((ecg_anchor_context or {}).get("ecg_baseline_state", "")).upper()
    absolute_direction = {
        "UP": "HIGHER_CAG_RISK_CONTEXT",
        "DOWN": "LOWER_CAG_RISK_CONTEXT",
        "PRESERVE": "NEUTRAL_CAG_RISK_CONTEXT",
    }.get(direction, "INDETERMINATE_CAG_RISK_CONTEXT")
    if ecg_baseline_state == "ECG_MODEL_NEGATIVE" and direction == "UP":
        relation = "ECG_BASELINE_POSSIBLY_UNDERSTATED"
    elif ecg_baseline_state == "ECG_MODEL_POSITIVE" and direction == "DOWN":
        relation = "ECG_BASELINE_POSSIBLY_OVERSTATED"
    elif ecg_baseline_state in {"ECG_MODEL_NEGATIVE", "ECG_MODEL_POSITIVE"} and direction == "PRESERVE":
        relation = "ECG_BASELINE_SUPPORTED"
    elif ecg_baseline_state == "ECG_MODEL_POSITIVE" and direction == "UP":
        relation = "ECG_BASELINE_SUPPORTED"
    elif ecg_baseline_state == "ECG_MODEL_NEGATIVE" and direction == "DOWN":
        relation = "ECG_BASELINE_SUPPORTED"
    else:
        relation = "ECG_BASELINE_RELATION_UNRESOLVED"
    grade = _strength_from_rank(strength_rank) if direction in {"UP", "DOWN"} else "NONE"
    confidence = {"NONE": "LOW", "WEAK": "LOW", "MODERATE": "MEDIUM", "STRONG": "HIGH"}[grade]
    return {
        "patient_evidence_direction": direction,
        "patient_evidence_strength": _strength_from_rank(strength_rank),
        "patient_evidence_basis": basis,
        "absolute_clinical_risk_direction": absolute_direction,
        "correction_relation_relative_to_ecg": relation,
        "correction_evidence_grade": grade,
        "correction_confidence_relative_to_ecg": confidence,
        "directional_signal_present": bool(direction in {"UP", "DOWN"} and grade != "NONE"),
        "correction_relation_basis": basis,
    }


def classify_rag_card_utilization_for_agent4_v2_7(
    agent3_output: dict[str, Any] | None,
    retrieved_cards: list[dict[str, Any]] | None = None,
    *,
    agent1: dict[str, Any] | None = None,
    agent2: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify RAG card role without making RAG mandatory for correction.

    Retrieved cards may be context-only or review-only. Only supporting,
    applicable, actionable, veto, or explicit contradictory cards should affect
    Agent4's correction policy.
    """

    output = agent3_output or {}
    retrieved_ids = set(map(str, output.get("retrieved_card_ids", []) or []))
    supporting_ids = set(map(str, output.get("supporting_card_ids", []) or []))
    applicable_ids = set(map(str, output.get("applicable_card_ids", []) or []))
    actionable_ids = set(map(str, output.get("actionable_card_ids", []) or []))
    requested_veto_ids = set(map(str, output.get("veto_card_ids", []) or []))
    contradictory_ids = set(map(str, output.get("contradictory_card_ids", []) or []))
    if not retrieved_ids and retrieved_cards:
        retrieved_ids = {str(card.get("card_id", "")) for card in retrieved_cards if str(card.get("card_id", ""))}
    def patient_grounded_veto(card_id: str) -> bool:
        # DANGER_ECG_005 protects persistent ischemia and dangerous ECG
        # patterns. Retrieval, a generic symptom-attack field, or the LLM's
        # veto designation alone is not patient-level evidence of that state.
        v2_grounded_safety_cards = {
            "DANGER_ECG_HYPERACUTE_T_001",
            "DANGER_ECG_JWAVE_001",
            "DANGER_ECG_LBBB_SGARBOSSA_001",
            "DANGER_ISCHEMIA_PERSISTENT_001",
        }
        if card_id != "DANGER_ECG_005" and card_id not in v2_grounded_safety_cards:
            return True
        a1 = agent1 or {}
        retrieval_concepts = {str(item).lower() for item in a1.get("retrieval_concepts", []) or []}
        confounder_ids = {
            str(item).lower() for item in a1.get("non_specific_or_confounding_evidence_ids", []) or []
        }
        ischemic_ids = {str(item).lower() for item in a1.get("ischemic_evidence_ids", []) or []}
        all_ecg_ids = retrieval_concepts | confounder_ids | ischemic_ids
        if card_id == "DANGER_ECG_JWAVE_001":
            return bool(all_ecg_ids & {"high_risk_j_wave_pattern", "early_repolarization_syndrome"})
        if card_id == "DANGER_ECG_LBBB_SGARBOSSA_001":
            sgarbossa_ids = {
                "modified_sgarbossa_pattern", "sgarbossa_pattern", "lbbb_sgarbossa_pattern",
            }
            return bool(all_ecg_ids & sgarbossa_ids) or (
                bool(a1.get("hidden_ischemia_warning"))
                and bool(all_ecg_ids & {"left_bundle_branch_block", "paced_rhythm"})
            )
        if card_id == "DANGER_ECG_HYPERACUTE_T_001":
            return bool(all_ecg_ids & {"hyperacute_t_wave"}) or bool(a1.get("hidden_ischemia_warning"))
        if bool(a1.get("hidden_ischemia_warning")):
            return True
        if str(a1.get("ecg_evidence_pattern", "")).upper() == "ISCHEMIA_SUPPORTIVE":
            return True
        danger_ids = {
            "posterior_occlusion_pattern",
            "dynamic_st_t_change",
            "global_ischemia_pattern",
            "high_risk_j_wave_pattern",
            "early_repolarization_syndrome",
        }
        return bool(ischemic_ids & danger_ids) or bool(ischemic_ids)

    grounded_veto_ids = {card_id for card_id in requested_veto_ids if patient_grounded_veto(card_id)}
    downgraded_veto_ids = requested_veto_ids - grounded_veto_ids
    grounded_contradictory_ids = {
        card_id for card_id in contradictory_ids if patient_grounded_veto(card_id)
    }
    downgraded_contradictory_ids = contradictory_ids - grounded_contradictory_ids
    # A downgraded safety veto remains external context; it must not turn into
    # correction support merely because the LLM also listed it as supporting.
    correction_ids = sorted((supporting_ids | actionable_ids) - requested_veto_ids)
    safety_ids = sorted(grounded_veto_ids)
    review_only = sorted(
        (
            retrieved_ids
            - supporting_ids
            - applicable_ids
            - actionable_ids
            - requested_veto_ids
            - contradictory_ids
        )
        | downgraded_veto_ids
        | downgraded_contradictory_ids
    )
    mechanism_context = sorted(
        (applicable_ids | retrieved_ids)
        - set(correction_ids)
        - set(safety_ids)
        - grounded_contradictory_ids
    )
    not_supporting = [
        f"{card_id}:retrieved_context_not_patient_level_support"
        for card_id in review_only
    ]
    not_supporting.extend(
        f"{card_id}:veto_downgraded_missing_patient_level_danger_trigger"
        for card_id in sorted(downgraded_veto_ids)
    )
    not_supporting.extend(
        f"{card_id}:contradiction_downgraded_missing_patient_level_danger_trigger"
        for card_id in sorted(downgraded_contradictory_ids)
    )
    if grounded_contradictory_ids:
        status = "RAG_CONTRADICTORY"
        strength = "MODERATE_SUPPORT"
    elif safety_ids:
        status = "RAG_SAFETY_WARNING"
        strength = "MODERATE_SUPPORT"
    elif correction_ids:
        status = "RAG_CORRECTION_SUPPORT"
        strength = "MODERATE_SUPPORT" if len(correction_ids) >= 2 else "WEAK_SUPPORT"
    elif applicable_ids:
        status = "RAG_MECHANISM_SUPPORT"
        strength = "WEAK_SUPPORT"
    elif retrieved_ids:
        status = "RAG_CONTEXT_ONLY"
        strength = "CONTEXT_ONLY"
    else:
        status = "RAG_UNAVAILABLE"
        strength = "NONE"
    should_affect = status in {"RAG_CORRECTION_SUPPORT", "RAG_SAFETY_WARNING", "RAG_CONTRADICTORY"}
    return {
        "rag_evidence_status": status,
        "rag_evidence_strength": strength,
        "rag_should_affect_agent4_policy": should_affect,
        "veto_card_ids": safety_ids,
        "mechanism_context_card_ids": mechanism_context,
        "safety_warning_card_ids": safety_ids,
        "correction_supporting_card_ids": correction_ids,
        "review_only_card_ids": review_only,
        "contradictory_card_ids": sorted(grounded_contradictory_ids),
        "card_support_direction": [
            f"{card_id}:correction_support" for card_id in correction_ids
        ] + [
            f"{card_id}:safety_warning" for card_id in safety_ids
        ] + [
            f"{card_id}:context_only" for card_id in review_only
        ],
        "why_card_is_not_supporting": not_supporting,
    }


def enrich_agent3_output_with_rag_aware_fields_v2_7(
    output: dict[str, Any],
    *,
    agent1: dict[str, Any] | None,
    agent2: dict[str, Any] | None,
    ecg_anchor_context: dict[str, Any] | None = None,
    retrieved_cards: list[dict[str, Any]] | None = None,
    canonical_relation_v3_core_enabled: bool = False,
    canonical_relation_v3_1_enabled: bool = False,
) -> dict[str, Any]:
    enriched = dict(output)
    provisional_relation = {
        "llm_provisional_relation": enriched.get("correction_relation_relative_to_ecg"),
        "pre_adjudication_ecg_relation": enriched.get("ecg_baseline_relation"),
    }
    enriched.update(assess_patient_evidence_for_rag_aware_policy_v2_7(agent1, agent2, ecg_anchor_context))
    enriched.update(classify_rag_card_utilization_for_agent4_v2_7(
        enriched,
        retrieved_cards,
        agent1=agent1,
        agent2=agent2,
    ))
    if canonical_relation_v3_1_enabled:
        canonical = derive_canonical_correction_relation_v3_1(
            agent1=agent1 or {},
            agent2=agent2 or {},
            ecg_anchor_context=ecg_anchor_context or {},
            rag_status=enriched,
            provisional_relation=provisional_relation,
        )
    elif canonical_relation_v3_core_enabled:
        canonical = derive_canonical_correction_relation_v3_core(
            agent1=agent1 or {},
            agent2=agent2 or {},
            ecg_anchor_context=ecg_anchor_context or {},
            rag_status=enriched,
            provisional_relation=provisional_relation,
        )
    else:
        canonical = None
    if canonical:
        protected_fields = {
            "correction_relation_relative_to_ecg",
            "ecg_baseline_relation",
            "correction_evidence_grade",
            "correction_confidence_relative_to_ecg",
            "directional_signal_present",
            "canonical_relation_source",
        }
        overwrite_attempt_count = sum(
            1 for field in protected_fields
            if field in enriched and field in canonical and enriched.get(field) != canonical.get(field)
        )
        enriched.update(canonical)
        relation = str(enriched.get("correction_relation_relative_to_ecg", ""))
        if relation in {
            "ECG_BASELINE_POSSIBLY_UNDERSTATED",
            "ECG_BASELINE_POSSIBLY_OVERSTATED",
        }:
            # Agent2 committed IDs already passed patient-visibility checks.
            # A raw Agent3 omission must not erase validated upstream evidence
            # after deterministic evidence adjudication accepts that direction.
            committed_support = list(dict.fromkeys([
                *map(str, enriched.get("supporting_evidence_ids", []) or []),
                *map(str, (agent2 or {}).get("supporting_factor_ids", []) or []),
                *(
                    list(map(str, (agent1 or {}).get("ischemic_evidence_ids", []) or []))
                    if relation == "ECG_BASELINE_POSSIBLY_UNDERSTATED"
                    else []
                ),
            ]))
            committed_contradiction = list(dict.fromkeys([
                *map(str, enriched.get("contradicting_evidence_ids", []) or []),
                *map(str, (agent2 or {}).get("contradicting_factor_ids", []) or []),
            ]))
            enriched["supporting_evidence_ids"] = committed_support
            enriched["contradicting_evidence_ids"] = [
                item for item in committed_contradiction if item not in set(committed_support)
            ]
        enriched["overwrite_attempt_count"] = overwrite_attempt_count
        enriched["relation_equality_check_pass"] = (
            enriched.get("ecg_baseline_relation")
            == enriched.get("correction_relation_relative_to_ecg")
        )
    return enriched


def map_preliminary_to_global_consistency_v2_6(pattern: str) -> str:
    return {
        "ecg_high_clinical_low": "ECG_HIGH_CLINICAL_LOW",
        "ecg_low_clinical_high": "ECG_LOW_CLINICAL_HIGH",
        "modal_evidence_direction_discordance": "MODAL_EVIDENCE_SCORE_CONFLICT",
        "concordant_high": "CONSISTENT_CONFIDENT",
        "concordant_low": "CONSISTENT_CONFIDENT",
        "unclear": "COMPLEX_DISCORDANCE",
    }.get(str(pattern), "COMPLEX_DISCORDANCE")


def build_disagreement_triage_v2_6(
    payload: dict[str, Any],
    agent1: dict[str, Any] | None = None,
    agent2: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """ECG-anchor disagreement triage used before optional Agent3 RAG reasoning."""

    ecg = _locked_score(payload, "ecg")
    ecg_risk = _locked_risk(payload, "ecg")
    ecg_threshold = _locked_threshold(payload, "ecg")
    ecg_margin = _score_margin(ecg, ecg_threshold)
    a1_dir = _agent1_direction(agent1)
    a2_dir = _agent2_direction(agent2)
    reasons: list[str] = []
    if a1_dir in {"up", "down"} and a2_dir in {"up", "down"} and a1_dir != a2_dir:
        reasons.append("agent1_agent2_patient_evidence_direction_disagreement")
        pattern = "modal_evidence_direction_discordance"
    elif ecg_risk == "above_threshold" and a2_dir == "down":
        reasons.append("clinical_evidence_challenges_positive_ecg_baseline")
        pattern = "ecg_high_clinical_low"
    elif ecg_risk == "below_threshold" and a2_dir == "up":
        reasons.append("clinical_evidence_challenges_negative_ecg_baseline")
        pattern = "ecg_low_clinical_high"
    elif ecg_risk == "above_threshold":
        pattern = "concordant_high"
    elif ecg_risk == "below_threshold":
        pattern = "concordant_low"
    else:
        pattern = "unclear"
    return {
        "triage_triggered": bool(reasons),
        "triage_trigger_reasons": reasons,
        "discordance_pattern_preliminary": pattern,
        "ecg_anchor_uncertainty_bin": _uncertainty_bin(ecg_margin),
        "ecg_threshold_side": ecg_risk,
        "ecg_distance_to_threshold": ecg_margin,
        "global_consistency_status": map_preliminary_to_global_consistency_v2_6(pattern),
    }


def _card_projection(card: dict[str, Any]) -> dict[str, Any]:
    return {
        key: card.get(key)
        for key in (
            "card_id", "card_group", "card_title", "key_evidence_summary",
            "blocked_action", "usage_boundary",
            "source_excerpt_ids", "confidence_level", "safety_level",
        )
        if key in card
    }


def build_agent1_packet_for_agent3(agent1: dict[str, Any]) -> dict[str, Any]:
    """Return the compact in-memory ECG evidence view for Agent3."""

    return {
            "ecg_evidence_pattern": agent1.get("ecg_evidence_pattern"),
            "ecg_directional_strength": agent1.get("ecg_directional_strength"),
            "ecg_evidence_informativeness": agent1.get("ecg_evidence_informativeness"),
            "ecg_signal_quality": agent1.get("ecg_signal_quality"),
            "ecg_uncertainty": agent1.get("ecg_uncertainty"),
            "hidden_ischemia_warning": bool(agent1.get("hidden_ischemia_warning", False)),
            "false_positive_mimic_warning": bool(agent1.get("false_positive_mimic_warning", False)),
            "ischemic_evidence_ids": list(agent1.get("ischemic_evidence_ids", []) or []),
            "non_specific_or_confounding_evidence_ids": list(agent1.get("non_specific_or_confounding_evidence_ids", []) or []),
            "brief_ecg_reasoning_summary": str(agent1.get("blind_reasoning_summary", ""))[:600],
    }


def build_agent2_packet_for_agent3(agent2: dict[str, Any]) -> dict[str, Any]:
    """Return the sole approved Agent2 Safe View for Agent3."""
    missingness = classify_dataset_v4_missingness(agent2)
    return {
        "selected_discrepancy_head": agent2.get("selected_discrepancy_head"),
        "head_score": agent2.get("head_score"),
        "risk_band": agent2.get("risk_band"),
        "alert_state": agent2.get("alert_state"),
        "evidence_stance": agent2.get("evidence_stance"),
        "supporting_factor_ids": list(agent2.get("supporting_factor_ids", []) or []),
        "contradicting_factor_ids": list(agent2.get("contradicting_factor_ids", []) or []),
        "confounder_factor_ids": list(agent2.get("confounder_factor_ids", []) or []),
        **missingness,
        "evidence_sufficiency": agent2.get("evidence_sufficiency"),
        "concise_rationale": str(agent2.get("concise_rationale", ""))[:320],
    }


def build_anchor_packet_for_agent3(context: dict[str, Any]) -> dict[str, Any]:
    anchor = context.get("ecg_anchor_context", {}) or {}
    return {
        "ecg_baseline_state": anchor.get("ecg_baseline_state"),
    }


def build_disagreement_vector_v2_6(
    payload: dict[str, Any],
    agent1: dict[str, Any],
    agent2: dict[str, Any],
    triage: dict[str, Any],
) -> dict[str, Any]:
    return {
        "discordance_pattern": triage.get("discordance_pattern_preliminary", "unclear"),
        "ecg_evidence_direction": _agent1_direction(agent1),
        "clinical_evidence_direction": _agent2_direction(agent2),
        "ecg_concepts": list(agent1.get("retrieval_concepts", agent1.get("ecg_retrieval_concept_ids", []))),
        "clinical_factor_ids": list(agent2.get("supporting_factor_ids", [])),
        "triage_trigger_reasons": list(triage.get("triage_trigger_reasons", [])),
    }


def build_initial_rag_query_v2_6(
    context: dict[str, Any],
    agent1: dict[str, Any],
    agent2: dict[str, Any],
    triage: dict[str, Any],
) -> dict[str, Any]:
    ecg_terms = sorted(set(
        str(item.get("canonical_concept_id"))
        for item in agent1.get("ecg_evidence_items", [])
        if item.get("evidence_domain") == "ECG"
        and item.get("concept_status") == "present"
        and item.get("projection_method") == "deterministic_lexical_concept_projection"
        and str(item.get("canonical_concept_id")) not in QUERY_MEDICAL_CONCEPT_EXCLUSIONS
    ))
    if not ecg_terms and not agent1.get("ecg_evidence_items"):
        ecg_terms = list(agent1.get("retrieval_concepts", agent1.get("ecg_positive_evidence_terms", [])))
    present_factors = list(agent2.get("supporting_factor_ids", []))
    confounders = list(agent2.get("confounder_factor_ids", []))
    query_terms = list(dict.fromkeys(
        ecg_terms[:2]
        + present_factors[:2]
        + confounders[:1]
        + ["ECG clinical evidence review", "supporting and opposing evidence", "uncertainty"]
    ))
    return {
        "retrieval_round_index": int(context.get("retrieval_round_index", 0)),
        "retrieval_query": " ".join(str(term).replace("_", " ") for term in query_terms),
        "query_terms": query_terms,
        "ecg_concept_ids": ecg_terms,
        "patient_factor_ids": present_factors,
        "discordance_pattern": triage.get("discordance_pattern_preliminary", "unclear"),
        "neutral_query": True,
        "constructed_from_patient_evidence": True,
    }


def choose_cohort_prior_branch_v2_6(
    agent1: dict[str, Any],
    agent2: dict[str, Any],
    triage: dict[str, Any],
) -> str:
    pattern = str(triage.get("discordance_pattern_preliminary", "unclear"))
    ecg_direction = _agent1_direction(agent1)
    clinical_direction = _agent2_direction(agent2)
    ecg_mimic = bool(agent1.get("false_positive_mimic_warning")) or bool(agent2.get("confounder_factor_ids"))
    hidden = bool(agent1.get("hidden_ischemia_warning")) or (
        str(agent2.get("evidence_stance", "")).upper() == "SUPPORTS_DISCREPANCY"
        and str(agent2.get("selected_discrepancy_head", "")).upper() == "UNDERESTIMATION"
    )
    if pattern == "ecg_high_clinical_low" or ecg_mimic or (ecg_direction == "up" and clinical_direction in {"down", "neutral"}):
        return "ecg_false_positive_like"
    if pattern == "ecg_low_clinical_high" or hidden or (ecg_direction in {"down", "neutral"} and clinical_direction == "up"):
        return "hidden_high_risk_like"
    return "global_cag"


def derive_rag_trigger_v2_7(triage: dict[str, Any], rag_query_eligible: bool = True) -> str:
    """Deprecated compatibility helper.

    Active v2.8 runtime uses the neutral Stage-A gate fields
    retrieval_decision/retrieval_reason instead of legacy trigger strings.
    """
    if not rag_query_eligible:
        return "NONE"
    return "NEUTRAL_STAGE_A_GATE"


V2_8_QUERY_ROUTES = {
    "NONE",
    "NEUTRAL_REVIEW",
    "INSUFFICIENT_EVIDENCE_REVIEW",
}

V2_8_EVIDENCE_SUFFICIENCY = {"SUFFICIENT", "INSUFFICIENT"}
V2_8_MODALITY_ALIGNMENT = {"ALIGNED", "DISAGREEMENT", "UNCERTAIN"}
V2_8_OVERALL_CONFIDENCE = {"HIGH", "MODERATE", "LOW"}
V2_8_RETRIEVAL_DECISIONS = {"SKIP", "RETRIEVE"}
V2_8_RETRIEVAL_REASONS = {
    "NONE",
    "INSUFFICIENT_EVIDENCE",
    "LOW_CONFIDENCE",
    "MODALITY_DISAGREEMENT",
    "COMBINED_UNCERTAINTY",
}


def derive_query_route_v2_8(
    triage: dict[str, Any],
    *,
    selected_branch: str,
    rag_query_eligible: bool = True,
) -> str:
    """Dataset-v4 v2.8 Agent3 Stage-A route contract.

    This function is the active route source for v2.8. It intentionally does
    not translate legacy trigger values into directional routes.
    """

    if not rag_query_eligible:
        return "INSUFFICIENT_EVIDENCE_REVIEW"
    status = str(triage.get("global_consistency_status", "COMPLEX_DISCORDANCE"))
    pattern = str(triage.get("discordance_pattern_preliminary", "unclear"))
    reasons = list(triage.get("triage_trigger_reasons", []))
    if status == "CONSISTENT_CONFIDENT" and not reasons:
        return "NONE"
    if status == "INSUFFICIENT_EVIDENCE":
        return "INSUFFICIENT_EVIDENCE_REVIEW"
    if status == "CONSISTENT_UNCERTAIN":
        return "INSUFFICIENT_EVIDENCE_REVIEW"
    return "NEUTRAL_REVIEW"


def query_route_allows_rag_v2_8(query_route: str) -> bool:
    if query_route not in V2_8_QUERY_ROUTES:
        raise ValueError(f"invalid v2.8 query_route: {query_route!r}")
    return query_route == "NEUTRAL_REVIEW"


def derive_modality_alignment_v2_8(triage: dict[str, Any]) -> str:
    status = str(triage.get("global_consistency_status", "COMPLEX_DISCORDANCE"))
    pattern = str(triage.get("discordance_pattern_preliminary", "unclear"))
    if status == "CONSISTENT_CONFIDENT" or pattern in {"concordant_high", "concordant_low"}:
        return "ALIGNED"
    if status in {"CONSISTENT_UNCERTAIN", "INSUFFICIENT_EVIDENCE"} or pattern == "unclear":
        return "UNCERTAIN"
    return "DISAGREEMENT"


def derive_evidence_alignment_v2_8(agent1: dict[str, Any], agent2: dict[str, Any]) -> str:
    a1 = _agent1_direction(agent1).lower()
    a2 = _agent2_direction(agent2).lower()
    if a1 in {"up", "down"} and a2 in {"up", "down"}:
        return "ALIGNED" if a1 == a2 else "DISAGREEMENT"
    if (a1 in {"up", "down"} and a2 in {"neutral", "indeterminate"}) or (
        a2 in {"up", "down"} and a1 in {"neutral", "indeterminate"}
    ):
        return "ALIGNED"
    return "UNCERTAIN"


def derive_local_model_evidence_alignment_v2_8(agent1: dict[str, Any], agent2: dict[str, Any]) -> str:
    values = {
        str(agent1.get("ecg_self_score_consistency", "")).upper(),
        str(agent2.get("clinical_self_score_consistency", "")).upper(),
    }
    if values <= {"LOCAL_CONCORDANT"}:
        return "ALIGNED"
    if any(item in values for item in {"LOCAL_SCORE_POSSIBLY_OVERSTATED", "LOCAL_SCORE_POSSIBLY_UNDERSTATED"}):
        return "DISAGREEMENT"
    return "UNCERTAIN"


def derive_ecg_baseline_relation_v2_8(
    payload: dict[str, Any],
    agent1: dict[str, Any],
    agent2: dict[str, Any],
) -> str:
    anchor = payload.get("ecg_anchor_context") or {}
    state = str(anchor.get("ecg_baseline_state", ""))
    a1_state = str(agent1.get("ecg_model_audit_state") or agent1.get("ecg_local_concern") or "")
    a2_direction = _agent2_direction(agent2)
    active_head = str(agent2.get("selected_discrepancy_head") or "").upper()
    alert = str(agent2.get("alert_state") or "").upper() == "ALERT"
    under_alert = active_head == "UNDERESTIMATION" and alert
    over_alert = active_head == "OVERESTIMATION" and alert
    a2_band = str(agent2.get("risk_band") or "").upper()
    a2_support = str(agent2.get("evidence_stance") or "").upper()
    has_strong_agent2 = a2_band == "HIGH" or a2_support == "SUPPORTS_DISCREPANCY" or under_alert or over_alert
    if state == "ECG_MODEL_NEGATIVE":
        if under_alert or "UNDER" in a2_direction or a1_state == "POSSIBLE_UNDERESTIMATION":
            return (
                "ECG_BASELINE_POSSIBLY_UNDERSTATED"
                if has_strong_agent2 or a1_state == "POSSIBLE_UNDERESTIMATION"
                else "ECG_BASELINE_RELATION_UNRESOLVED"
            )
        return "ECG_BASELINE_SUPPORTED"
    if state == "ECG_MODEL_POSITIVE":
        if over_alert or "OVER" in a2_direction or a1_state == "POSSIBLE_OVERESTIMATION":
            return (
                "ECG_BASELINE_POSSIBLY_OVERSTATED"
                if has_strong_agent2
                else "ECG_BASELINE_RELATION_UNRESOLVED"
            )
        return "ECG_BASELINE_SUPPORTED"
    return "ECG_BASELINE_RELATION_UNRESOLVED"


def _v3_strength_grade(*, quantitative: str, semantic: str, safety_block: str) -> str:
    if safety_block not in {"NONE", ""}:
        return "WEAK"
    if quantitative == "SUPPORTED" and semantic == "SUPPORTED":
        return "MODERATE"
    if quantitative == "LIMITED" and semantic == "SUPPORTED":
        return "WEAK"
    return "NONE"


def derive_canonical_correction_relation_v3_core(
    *,
    agent1: dict[str, Any],
    agent2: dict[str, Any],
    ecg_anchor_context: dict[str, Any],
    rag_status: dict[str, Any] | None = None,
    provisional_relation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministic Core V3 evidence adjudication.

    This function is intentionally label-free, API-free, RAG-retriever-free,
    and model-score-free. It consumes only the locked Agent2 head/risk-band/
    alert outputs plus patient-level semantic evidence already committed by
    Agent1/Agent2. Patient-level feature contributions are not required and
    are not used as an independent direction vote.
    """

    state = str(ecg_anchor_context.get("ecg_baseline_state", "")).upper()
    selected_head = str(agent2.get("selected_discrepancy_head", "")).upper()
    risk_band = str(agent2.get("risk_band", "")).upper()
    risk_alert = str(agent2.get("alert_state", "")) == "ALERT"

    quant_limited = risk_band == "INTERMEDIATE"
    quant_supported = risk_band in {"INTERMEDIATE", "HIGH"} or risk_alert
    quantitative_support_status = (
        "SUPPORTED" if risk_band == "HIGH" or risk_alert else
        "LIMITED" if quant_limited else
        "NONE"
    )

    evidence_stance = str(agent2.get("evidence_stance", "INSUFFICIENT")).upper()
    ecg_pattern = str(agent1.get("ecg_evidence_pattern", "")).upper()
    ecg_strength = str(agent1.get("ecg_directional_strength", "")).upper()
    hidden_ischemia = bool(agent1.get("hidden_ischemia_warning"))
    mimic_warning = bool(agent1.get("false_positive_mimic_warning"))
    mimic_ids = list(agent1.get("non_specific_or_confounding_evidence_ids", []) or [])
    ischemic_ids = list(agent1.get("ischemic_evidence_ids", []) or [])
    missing_ids = list(agent2.get("critical_missing_factor_ids", []) or [])
    rag = rag_status or {}
    rag_contradiction = str(rag.get("rag_evidence_status", "")).upper() in {
        "RAG_CONTRADICTORY", "RAG_SAFETY_WARNING",
    }

    under_semantic = (
        (selected_head == "UNDERESTIMATION" and evidence_stance == "SUPPORTS_DISCREPANCY")
        or hidden_ischemia
        or ecg_pattern == "ISCHEMIA_SUPPORTIVE"
        or bool(ischemic_ids)
    )
    over_semantic = (
        (selected_head == "OVERESTIMATION" and evidence_stance == "SUPPORTS_DISCREPANCY")
        or (mimic_warning and bool(mimic_ids))
        or ecg_pattern == "NON_SPECIFIC_OR_CONFOUNDED"
    )
    ecg_support_status = (
        "HIDDEN_ISCHEMIA_SUPPORT" if hidden_ischemia else
        "MIMIC_SUPPORT" if mimic_warning and mimic_ids else
        "ISCHEMIC_SUPPORT" if ecg_pattern == "ISCHEMIA_SUPPORTIVE" or ischemic_ids else
        "NON_SPECIFIC_SUPPORT" if ecg_pattern == "NON_SPECIFIC_OR_CONFOUNDED" else
        "NONE"
    )

    safety_block_status = "NONE"
    if selected_head == "OVERESTIMATION" and (
        hidden_ischemia
        or (selected_head == "OVERESTIMATION" and evidence_stance == "SUPPORTS_DISCREPANCY" and not over_semantic)
        or ecg_pattern == "ISCHEMIA_SUPPORTIVE"
        or bool(ischemic_ids)
        or rag_contradiction
    ):
        safety_block_status = "DOWN_BLOCKED_OR_WEAKENED"
    if len(missing_ids) >= 3 and selected_head == "OVERESTIMATION":
        safety_block_status = "DOWN_BLOCKED_OR_WEAKENED"

    relation = "ECG_BASELINE_SUPPORTED"
    semantic_support_status = "NONE"
    if state == "ECG_MODEL_NEGATIVE" and selected_head == "UNDERESTIMATION":
        if quant_supported and under_semantic:
            relation = "ECG_BASELINE_POSSIBLY_UNDERSTATED"
            semantic_support_status = "SUPPORTED"
        elif quant_supported or under_semantic:
            relation = "ECG_BASELINE_RELATION_UNRESOLVED"
            semantic_support_status = "PARTIAL" if under_semantic else "NONE"
    elif state == "ECG_MODEL_POSITIVE" and selected_head == "OVERESTIMATION":
        if quant_supported and over_semantic and safety_block_status == "NONE":
            relation = "ECG_BASELINE_POSSIBLY_OVERSTATED"
            semantic_support_status = "SUPPORTED"
        elif quant_supported or over_semantic:
            relation = "ECG_BASELINE_RELATION_UNRESOLVED"
            semantic_support_status = "PARTIAL" if over_semantic else "NONE"
    elif state not in {"ECG_MODEL_NEGATIVE", "ECG_MODEL_POSITIVE"}:
        relation = "ECG_BASELINE_RELATION_UNRESOLVED"

    grade = _v3_strength_grade(
        quantitative=quantitative_support_status,
        semantic=semantic_support_status,
        safety_block=safety_block_status if relation != "ECG_BASELINE_RELATION_UNRESOLVED" else "NONE",
    )
    if relation in {"ECG_BASELINE_SUPPORTED", "ECG_BASELINE_RELATION_UNRESOLVED"}:
        grade = "NONE"
    confidence = {"NONE": "LOW", "WEAK": "LOW", "MODERATE": "MEDIUM", "STRONG": "HIGH"}[grade]
    directional_signal = relation in {
        "ECG_BASELINE_POSSIBLY_UNDERSTATED",
        "ECG_BASELINE_POSSIBLY_OVERSTATED",
    }
    trace = [{
        "llm_provisional_relation": (provisional_relation or {}).get("llm_provisional_relation"),
        "pre_adjudication_ecg_relation": (provisional_relation or {}).get("pre_adjudication_ecg_relation"),
        "selected_head": selected_head,
        "risk_band": risk_band,
        "risk_alert": risk_alert,
        "quantitative_support_status": quantitative_support_status,
        "clinical_semantic_support": semantic_support_status,
        "ecg_semantic_support": ecg_support_status,
        "rag_safety_status": "CONTRADICTION_OR_SAFETY" if rag_contradiction else "NONE",
        "safety_block_status": safety_block_status,
        "canonical_relation": relation,
        "canonical_grade": grade,
        "canonical_confidence": confidence,
        "canonical_relation_source": "DETERMINISTIC_V3_CORE_EVIDENCE_ADJUDICATION",
    }]
    return {
        "correction_relation_relative_to_ecg": relation,
        "ecg_baseline_relation": relation,
        "correction_evidence_grade": grade,
        "correction_confidence_relative_to_ecg": confidence,
        "directional_signal_present": directional_signal,
        "correction_relation_basis": "DETERMINISTIC_V3_CORE_EVIDENCE_ADJUDICATION",
        "quantitative_support_status": quantitative_support_status,
        "semantic_support_status": semantic_support_status,
        "ecg_support_status": ecg_support_status,
        "safety_block_status": safety_block_status,
        "canonical_relation_source": "DETERMINISTIC_V3_CORE_EVIDENCE_ADJUDICATION",
        "selected_discrepancy_head": selected_head,
        "risk_band": risk_band,
        "risk_alert": risk_alert,
        "relation_adjudication_trace": trace,
    }


def _ecg_only_cag_prediction_class(ecg_anchor_context: dict[str, Any]) -> str:
    explicit = str(ecg_anchor_context.get("ecg_only_cag_prediction_class", "")).upper()
    if explicit in {"PREDICTED_CAG_POSITIVE", "PREDICTED_CAG_NEGATIVE"}:
        return explicit
    state = str(ecg_anchor_context.get("ecg_baseline_state", "")).upper()
    if state == "ECG_MODEL_POSITIVE":
        return "PREDICTED_CAG_POSITIVE"
    if state == "ECG_MODEL_NEGATIVE":
        return "PREDICTED_CAG_NEGATIVE"
    return "PREDICTION_UNAVAILABLE"


def _explicit_mimic_support_v3_1(agent1: dict[str, Any]) -> bool:
    if not bool(agent1.get("false_positive_mimic_warning")):
        return False
    ids = {str(item) for item in agent1.get("non_specific_or_confounding_evidence_ids", []) or []}
    return bool(ids & {
        "left_ventricular_hypertrophy_mimic",
        "lvh_strain_secondary_st_t",
        "bundle_branch_block_mimic",
        "drug_or_electrolyte_st_t_mimic",
    })


def derive_canonical_correction_relation_v3_1(
    *,
    agent1: dict[str, Any],
    agent2: dict[str, Any],
    ecg_anchor_context: dict[str, Any],
    rag_status: dict[str, Any] | None = None,
    provisional_relation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Candidate V3.1 minimal semantic hardening evidence adjudication.

    V3.1 clarifies that the ECG baseline is an ECG-only prediction of CAG
    risk, not an ECG diagnosis label. Generic nonspecific ECG findings or QC
    limitations are not sufficient DOWN evidence. Missing clinical fields
    reduce confidence through Agent2 evidence sufficiency; their raw count is
    not a second absolute DOWN veto.
    """

    prediction_class = _ecg_only_cag_prediction_class(ecg_anchor_context)
    selected_head = str(agent2.get("selected_discrepancy_head", "")).upper()
    risk_band = str(agent2.get("risk_band", "")).upper()
    risk_alert = str(agent2.get("alert_state", "")) == "ALERT"
    quantitative_support_status = (
        "SUPPORTED" if risk_band == "HIGH" or risk_alert else
        "LIMITED" if risk_band == "INTERMEDIATE" else
        "NONE"
    )
    quant_supported = quantitative_support_status in {"SUPPORTED", "LIMITED"}

    evidence_stance = str(agent2.get("evidence_stance", "INSUFFICIENT")).upper()
    agent2_evidence_sufficiency = str(agent2.get("evidence_sufficiency", "INSUFFICIENT")).upper()
    ecg_pattern = str(agent1.get("ecg_evidence_pattern", "")).upper()
    hidden_ischemia = bool(agent1.get("hidden_ischemia_warning"))
    ischemic_ids = list(agent1.get("ischemic_evidence_ids", []) or [])
    missing_ids = list(agent2.get("critical_missing_factor_ids", []) or [])
    rag = rag_status or {}
    rag_contradiction = str(rag.get("rag_evidence_status", "")).upper() in {
        "RAG_CONTRADICTORY", "RAG_SAFETY_WARNING",
    }

    under_semantic = selected_head == "UNDERESTIMATION" and evidence_stance == "SUPPORTS_DISCREPANCY"
    if hidden_ischemia or ecg_pattern == "ISCHEMIA_SUPPORTIVE" or bool(ischemic_ids):
        under_semantic = True
    explicit_mimic = _explicit_mimic_support_v3_1(agent1)
    strong_mixed_under_support = (
        selected_head == "UNDERESTIMATION"
        and evidence_stance == "MIXED"
        and agent2_evidence_sufficiency in {"LIMITED", "SUFFICIENT"}
        and risk_band == "HIGH"
        and risk_alert
    )
    strong_mixed_over_support = (
        selected_head == "OVERESTIMATION"
        and evidence_stance == "MIXED"
        and agent2_evidence_sufficiency in {"LIMITED", "SUFFICIENT"}
        and risk_band == "HIGH"
        and risk_alert
    )
    over_semantic = (
        selected_head == "OVERESTIMATION"
        and evidence_stance == "SUPPORTS_DISCREPANCY"
        and agent2_evidence_sufficiency != "INSUFFICIENT"
    ) or strong_mixed_over_support or explicit_mimic

    safety_block_status = "NONE"
    if selected_head == "OVERESTIMATION" and (
        hidden_ischemia
        or (selected_head == "OVERESTIMATION" and evidence_stance == "SUPPORTS_DISCREPANCY" and not over_semantic)
        or ecg_pattern == "ISCHEMIA_SUPPORTIVE"
        or bool(ischemic_ids)
        or rag_contradiction
    ):
        safety_block_status = "DOWN_BLOCKED_OR_WEAKENED"
    missingness_caution_status = (
        "MATERIAL_MISSINGNESS_PRESENT" if len(missing_ids) >= 3 else
        "LIMITED_MISSINGNESS_PRESENT" if missing_ids else
        "NONE"
    )

    relation = "ECG_BASELINE_SUPPORTED"
    semantic_support_status = "NONE"
    if prediction_class == "PREDICTED_CAG_NEGATIVE" and selected_head == "UNDERESTIMATION":
        if quant_supported and (under_semantic or strong_mixed_under_support):
            relation = "ECG_BASELINE_POSSIBLY_UNDERSTATED"
            semantic_support_status = "SUPPORTED"
        elif quant_supported or under_semantic or strong_mixed_under_support:
            relation = "ECG_BASELINE_RELATION_UNRESOLVED"
            semantic_support_status = "PARTIAL" if (under_semantic or strong_mixed_under_support) else "NONE"
    elif prediction_class == "PREDICTED_CAG_POSITIVE" and selected_head == "OVERESTIMATION":
        if quant_supported and over_semantic and safety_block_status == "NONE":
            relation = "ECG_BASELINE_POSSIBLY_OVERSTATED"
            semantic_support_status = "SUPPORTED"
        elif quant_supported or over_semantic:
            relation = "ECG_BASELINE_RELATION_UNRESOLVED"
            semantic_support_status = "PARTIAL" if over_semantic else "NONE"
    elif prediction_class not in {"PREDICTED_CAG_NEGATIVE", "PREDICTED_CAG_POSITIVE"}:
        relation = "ECG_BASELINE_RELATION_UNRESOLVED"

    grade = _v3_strength_grade(
        quantitative=quantitative_support_status,
        semantic=semantic_support_status,
        safety_block=safety_block_status if relation != "ECG_BASELINE_RELATION_UNRESOLVED" else "NONE",
    )
    if relation in {"ECG_BASELINE_SUPPORTED", "ECG_BASELINE_RELATION_UNRESOLVED"}:
        grade = "NONE"
    confidence = {"NONE": "LOW", "WEAK": "LOW", "MODERATE": "MEDIUM", "STRONG": "HIGH"}[grade]
    directional_signal = relation in {
        "ECG_BASELINE_POSSIBLY_UNDERSTATED",
        "ECG_BASELINE_POSSIBLY_OVERSTATED",
    }
    trace = [{
        "llm_provisional_relation": (provisional_relation or {}).get("llm_provisional_relation"),
        "pre_adjudication_ecg_relation": (provisional_relation or {}).get("pre_adjudication_ecg_relation"),
        "ecg_only_cag_prediction_class": prediction_class,
        "selected_head": selected_head,
        "risk_band": risk_band,
        "risk_alert": risk_alert,
        "quantitative_support_status": quantitative_support_status,
        "clinical_semantic_support": semantic_support_status,
        "agent2_evidence_sufficiency": agent2_evidence_sufficiency,
        "strong_mixed_under_support": strong_mixed_under_support,
        "strong_mixed_over_support": strong_mixed_over_support,
        "explicit_mimic_support": explicit_mimic,
        "nonspecific_ecg_supports_down": False,
        "critical_missing_factor_count": len(missing_ids),
        "missingness_caution_status": missingness_caution_status,
        "missingness_is_absolute_down_veto": False,
        "rag_safety_status": "CONTRADICTION_OR_SAFETY" if rag_contradiction else "NONE",
        "safety_block_status": safety_block_status,
        "canonical_relation": relation,
        "canonical_grade": grade,
        "canonical_confidence": confidence,
        "canonical_relation_source": "DETERMINISTIC_V3_1_COMMITTED_EVIDENCE_ADJUDICATION",
    }]
    return {
        "correction_relation_relative_to_ecg": relation,
        "ecg_baseline_relation": relation,
        "correction_evidence_grade": grade,
        "correction_confidence_relative_to_ecg": confidence,
        "directional_signal_present": directional_signal,
        "correction_relation_basis": "DETERMINISTIC_V3_1_COMMITTED_EVIDENCE_ADJUDICATION",
        "quantitative_support_status": quantitative_support_status,
        "semantic_support_status": semantic_support_status,
        "ecg_support_status": "EXPLICIT_MIMIC_SUPPORT" if explicit_mimic else "NONE",
        "safety_block_status": safety_block_status,
        "canonical_relation_source": "DETERMINISTIC_V3_1_COMMITTED_EVIDENCE_ADJUDICATION",
        "ecg_only_cag_prediction_class": prediction_class,
        "selected_discrepancy_head": selected_head,
        "risk_band": risk_band,
        "risk_alert": risk_alert,
        "relation_adjudication_trace": trace,
    }


def derive_overall_confidence_v2_8(
    triage: dict[str, Any],
    *,
    evidence_sufficiency: str,
    modality_alignment: str,
) -> str:
    status = str(triage.get("global_consistency_status", "COMPLEX_DISCORDANCE"))
    if status == "CONSISTENT_CONFIDENT" and modality_alignment == "ALIGNED":
        return "HIGH"
    if modality_alignment == "DISAGREEMENT":
        return "MODERATE"
    if modality_alignment == "UNCERTAIN":
        return "LOW"
    if evidence_sufficiency == "INSUFFICIENT":
        return "MODERATE"
    return "MODERATE"


def deterministic_neutral_rag_gate_v2_8(
    *,
    evidence_sufficiency: str,
    modality_alignment: str,
    overall_confidence: str,
) -> dict[str, Any]:
    """Direction-neutral v2.8 RAG gate.

    RAG is allowed only when evidence is insufficient and the uncertainty is
    low-confidence or modality-not-aligned. Missing/invalid values fail closed
    to SKIP rather than defaulting to retrieval or any medical direction.
    """
    invalid = []
    if evidence_sufficiency not in V2_8_EVIDENCE_SUFFICIENCY:
        invalid.append(f"invalid_evidence_sufficiency:{evidence_sufficiency}")
    if modality_alignment not in V2_8_MODALITY_ALIGNMENT:
        invalid.append(f"invalid_modality_alignment:{modality_alignment}")
    if overall_confidence not in V2_8_OVERALL_CONFIDENCE:
        invalid.append(f"invalid_overall_confidence:{overall_confidence}")
    if invalid:
        return {
            "retrieval_decision": "SKIP",
            "retrieval_reason": "INSUFFICIENT_EVIDENCE",
            "deterministic_rag_gate_pass": False,
            "deterministic_rag_gate_fail_closed": True,
            "deterministic_rag_gate_override": True,
            "deterministic_rag_gate_override_reason": invalid,
        }
    modality_not_aligned = modality_alignment in {"DISAGREEMENT", "UNCERTAIN"}
    low_confidence = overall_confidence == "LOW"
    if modality_alignment == "DISAGREEMENT":
        reason = "MODALITY_DISAGREEMENT"
        if evidence_sufficiency == "INSUFFICIENT" and low_confidence:
            reason = "COMBINED_UNCERTAINTY"
        return {
            "retrieval_decision": "RETRIEVE",
            "retrieval_reason": reason,
            "deterministic_rag_gate_pass": True,
            "deterministic_rag_gate_fail_closed": False,
            "deterministic_rag_gate_override": False,
            "deterministic_rag_gate_override_reason": [],
        }
    if evidence_sufficiency == "INSUFFICIENT" and (low_confidence or modality_alignment == "UNCERTAIN"):
        if low_confidence and modality_not_aligned:
            reason = "COMBINED_UNCERTAINTY"
        elif low_confidence:
            reason = "LOW_CONFIDENCE"
        else:
            reason = "MODALITY_DISAGREEMENT"
        return {
            "retrieval_decision": "RETRIEVE",
            "retrieval_reason": reason,
            "deterministic_rag_gate_pass": True,
            "deterministic_rag_gate_fail_closed": False,
            "deterministic_rag_gate_override": False,
            "deterministic_rag_gate_override_reason": [],
        }
    return {
        "retrieval_decision": "SKIP",
        "retrieval_reason": "NONE" if evidence_sufficiency == "SUFFICIENT" else "INSUFFICIENT_EVIDENCE",
        "deterministic_rag_gate_pass": True,
        "deterministic_rag_gate_fail_closed": False,
        "deterministic_rag_gate_override": False,
        "deterministic_rag_gate_override_reason": [],
    }


def build_neutral_query_text_v2_8(plan: dict[str, Any]) -> str:
    concepts = list(plan.get("patient_ecg_concept_ids", []))[:2]
    factors = list(plan.get("patient_clinical_factor_ids", []))[:2]
    uncertainty = list(plan.get("patient_missingness_ids", []))[:2]
    terms = (
        concepts
        + factors
        + ["ECG clinical evidence review", "supporting and opposing evidence", "uncertainty"]
        + uncertainty
    )
    return " ".join(str(term).replace("_", " ") for term in dict.fromkeys(terms) if str(term).strip())


def _compact_patient_summary_query_terms(agent1: dict[str, Any], agent2: dict[str, Any], limit: int = 2) -> list[str]:
    terms: list[str] = []
    for text in (
        agent1.get("blind_reasoning_summary"),
        agent2.get("blind_reasoning_summary"),
    ):
        cleaned = " ".join(str(text or "").replace("\n", " ").split())
        if not cleaned:
            continue
        if len(cleaned) > 140:
            cleaned = cleaned[:140].rsplit(" ", 1)[0]
        terms.append(cleaned)
        if len(terms) >= limit:
            break
    return terms


def expand_query_with_cohort_priors_v2_6(
    query: dict[str, Any],
    patient_factor_matches: list[dict[str, Any]],
    branch_scope: str = "global_cag",
) -> dict[str, Any]:
    # v2.7 repair: cohort priors are a parallel auxiliary layer.  They may
    # support rerank/context after Stage-A has already triggered RAG, but they
    # must not add query terms, change rag_trigger, or trigger RAG alone.
    eligible: list[str] = []
    rerank = [
        str(item.get("factor_id")) for item in patient_factor_matches
        if item.get("eligible_for_retrieval_rerank") is True
        and str(item.get("branch_scope", "global_cag")) in {branch_scope, "global_cag"}
    ]
    terms = list(query.get("query_terms", []))
    return {
        **query,
        "query_terms": terms,
        "retrieval_query": " ".join(terms),
        "cohort_prior_factor_ids": [],
        "cohort_prior_rerank_factor_ids": sorted(set(rerank)),
        "cohort_prior_branch_scope": branch_scope,
        "cohort_prior_use": "rerank_or_context_only",
        "cohort_prior_query_expansion_enabled": False,
        "cohort_prior_query_expansion_used": False,
    }


def _truthy_v2_6(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


FACTOR_ID_ALIASES_V2_6: dict[str, str] = {}


def resolve_cohort_factor_id_v2_6(factor_id: str) -> str:
    text = str(factor_id).strip()
    if text in FACTOR_ID_ALIASES_V2_6:
        return FACTOR_ID_ALIASES_V2_6[text]
    raise ValueError(f"unknown v2.6 cohort factor alias: {factor_id}")


def _default_cohort_factor_registry_path_v2_6() -> Path:
    """Return the active train-only cohort prior registry path.

    v2.7 keeps the historical function name because the runtime still uses the
    v2.6 entrypoint, but the active registry is the v2.7 AB/CD train-only audit
    artifact when present.  The older data/processed registry remains a safe
    fallback for historical reports and tests.
    """

    root = Path(__file__).resolve().parents[2]
    active = root / "outputs" / "v2_7_abcd_factor_mining_audit" / "v2_7_abcd_cohort_factor_prior_registry.csv"
    if active.exists():
        return active
    return root / "data" / "processed" / "cohort_factor_prior_registry_v2_6.csv"


def _default_cohort_prior_auxiliary_cards_path_v2_7() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "knowledge_base" / "cohort_priors" / "v2_7_abcd_cohort_prior_cards.jsonl"


def _load_cohort_prior_auxiliary_card_index_v2_7(context: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    path_value = context.get("cohort_prior_auxiliary_cards_path")
    path = Path(str(path_value)) if path_value else _default_cohort_prior_auxiliary_cards_path_v2_7()
    if not path.exists():
        return {}
    index: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            card = json.loads(line)
            if str(card.get("card_type")) != "train_only_cohort_prior":
                continue
            if str(card.get("runtime_pool")) != "cohort_context":
                continue
            if card.get("patient_fact_allowed") is not False:
                continue
            if card.get("causal_claim_allowed") is not False:
                continue
            if card.get("support_patient_correction_allowed") is not False:
                continue
            if card.get("enter_patient_specific_applicable_cards") is not False:
                continue
            index[(str(card.get("contrast")), str(card.get("factor_id")))] = card
    return index


def _load_source_cohort_factor_registry_v2_6(context: dict[str, Any]) -> list[dict[str, str]]:
    path_value = (
        context.get("cohort_factor_layer_registry_path")
        or context.get("cohort_factor_registry_final_path")
        or context.get("source_cohort_factor_registry_path")
        or context.get("cohort_factor_prior_registry_path")
    )
    path = Path(str(path_value)) if path_value else _default_cohort_factor_registry_path_v2_6()
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _cohort_prior_sort_key_v2_6(row: dict[str, Any]) -> tuple[int, float, str]:
    try:
        rank = int(float(str(row.get("importance_rank", "")).strip()))
    except ValueError:
        rank = 9999
    try:
        stability = float(str(row.get("stability_frequency", "")).strip())
    except ValueError:
        stability = 0.0
    return (rank, -stability, str(row.get("factor_id", "")))


def _safe_cohort_prior_row_v2_6(row: dict[str, Any], branch: str) -> bool:
    runtime_use = str(row.get("runtime_use", "")).strip()
    runtime_status = str(row.get("runtime_prior_status", "")).strip()
    branch_scope = str(row.get("branch_scope", "")).strip()
    if branch_scope not in {branch, "global_cag"}:
        return False
    if str(row.get("train_only", "")).strip().lower() != "true":
        return False
    if str(row.get("source_scope", row.get("source_split", ""))).strip().lower() != "train":
        return False
    if str(row.get("patient_fact_allowed", "")).strip().lower() == "true":
        return False
    if str(row.get("causal_claim_allowed", "")).strip().lower() == "true":
        return False
    if runtime_use not in {"query_expansion", "retrieval_rerank"}:
        return False
    if runtime_status not in {"ELIGIBLE_QUERY_EXPANSION", "ELIGIBLE_RERANK_ONLY"}:
        return False
    return True


def select_source_cohort_factors_v2_6(
    context: dict[str, Any],
    branch: str,
    limit: int = 2,
) -> dict[str, Any]:
    """Select frozen train-only cohort factors as context, not patient facts.

    The final v2.6 factor layer intentionally allows asymmetric activation:
    AB may have zero valid frozen factors while CD can still provide hidden-risk
    cohort context. Continuous factors without a frozen patient threshold remain
    ``patient_match_status=not_evaluated`` and must not hard-gate retrieval.
    """

    if branch == "global_cag":
        return {
            "source_cohort_factor_ids": [],
            "cohort_factor_context": [],
            "linked_cohort_external_card_ids": [],
            "cohort_prior_auxiliary_card_ids": [],
            "cohort_factor_context_status": "NOT_APPLICABLE_GLOBAL_BRANCH",
        }
    registry = _load_source_cohort_factor_registry_v2_6(context)
    aux_cards = _load_cohort_prior_auxiliary_card_index_v2_7(context)
    selected = sorted(
        [row for row in registry if _safe_cohort_prior_row_v2_6(row, branch)],
        key=_cohort_prior_sort_key_v2_6,
    )[: max(0, int(limit))]
    if selected:
        context_rows = []
        auxiliary_card_ids: list[str] = []
        for row in selected:
            factor_id = str(row.get("factor_id", "")).strip()
            aux_card = aux_cards.get((str(row.get("contrast", row.get("analysis_target", ""))), factor_id))
            aux_card_id = str(aux_card.get("card_id", "")) if aux_card else ""
            if aux_card_id:
                auxiliary_card_ids.append(aux_card_id)
            context_rows.append({
                "factor_id": factor_id,
                "cohort_prior_auxiliary_card_id": aux_card_id,
                "canonical_factor_name": row.get("canonical_factor_name", factor_id),
                "branch_scope": row.get("branch_scope", branch),
                "contrast": row.get("contrast", row.get("analysis_target", "")),
                "association_direction": row.get("association_direction", ""),
                "effect_summary": row.get("effect_summary", row.get("effect_size", "")),
                "effect_size": row.get("effect_size", ""),
                "runtime_use": row.get("runtime_use", ""),
                "runtime_prior_status": row.get("runtime_prior_status", ""),
                "source_artifact": row.get("source_artifact", ""),
                "train_only": True,
                "source_scope": "train",
                "patient_fact_allowed": False,
                "causal_claim_allowed": False,
                "allowed_agent3_role": row.get(
                    "allowed_agent3_role",
                    "query_expansion_or_rerank_or_cohort_context_only",
                ),
                "patient_match_status": "not_evaluated",
                "usage_boundary": (
                    "train-only cohort association; not a patient fact, "
                    "not a causal claim, and not sufficient alone for correction"
                ),
            })
        return {
            "source_cohort_factor_ids": [str(row.get("factor_id", "")).strip() for row in selected],
            "cohort_factor_context": context_rows,
            "linked_cohort_external_card_ids": [],
            "cohort_prior_auxiliary_card_ids": sorted(set(auxiliary_card_ids)),
            "cohort_factor_context_status": "ACTIVE_TRAIN_ONLY_COHORT_FACTORS_SELECTED",
        }
    return {
        "source_cohort_factor_ids": [],
        "cohort_factor_context": [],
        "linked_cohort_external_card_ids": [],
        "cohort_prior_auxiliary_card_ids": [],
        "cohort_factor_context_status": "NO_BRANCH_MATCHING_COHORT_FACTORS",
    }


ALLOWED_KNOWLEDGE_NEED_TYPES = {
    "ecg_false_positive_mimic", "non_diagnostic_ecg_limitation",
    "hidden_high_risk_mechanism", "clinical_risk_mechanism",
    "drug_effect", "electrolyte_effect", "renal_metabolic_context",
    "contradictory_evidence",
}

QUERY_MEDICAL_CONCEPT_EXCLUSIONS = {
    "basic_qc", "large_amplitude_warning", "uncommon_signal_length",
    "machine_interpretation_available", "machine_interpretation_missing",
    "machine_interpretation_availability_unknown",
}

KNOWLEDGE_NEED_QUERY_PHRASES = {
    "ecg_false_positive_mimic": "ECG false positive alternative mechanism",
    "non_diagnostic_ecg_limitation": "non diagnostic ECG limitation",
    "hidden_high_risk_mechanism": "hidden high risk despite ECG",
    "clinical_risk_mechanism": "clinical risk mechanism",
    "drug_effect": "drug related ECG confounding",
    "electrolyte_effect": "electrolyte related ECG confounding",
    "renal_metabolic_context": "renal metabolic context",
    "contradictory_evidence": "evidence against candidate mechanism",
}

BRANCH_QUERY_PHRASES = {
    "global_cag": "ECG clinical discordance",
    "ecg_false_positive_like": "possible ECG false positive",
    "hidden_high_risk_like": "possible hidden clinical risk",
}

DRUG_CONFOUNDER_IDS = {
    "digoxin_use", "antiarrhythmic_use",
    "tricyclic_antidepressant_use", "st_t_affecting_drug_use",
}

ELECTROLYTE_CONCEPT_IDS = {
    "hypokalemia", "potassium_abnormality", "electrolyte_abnormality",
}

ECG_SPECIFICITY_TIER_1 = {
    "st_elevation", "st_depression", "left_bundle_branch_block",
    "right_bundle_branch_block", "atrial_fibrillation", "pathologic_q_wave",
    "q_wave_abnormality", "paced_rhythm", "persistent_st_elevation",
}

ECG_SPECIFICITY_TIER_2 = {
    "nonspecific_st_t_change", "conduction_abnormality", "rhythm_abnormality",
}

ECG_SPECIFICITY_TIER_3 = {
    "t_wave_abnormality", "abnormal_ecg",
}

CLINICAL_QUERY_PRIORITY = {
    "diabetes": 0,
    "stroke_history": 1,
    "peripheral_artery_disease": 2,
    "stroke_or_peripheral_artery_disease": 3,
    "hypertension": 8,
}


def _sort_ecg_by_specificity(concepts: list[str]) -> list[str]:
    def key(concept: str) -> tuple[int, str]:
        if concept in ECG_SPECIFICITY_TIER_1:
            return (0, concept)
        if concept in ECG_SPECIFICITY_TIER_2:
            return (1, concept)
        if concept in ECG_SPECIFICITY_TIER_3:
            return (2, concept)
        return (1, concept)
    return sorted(dict.fromkeys(map(str, concepts)), key=key)


def _select_ecg_query_concepts(concepts: list[str], limit: int = 2) -> list[str]:
    ordered = _sort_ecg_by_specificity(concepts)
    if any(concept in ECG_SPECIFICITY_TIER_1 for concept in ordered):
        ordered = [concept for concept in ordered if concept not in ECG_SPECIFICITY_TIER_3]
    elif any(concept in ECG_SPECIFICITY_TIER_2 for concept in ordered):
        ordered = [concept for concept in ordered if concept not in ECG_SPECIFICITY_TIER_3] or ordered
    return ordered[:limit]


def _select_clinical_query_factors(factors: list[str], limit: int = 2) -> list[str]:
    ordered = sorted(
        dict.fromkeys(map(str, factors)),
        key=lambda factor: (CLINICAL_QUERY_PRIORITY.get(factor, 5), factor),
    )
    high_priority_present = any(
        factor in ordered for factor in {
            "diabetes", "stroke_history", "peripheral_artery_disease",
            "stroke_or_peripheral_artery_disease",
        }
    )
    if high_priority_present and "hypertension" in ordered:
        ordered = [factor for factor in ordered if factor != "hypertension"] + ["hypertension"]
    return ordered[:limit]


def derive_evidence_sufficiency_v2_8(
    agent1: dict[str, Any],
    agent2: dict[str, Any],
    triage: dict[str, Any],
) -> tuple[str, dict[str, Any], list[str]]:
    """Separate patient-evidence sufficiency from RAG/query/model-score status.

    Query eligibility only answers whether patient facts can form a medical
    query. Evidence sufficiency answers whether Agent1-A/Agent2-A patient
    evidence is interpretable enough for Stage-A to preserve a direction,
    identify disagreement, or safely preserve. It intentionally ignores local
    score-consistency audit fields so Stage-A remains patient-evidence-led.
    """
    inputs = {
        "agent1_patient_direction": _agent1_direction(agent1),
        "agent1_directional_strength": agent1.get("ecg_directional_strength"),
        "agent1_ecg_evidence_informativeness": agent1.get("ecg_evidence_informativeness"),
        "agent1_ecg_signal_quality": agent1.get("ecg_signal_quality"),
        "agent1_ecg_uncertainty": agent1.get("ecg_uncertainty"),
        "agent1_ecg_local_concern": agent1.get("ecg_local_concern"),
        "agent1_ecg_audit_confidence": agent1.get("ecg_audit_confidence"),
        "agent1_patient_evidence_item_count": len(list(agent1.get("ecg_evidence_items", []))),
        "agent1_blind_reasoning_present": bool(str(agent1.get("blind_reasoning_summary", "")).strip()),
        "agent2_patient_direction": _agent2_direction(agent2),
        "agent2_evidence_stance": agent2.get("evidence_stance"),
        "agent2_evidence_sufficiency": agent2.get("evidence_sufficiency"),
        "agent2_patient_evidence_item_count": len(list(agent2.get("supporting_factor_ids", []))) + len(list(agent2.get("contradicting_factor_ids", []))) + len(list(agent2.get("confounder_factor_ids", []))),
        "agent2_blind_reasoning_present": bool(str(agent2.get("concise_rationale", "")).strip()),
        "global_consistency_status": triage.get("global_consistency_status"),
        "discordance_pattern_preliminary": triage.get("discordance_pattern_preliminary"),
    }
    reasons: list[str] = []
    a1_informative = str(inputs["agent1_ecg_evidence_informativeness"]) in {"MEDIUM", "HIGH"}
    a2_informative = str(inputs["agent2_evidence_sufficiency"]) in {"SUFFICIENT", "LIMITED"}
    a1_traceable = bool(inputs["agent1_patient_evidence_item_count"] or inputs["agent1_blind_reasoning_present"])
    a2_traceable = bool(inputs["agent2_patient_evidence_item_count"] or inputs["agent2_blind_reasoning_present"])
    a1_directional = str(inputs["agent1_patient_direction"]) in {"up", "down"}
    a2_directional = str(inputs["agent2_patient_direction"]) in {"up", "down"}
    a1_usable = (
        str(inputs["agent1_ecg_signal_quality"]) != "LOW"
        and str(inputs["agent1_ecg_uncertainty"]) != "HIGH"
        and a1_traceable
    )
    a2_usable = (
        str(inputs["agent2_evidence_stance"]) != "INSUFFICIENT" and a2_traceable
    )
    if a1_informative and a1_usable:
        reasons.append("agent1_patient_evidence_interpretable")
    if a2_informative and a2_usable:
        reasons.append("agent2_patient_evidence_interpretable")
    if a1_directional:
        reasons.append("agent1_directional_patient_signal_present")
    if a2_directional:
        reasons.append("agent2_directional_patient_signal_present")

    if len([reason for reason in reasons if reason.endswith("_patient_evidence_interpretable")]) >= 2:
        return "SUFFICIENT", inputs, reasons + ["both_modalities_patient_evidence_interpretable"]
    if (a1_directional and a1_informative and a1_usable) or (a2_directional and a2_informative and a2_usable):
        return "SUFFICIENT", inputs, reasons + ["single_directional_modality_with_traceable_patient_evidence"]
    if str(inputs["global_consistency_status"]) in {"CONSISTENT_CONFIDENT", "CONSISTENT_UNCERTAIN"} and reasons:
        return "SUFFICIENT", inputs, reasons + ["consistent_stage_a_patient_evidence_interpretable"]
    if str(inputs["global_consistency_status"]) == "INSUFFICIENT_EVIDENCE":
        return "INSUFFICIENT", inputs, reasons or ["triage_global_insufficient_patient_evidence"]
    return "INSUFFICIENT", inputs, reasons or ["insufficient_interpretable_patient_evidence"]


def build_retrieval_plan_v2_6(
    payload: dict[str, Any],
    agent1: dict[str, Any],
    agent2: dict[str, Any],
    triage: dict[str, Any],
) -> dict[str, Any]:
    branch = "global_cag"
    ecg_items = list(agent1.get("ecg_evidence_items", []))
    clinical_factor_ids = sorted(set(
        map(str, list(agent2.get("supporting_factor_ids", [])) + list(agent2.get("contradicting_factor_ids", [])))
    ))
    clinical_packet = build_safe_clinical_packet(payload)
    clinical_item_by_factor = {
        str(item.get("canonical_concept_id")): dict(item)
        for item in clinical_packet.get("clinical_evidence_items", [])
        if item.get("canonical_concept_id")
    }
    clinical_items: list[dict[str, Any]] = []
    for factor in clinical_factor_ids:
        item = dict(clinical_item_by_factor.get(factor, {}))
        if not item:
            item = {
                "evidence_id": f"CLINICAL:{factor}",
                "evidence_domain": "CLINICAL",
                "canonical_concept_id": factor,
                "status": "unknown",
                "patient_specific": True,
                "score_derived": False,
                "label_derived": False,
            }
        clinical_items.append(item)
    ecg = sorted(set(
        str(item.get("canonical_concept_id"))
        for item in ecg_items
        if item.get("projection_method") == "deterministic_lexical_concept_projection"
        and item.get("concept_status") == "present"
        and str(item.get("canonical_concept_id")) not in QUERY_MEDICAL_CONCEPT_EXCLUSIONS
    ))
    if not ecg and not ecg_items:
        ecg = sorted(set(map(str, agent1.get("retrieval_concepts", agent1.get("ecg_positive_evidence_terms", [])))))
    ecg = _sort_ecg_by_specificity(ecg)
    ecg_negative = sorted(set(
        str(item.get("canonical_concept_id"))
        for item in ecg_items
        if item.get("concept_status") == "absent"
        and item.get("projection_method") == "deterministic_lexical_concept_projection"
    ))
    clinical = _select_clinical_query_factors(clinical_factor_ids, limit=max(2, len(clinical_factor_ids)))
    confounders = sorted(set(map(str, agent2.get("confounder_factor_ids", []))))
    missingness = sorted(set(map(str, agent2.get("critical_missing_factor_ids", []))))
    patient_evidence = [
        item for item in ecg_items + clinical_items
        if item.get("patient_specific") is True
        and item.get("score_derived") is False
        and item.get("label_derived") is False
    ]
    evidence_ids = sorted(set(str(item.get("evidence_id")) for item in patient_evidence if item.get("evidence_id")))
    natural_language_query_terms = _compact_patient_summary_query_terms(agent1, agent2, limit=2)
    query_eligible_evidence_ids = sorted(set(
        str(item.get("evidence_id"))
        for item in patient_evidence
        if (
            item.get("evidence_domain") == "CLINICAL"
            and item.get("status") == "present"
        ) or (
            item.get("evidence_domain") == "ECG"
            and item.get("concept_status") == "present"
            and item.get("projection_method") == "deterministic_lexical_concept_projection"
            and str(item.get("canonical_concept_id")) not in QUERY_MEDICAL_CONCEPT_EXCLUSIONS
        )
    ))
    present_clinical_factor_ids = sorted(set(
        str(item.get("canonical_concept_id"))
        for item in clinical_items
        if str(item.get("status", "")).lower() in PATIENT_PRESENT_STATUSES
        and item.get("canonical_concept_id")
    ))
    needs: list[str] = []
    counter: list[str] = []
    rag_query_eligible = bool(query_eligible_evidence_ids or natural_language_query_terms)
    safe_fallback_reason = "" if rag_query_eligible else "SAFE_EVIDENCE_INSUFFICIENT_FALLBACK"
    evidence_sufficiency, sufficiency_inputs, sufficiency_reasons = derive_evidence_sufficiency_v2_8(
        agent1, agent2, triage
    )
    evidence_alignment = derive_evidence_alignment_v2_8(agent1, agent2)
    local_model_evidence_alignment = derive_local_model_evidence_alignment_v2_8(agent1, agent2)
    locked_value_error_codes: list[str] = []
    try:
        score_alignment = "ECG_ANCHOR_ONLY"
        ecg_baseline_relation = derive_ecg_baseline_relation_v2_8(payload, agent1, agent2)
    except Exception as exc:
        score_alignment = "LOCKED_VALUE_UNAVAILABLE_FAIL_CLOSED"
        ecg_baseline_relation = "ECG_BASELINE_RELATION_UNRESOLVED"
        locked_value_error_codes.append(str(exc))
    overall_confidence = derive_overall_confidence_v2_8(
        triage,
        evidence_sufficiency=evidence_sufficiency,
        modality_alignment=evidence_alignment,
    )
    pre_rag_patient_summary = assess_patient_evidence_for_rag_aware_policy_v2_7(agent1, agent2)
    gate = deterministic_neutral_rag_gate_v2_8(
        evidence_sufficiency=evidence_sufficiency,
        modality_alignment=evidence_alignment,
        overall_confidence=overall_confidence,
    )
    if locked_value_error_codes:
        gate = {
            "retrieval_decision": "SKIP",
            "retrieval_reason": "LOCKED_VALUE_UNAVAILABLE_FAIL_CLOSED",
            "deterministic_rag_gate_pass": False,
            "deterministic_rag_gate_fail_closed": True,
            "deterministic_rag_gate_override": True,
            "deterministic_rag_gate_override_reason": locked_value_error_codes,
        }
    if not rag_query_eligible:
        gate = {
            "retrieval_decision": "SKIP",
            "retrieval_reason": "INSUFFICIENT_EVIDENCE",
            "deterministic_rag_gate_pass": True,
            "deterministic_rag_gate_fail_closed": True,
            "deterministic_rag_gate_override": True,
            "deterministic_rag_gate_override_reason": [
                "no_patient_medical_evidence_available_for_neutral_query"
            ],
        }
    discrepancy_alert_active = str(agent2.get("alert_state", "")).upper() == "ALERT"
    ecg_anchor_relation_uncertain = ecg_baseline_relation in {
        "ECG_BASELINE_POSSIBLY_UNDERSTATED",
        "ECG_BASELINE_POSSIBLY_OVERSTATED",
        "ECG_BASELINE_RELATION_UNRESOLVED",
    }
    if (
        not locked_value_error_codes
        and rag_query_eligible
        and gate["retrieval_decision"] == "SKIP"
        and discrepancy_alert_active
        and ecg_anchor_relation_uncertain
    ):
        gate = {
            "retrieval_decision": "RETRIEVE",
            "retrieval_reason": "ECG_ANCHOR_DISCREPANCY_ALERT",
            "deterministic_rag_gate_pass": True,
            "deterministic_rag_gate_fail_closed": False,
            "deterministic_rag_gate_override": True,
            "deterministic_rag_gate_override_reason": [
                "agent2_locked_discrepancy_alert_with_uncertain_ecg_anchor_relation"
            ],
        }
    agent3b_rag_allowed = gate["retrieval_decision"] == "RETRIEVE"
    if agent3b_rag_allowed:
        source_cohort = select_source_cohort_factors_v2_6(payload, branch)
    else:
        source_cohort = {
            "source_cohort_factor_ids": [],
            "cohort_factor_context": [],
            "linked_cohort_external_card_ids": [],
            "cohort_prior_auxiliary_card_ids": [],
            "cohort_factor_context_status": "RAG_TRIGGER_NONE_NO_COHORT_PRIOR_USED",
        }
    if clinical:
        needs.append("clinical_risk_mechanism")
    if not ecg or evidence_alignment in {"DISAGREEMENT", "UNCERTAIN"}:
        needs.append("non_diagnostic_ecg_limitation")
    if (set(clinical) | set(ecg)) & ELECTROLYTE_CONCEPT_IDS:
        needs.append("electrolyte_effect")
    if set(clinical) & DRUG_CONFOUNDER_IDS:
        needs.append("drug_effect")
    if any(f in {"ckd", "renal_impairment"} for f in clinical):
        needs.append("renal_metabolic_context")
    if ecg or clinical:
        counter.append("contradictory_evidence")
    matches = list(agent2.get("patient_factor_matches", []))
    query_prior: list[str] = []
    rerank_prior = sorted(set(
        str(item.get("factor_id")) for item in matches
        if item.get("eligible_for_retrieval_rerank") is True
        and str(item.get("branch_scope", "global_cag")) in {branch, "global_cag"}
    ))
    excluded: list[str] = []
    exclusion_reasons: list[str] = []
    for item in matches:
        factor = str(item.get("factor_id", ""))
        if factor in query_prior or factor in rerank_prior:
            continue
        excluded.append(factor)
        exclusion_reasons.append(
            f"{factor}:{item.get('patient_factor_status', 'unknown')}:{item.get('runtime_prior_status', 'not_eligible')}"
        )
    neutral_query_terms = (
        natural_language_query_terms[:2]
        + _select_ecg_query_concepts(ecg, limit=2)
        + _select_clinical_query_factors(clinical, limit=2)
        + _select_ecg_query_concepts(ecg_negative, limit=1)
        + confounders[:1]
        + missingness[:1]
        + [str(triage.get("discordance_pattern_preliminary", "")).lower()]
    )
    neutral_query = " ".join(
        str(term).replace("_", " ")
        for term in dict.fromkeys(neutral_query_terms)
        if str(term).strip()
    ) if agent3b_rag_allowed else ""
    return {
        "selected_branch": branch,
        "branch_reason": list(triage.get("triage_trigger_reasons", [])) or ["patient_evidence_concordance_or_unclear"],
        "patient_ecg_concepts": ecg,
        "patient_ecg_concept_ids": ecg,
        "patient_negative_ecg_concept_ids": ecg_negative,
        "patient_clinical_factor_ids": clinical,
        "patient_present_clinical_factor_ids": present_clinical_factor_ids,
        "patient_confounder_ids": confounders,
        "patient_missingness_ids": missingness,
        "patient_specific_evidence_ids": evidence_ids,
        "query_eligible_patient_evidence_ids": query_eligible_evidence_ids,
        "natural_language_query_terms": natural_language_query_terms,
        "natural_language_query_without_concept_ids_supported": bool(natural_language_query_terms),
        "patient_evidence_trace": patient_evidence,
        "no_projectable_patient_evidence": not bool(evidence_ids),
        "discordance_type": str(triage.get("discordance_pattern_preliminary", "unclear")),
        "knowledge_need_types": sorted(set(needs) & ALLOWED_KNOWLEDGE_NEED_TYPES),
        "counter_evidence_need_types": sorted(set(counter) & ALLOWED_KNOWLEDGE_NEED_TYPES),
        "cohort_prior_query_factor_ids": query_prior,
        "cohort_prior_rerank_factor_ids": rerank_prior,
        "source_cohort_factor_ids": source_cohort["source_cohort_factor_ids"],
        "cohort_factor_context": source_cohort["cohort_factor_context"],
        "linked_cohort_external_card_ids": source_cohort["linked_cohort_external_card_ids"],
        "cohort_prior_auxiliary_card_ids": source_cohort["cohort_prior_auxiliary_card_ids"],
        "cohort_factor_context_status": source_cohort["cohort_factor_context_status"],
        "excluded_factor_ids": sorted(set(excluded)),
        "exclusion_reasons": sorted(set(exclusion_reasons)),
        "score_generated_medical_fact_count": 0,
        "rag_query_eligible": rag_query_eligible,
        "query_eligibility": rag_query_eligible,
        "query_eligibility_reason_codes": (
            ["patient_specific_present_medical_evidence_available"] if query_eligible_evidence_ids
            else ["natural_language_patient_summary_available_for_query"] if natural_language_query_terms
            else ["no_patient_specific_present_medical_evidence_for_query"]
        ),
        "safe_fallback_reason": safe_fallback_reason,
        "evidence_sufficiency": evidence_sufficiency,
        "evidence_sufficiency_rule_id": "v2_8_agent1_agent2_local_evidence_sufficiency_v1",
        "evidence_sufficiency_rule_inputs": sufficiency_inputs,
        "evidence_sufficiency_reason_codes": sufficiency_reasons,
        "evidence_alignment": evidence_alignment,
        "local_model_evidence_alignment": local_model_evidence_alignment,
        "score_alignment": score_alignment,
        "ecg_baseline_relation": ecg_baseline_relation,
        "locked_value_error_codes": locked_value_error_codes,
        "modality_alignment": evidence_alignment,
        "overall_confidence": overall_confidence,
        "pre_rag_patient_evidence_direction": pre_rag_patient_summary["patient_evidence_direction"],
        "pre_rag_patient_evidence_strength": pre_rag_patient_summary["patient_evidence_strength"],
        "pre_rag_patient_evidence_basis": pre_rag_patient_summary["patient_evidence_basis"],
        "pre_rag_evidence_alignment": evidence_alignment,
        "pre_rag_evidence_sufficiency": evidence_sufficiency,
        "pre_rag_overall_confidence": overall_confidence,
        "retrieval_decision": gate["retrieval_decision"],
        "retrieval_reason": gate["retrieval_reason"],
        "neutral_query": neutral_query,
        "deterministic_rag_gate_pass": gate["deterministic_rag_gate_pass"],
        "deterministic_rag_gate_fail_closed": gate["deterministic_rag_gate_fail_closed"],
        "deterministic_rag_gate_override": gate["deterministic_rag_gate_override"],
        "deterministic_rag_gate_override_reason": gate["deterministic_rag_gate_override_reason"],
        "agent3b_rag_allowed": agent3b_rag_allowed,
        "cohort_prior_changed_rag_trigger": False,
        "cohort_prior_triggered_rag_alone": False,
        "cohort_prior_query_expansion_enabled": False,
        "cohort_prior_query_expansion_used": False,
    }


def build_agent3_stage_a_only_output_v2_7(
    payload: dict[str, Any],
    agent1: dict[str, Any],
    agent2: dict[str, Any],
    triage: dict[str, Any],
    retrieval_plan: dict[str, Any],
) -> dict[str, Any]:
    """Build a schema-valid Agent3 Stage-A-only output when RAG is not allowed.

    This is the strict conditional-RAG path for v2.7:
    Stage-A gate executes, Stage-B STANDARD_RAG is explicitly skipped, no
    retrieval query is constructed, no medical card is retrieved, and cohort
    priors are not used.
    """

    global_status = str(triage.get("global_consistency_status", "COMPLEX_DISCORDANCE"))
    patient_evidence_ids = list(retrieval_plan.get("query_eligible_patient_evidence_ids", []))
    if not patient_evidence_ids:
        patient_evidence_ids = list(retrieval_plan.get("patient_specific_evidence_ids", []))
    score_ids = [
        "locked_ecg_score"
    ] if payload.get("ecg_anchor_context", {}).get("locked_ecg_score") is not None else []
    if global_status == "CONSISTENT_CONFIDENT":
        proposal_direction = "PRESERVE"
        proposal_strength = "NONE"
        proposal_confidence = "HIGH"
        sufficiency = "SUFFICIENT"
        uncertainty = []
        reason = "Stage-A found sufficient aligned evidence; neutral RAG gate skipped Stage-B."
    elif not retrieval_plan.get("rag_query_eligible", False):
        proposal_direction = "REVIEW_ONLY"
        proposal_strength = "NONE"
        proposal_confidence = "LOW"
        sufficiency = "INSUFFICIENT"
        uncertainty = ["no_query_eligible_patient_evidence"]
        reason = "Stage-A found insufficient query-eligible patient evidence; neutral RAG gate skipped Stage-B."
    else:
        proposal_direction = "REVIEW_ONLY"
        proposal_strength = "NONE"
        proposal_confidence = "LOW"
        sufficiency = "PARTIAL"
        uncertainty = ["rag_not_required_by_stage_a_gate"]
        reason = "Stage-A neutral RAG gate did not require retrieval; Stage-B was skipped."
    output = {
        "schema_version": V2_6_SCHEMA_VERSION,
        "agent_name": "Agent3_CrossModal_Discrepancy_Arbitration_RAG_Agent",
        "case_id": str(payload.get("case_id", "")),
        "safety_boundary_check": V2_6_SAFE_BOUNDARY_CHECK,
        "global_consistency_status": global_status,
        "patient_evidence_ids": sorted(set(map(str, patient_evidence_ids))),
        "model_evidence_ids": score_ids,
        "cohort_background_factor_ids": [],
        "external_knowledge_card_ids": [],
        "retrieved_card_ids": [],
        "applicable_card_ids": [],
        "actionable_card_ids": [],
        "cited_card_ids": [],
        "supporting_card_ids": [],
        "veto_card_ids": [],
        "supporting_evidence": [],
        "contradictory_evidence": [],
        "supporting_evidence_ids": [],
        "contradicting_evidence_ids": [],
        "remaining_uncertainty": uncertainty,
        "brief_arbitration_rationale": reason,
        "proposal_direction": proposal_direction,
        "proposal_strength": proposal_strength,
        "proposal_confidence": proposal_confidence,
        "proposal_reason": reason,
        "retrieval_round_index": 0,
        "rag_mode": "STANDARD_RAG",
        "retrieval_plan": {
            **retrieval_plan,
            "agent3b_rag_allowed": False,
            "cohort_prior_query_factor_ids": [],
            "cohort_prior_rerank_factor_ids": [],
            "source_cohort_factor_ids": [],
            "cohort_factor_context": [],
            "linked_cohort_external_card_ids": [],
            "cohort_prior_auxiliary_card_ids": [],
            "cohort_prior_query_expansion_enabled": False,
            "cohort_prior_query_expansion_used": False,
        },
        "claim_evidence_chain": [{
            "claim_id": "stage_a_gate_claim",
            "claim_type": "stage_a_consistency_gate",
            "claim_text": reason,
            "patient_evidence_ids": sorted(set(map(str, patient_evidence_ids))),
            "model_evidence_ids": score_ids,
            "cohort_prior_factor_ids": [],
            "external_card_ids": [],
            "support_status": "patient_only" if patient_evidence_ids else "unsupported",
            "counterevidence_card_ids": [],
            "uncertainty": "; ".join(uncertainty) if uncertainty else "none",
        }],
        "agent3_stageA_executed": True,
        "agent3_stageB_executed": False,
        "rag_trigger": "NEUTRAL_STAGE_A_GATE",
        "evidence_sufficiency": str(retrieval_plan.get("evidence_sufficiency", sufficiency)),
        "modality_alignment": str(retrieval_plan.get("modality_alignment", "UNCERTAIN")),
        "overall_confidence": str(retrieval_plan.get("overall_confidence", proposal_confidence)),
        "retrieval_decision": str(retrieval_plan.get("retrieval_decision", "SKIP")),
        "retrieval_reason": str(retrieval_plan.get("retrieval_reason", "NONE")),
        "ecg_baseline_relation": str(retrieval_plan.get("ecg_baseline_relation", "ECG_BASELINE_RELATION_UNRESOLVED")),
        "deterministic_rag_gate_override": bool(retrieval_plan.get("deterministic_rag_gate_override", False)),
        "deterministic_rag_gate_override_reason": list(retrieval_plan.get("deterministic_rag_gate_override_reason", [])),
        "rag_used": False,
        "retrieval_executed": False,
        "retrieval_query_count": 0,
        "retrieved_card_count": 0,
        "agent3b_skipped_reason": "neutral_rag_gate_skip",
        "cohort_prior_used": False,
        "cohort_prior_rerank_used": False,
        "cohort_prior_context_used": False,
        "cohort_prior_query_expansion_used": False,
        "cohort_prior_enters_supporting_cards": False,
        "cohort_prior_enters_applicable_card_ids": False,
        "agent3_uncertainty_reasons": uncertainty,
    }
    output = enrich_agent3_output_with_rag_aware_fields_v2_7(
        output,
        agent1=agent1,
        agent2=agent2,
        ecg_anchor_context=payload.get("ecg_anchor_context", {}),
        retrieved_cards=[],
        canonical_relation_v3_core_enabled=bool(
            payload.get("canonical_relation_v3_core_enabled")
            or str(payload.get("runtime_policy_variant", "")).endswith("candidate_v3_core_canonical")
        ),
        canonical_relation_v3_1_enabled=bool(
            payload.get("canonical_relation_v3_1_enabled")
            or "candidate_v3_1_minimal_semantic_hardening" in str(payload.get("runtime_policy_variant", ""))
        ),
    )
    # SKIP is a deterministic gate result.  Keep the persisted Agent3 record
    # on the same compact contract used by the RETRIEVE path; old query plans
    # and trace structures are local Runtime data, not Agent3 output fields.
    output = {
        "schema_version": V2_6_SCHEMA_VERSION,
        "agent_name": "Agent3_CrossModal_Discrepancy_Arbitration_RAG_Agent",
        "case_id": str(payload.get("case_id", "")),
        "safety_boundary_check": V2_6_SAFE_BOUNDARY_CHECK,
        "evidence_alignment": str(output.get("modality_alignment", "UNCERTAIN")),
        "evidence_sufficiency": str(output.get("evidence_sufficiency", "INSUFFICIENT")),
        "overall_confidence": str(output.get("overall_confidence", "LOW")),
        "supporting_evidence_ids": [],
        "contradicting_evidence_ids": [],
        "remaining_uncertainty": list(output.get("remaining_uncertainty", [])),
        "applicable_card_ids": [],
        "supporting_card_ids": [],
        "veto_card_ids": [],
        "brief_arbitration_rationale": reason,
        "global_consistency_status": global_status,
        "retrieval_decision": str(retrieval_plan.get("retrieval_decision", "SKIP")),
        "retrieval_reason": str(retrieval_plan.get("retrieval_reason", "NONE")),
        "retrieved_card_ids": [],
        "retrieval_round_index": 0,
        "agent3_stageA_executed": True,
        "agent3_stageB_executed": False,
        "rag_used": False,
        "retrieval_executed": False,
        "selected_discrepancy_head": agent2.get("selected_discrepancy_head", ""),
        "correction_relation_relative_to_ecg": output.get("correction_relation_relative_to_ecg", "ECG_BASELINE_RELATION_UNRESOLVED"),
        "correction_evidence_grade": output.get("correction_evidence_grade", "NONE"),
        "correction_confidence_relative_to_ecg": output.get("correction_confidence_relative_to_ecg", "LOW"),
        "canonical_relation_source": output.get("canonical_relation_source", "DETERMINISTIC_AGENT3_EVIDENCE_ADJUDICATION"),
    }
    forbidden = find_forbidden_v2_6_fields(output)
    if forbidden:
        raise ValueError(f"forbidden Agent3 Stage-A output fields: {forbidden}")
    validate_v2_6_agent_output("agent3", output)
    return output


def build_multi_query_bundle_v2_6(plan: dict[str, Any], round_index: int = 0) -> list[dict[str, Any]]:
    if plan.get("rag_query_eligible") is False or plan.get("agent3b_rag_allowed") is False:
        return []
    evidence_trace = list(plan.get("patient_evidence_trace", []))
    evidence_by_concept: dict[str, list[str]] = {}
    for item in evidence_trace:
        concept = str(item.get("canonical_concept_id", ""))
        evidence_id = str(item.get("evidence_id", ""))
        if concept and evidence_id:
            evidence_by_concept.setdefault(concept, []).append(evidence_id)
    ecg = _select_ecg_query_concepts(list(plan.get("patient_ecg_concept_ids", [])), limit=2)
    clinical = _select_clinical_query_factors(list(plan.get("patient_clinical_factor_ids", [])), limit=2)
    confounders = [
        factor for factor in plan.get("patient_confounder_ids", [])
        if factor in set(plan.get("patient_clinical_factor_ids", []))
    ][:2]
    needs = list(plan.get("knowledge_need_types", []))
    counter = list(plan.get("counter_evidence_need_types", []))
    source_cohort_factor_ids = list(plan.get("source_cohort_factor_ids", []))

    query_specs: list[tuple[str, list[str], list[str], list[str]]] = []
    core_patient = (ecg[:2] if ecg else []) + clinical[: max(0, 2 - min(len(ecg), 2))]
    if not core_patient:
        return []
    core_terms = core_patient + ["ECG clinical evidence review", "supporting and opposing evidence", "uncertainty"]
    if needs:
        core_terms.append(KNOWLEDGE_NEED_QUERY_PHRASES[needs[0]])
    query_specs.append(("core_mechanism", core_terms, core_patient, []))

    branch_patient = confounders or clinical[1:2]
    branch_need = next(
        (need for need in needs[1:] if need in {"drug_effect", "electrolyte_effect", "clinical_risk_mechanism", "renal_metabolic_context"}),
        "",
    )
    if branch_patient or branch_need:
        branch_terms = branch_patient + ["clinical ECG uncertainty context", "alternative explanations"]
        if branch_need:
            branch_terms.append(KNOWLEDGE_NEED_QUERY_PHRASES[branch_need])
        query_specs.append(("branch_or_confounder", branch_terms, branch_patient, []))

    if counter and (ecg or clinical):
        counter_patient = (ecg or clinical)[:1]
        counter_terms = counter_patient + ["opposing evidence", "alternative explanation", "uncertainty"]
        query_specs.append(("counterevidence", counter_terms, counter_patient, []))

    if source_cohort_factor_ids:
        cohort_terms = ["ECG clinical evidence review", "cohort factor context"]
        if needs:
            cohort_terms.append(KNOWLEDGE_NEED_QUERY_PHRASES[needs[0]])
        query_specs.append(("cohort_context", cohort_terms, [], source_cohort_factor_ids))

    queries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, (purpose, terms, patient_concepts, cohort_factors) in enumerate(query_specs, start=1):
        clean = list(dict.fromkeys(str(term).strip() for term in terms if str(term).strip()))
        if not clean:
            continue
        text = " ".join(term.replace("_", " ") for term in clean)
        if text in seen:
            continue
        seen.add(text)
        patient_sources = [
            {
                "query_term": concept,
                "source_type": (
                    "patient_ecg_evidence" if concept in ecg
                    else "patient_clinical_evidence"
                ),
                "source_evidence_ids": sorted(set(evidence_by_concept.get(concept, []))),
                "source_cohort_factor_ids": [],
                "query_purpose": purpose,
            }
            for concept in patient_concepts
        ]
        framework_sources = [
            {
                "query_term": term,
                "source_type": (
                    "counterevidence_need" if purpose == "counterevidence"
                    else "branch_mechanism"
                ),
                "source_evidence_ids": [],
                "source_cohort_factor_ids": [],
                "query_purpose": purpose,
            }
            for term in clean if term not in patient_concepts
        ]
        cohort_sources = [
            {
                "query_term": factor_id,
                "source_type": "cohort_factor_context",
                "source_evidence_ids": [],
                "source_cohort_factor_ids": [factor_id],
                "query_purpose": purpose,
                "patient_match_status": "not_evaluated",
            }
            for factor_id in cohort_factors
        ]
        consumed = sorted(set(
            evidence_id for concept in patient_concepts
            for evidence_id in evidence_by_concept.get(concept, [])
        ))
        queries.append({
            "query_id": f"round_{round_index}_query_{index}",
            "query_purpose": purpose,
            "query_terms": clean,
            "query_text": text,
            "source_patient_evidence_ids": consumed,
            "query_term_sources": patient_sources + framework_sources + cohort_sources,
            "consumed_patient_evidence_ids": consumed,
            "source_cohort_factor_ids": cohort_factors,
            "forbidden_inference_check": {
                "score_generated_medical_fact_count": 0,
                "missing_treated_as_present": False,
                "continuous_without_state_rule_added": False,
                "pass": True,
            },
            "query_hash": stable_hash({"round": round_index, "purpose": purpose, "terms": clean}),
        })
        if len(queries) == 3:
            break
    return queries


def rerank_cards_with_metadata_v2_6(
    cards: list[dict[str, Any]],
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    branch = str(plan.get("selected_branch", "global_cag"))
    patient_factors = patient_present_concepts_v2_6(plan)
    ecg_concepts = set(map(str, plan.get("patient_ecg_concepts", [])))
    needs = set(map(str, plan.get("knowledge_need_types", [])))
    prior_rerank = set(map(str, plan.get("cohort_prior_rerank_factor_ids", [])))
    linked_cohort_cards = set(map(str, plan.get("linked_cohort_external_card_ids", [])))
    ranked: list[dict[str, Any]] = []
    for card in cards:
        text = " ".join(str(card.get(key, "")) for key in (
            "card_id", "card_group", "card_title", "title", "category",
            "key_evidence_summary", "content", "applicable_when",
            "usage_boundary", "keywords",
        )).lower()
        components = {
            "branch_scope_match": 1 if branch.replace("_like", "").replace("_", " ") in text or branch == "global_cag" else 0,
            "patient_factor_match": sum(1 for factor in patient_factors if factor.lower() in text),
            "ecg_concept_match": sum(1 for concept in ecg_concepts if concept.lower() in text),
            "knowledge_need_match": sum(1 for need in needs if need.replace("_", " ") in text or any(part in text for part in need.split("_"))),
            "reviewer_status": 1 if str(card.get("reviewer_status", "")).lower() == "ai_reviewed_pending_clinician" else 0,
            "source_quality": 1 if card.get("source_excerpt_ids") or card.get("evidence_sources") else 0,
            "usage_boundary_match": 1 if card.get("usage_boundary") else 0,
            "cohort_prior_rerank_match": sum(1 for factor in prior_rerank if factor.lower() in text),
            "cohort_factor_card_link_match": 1 if str(card.get("card_id", "")) in linked_cohort_cards else 0,
        }
        score = (
            2 * components["branch_scope_match"]
            + 3 * components["patient_factor_match"]
            + 3 * components["ecg_concept_match"]
            + 2 * components["knowledge_need_match"]
            + components["reviewer_status"]
            + 2 * components["source_quality"]
            + components["usage_boundary_match"]
            + 2 * components["cohort_prior_rerank_match"]
            + 4 * components["cohort_factor_card_link_match"]
        )
        ranked.append({**card, "metadata_rerank_score": score, "metadata_rerank_components": components})
    ranked.sort(key=lambda item: (-int(item["metadata_rerank_score"]), str(item.get("card_id", ""))))
    return ranked


def evaluate_card_applicability_gate_v2_6(
    card: dict[str, Any],
    plan: dict[str, Any],
    query: dict[str, Any],
) -> dict[str, Any]:
    """Fail-closed patient applicability gate applied before medical Top-K."""
    card_id = str(card.get("card_id", ""))
    branch = str(plan.get("selected_branch", "global_cag"))
    purpose = str(query.get("query_purpose", "core_mechanism"))
    present_concepts = patient_present_concepts_v2_6(plan)
    status_by_concept = patient_concept_statuses_v2_6(plan)
    reasons: list[str] = []
    if str(card.get("runtime_pool", "patient_medical")) != "patient_medical":
        reasons.append("governance_or_policy_card_excluded")
    runtime_disposition = str(card.get("runtime_disposition", "KEEP_IN_RUNTIME"))
    if runtime_disposition in {
        "EXCLUDE_FROM_RUNTIME", "EXCLUDE_AS_STANDALONE_CARD",
        "MOVE_TO_DETERMINISTIC_CONTROLLER",
    }:
        reasons.append(f"runtime_disposition_excluded:{runtime_disposition}")
    reviewer_status = str(card.get("reviewer_status", "pending_review"))
    if reviewer_status == "excluded_from_runtime":
        reasons.append("reviewer_status_excluded_from_runtime")
    if (
        str(card.get("runtime_pool", "patient_medical")) == "patient_medical"
        and runtime_disposition == "KEEP_IN_RUNTIME"
        and reviewer_status not in {
            "ai_reviewed_pending_clinician",
            "doctor_review_informed_v2_candidate",
        }
    ):
        reasons.append(f"reviewer_status_not_closed:{reviewer_status}")
    applicable_branches = set(map(str, card.get("applicable_branches", [])))
    neutral_gate_active = str(plan.get("retrieval_decision", "SKIP")) in {"SKIP", "RETRIEVE"}
    if applicable_branches and branch not in applicable_branches and not neutral_gate_active:
        reasons.append("branch_not_applicable")
    allowed_purposes = set(map(str, card.get("allowed_query_purposes", [])))
    if allowed_purposes and purpose not in allowed_purposes:
        reasons.append("query_purpose_not_allowed")
    linked_cohort_card = (
        purpose == "cohort_context"
        and card_id in set(map(str, plan.get("linked_cohort_external_card_ids", [])))
        and bool(query.get("source_cohort_factor_ids"))
    )
    required_any = set(map(str, card.get("required_any_present_concepts", [])))
    if required_any and not linked_cohort_card and not (required_any & present_concepts):
        reasons.append("required_present_evidence_missing")
    required_all = set(map(str, card.get("required_all_present_concepts", [])))
    if required_all and not linked_cohort_card and not required_all.issubset(present_concepts):
        reasons.append("required_all_present_evidence_missing")
    for concept in map(str, card.get("forbidden_evidence_concepts", [])):
        forbidden_statuses = set(map(str, card.get("forbidden_status", ["absent", "missing", "unknown"])))
        if status_by_concept.get(concept) in forbidden_statuses:
            reasons.append(f"forbidden_status:{concept}:{status_by_concept[concept]}")
    if (
        str(card.get("runtime_pool", "patient_medical")) == "patient_medical"
        and not required_any
        and not required_all
        and not linked_cohort_card
    ):
        card_text = " ".join(str(card.get(key, "")) for key in (
            "card_id", "title", "category", "content", "keywords", "usage_boundary",
        )).lower()
        patient_match = any(
            concept.lower() in card_text
            or any(len(token) > 3 and token in card_text for token in concept.lower().split("_"))
            for concept in present_concepts
        )
        need_match = any(
            any(len(token) > 4 and token in card_text for token in str(need).lower().split("_"))
            for need in plan.get("knowledge_need_types", [])
        )
        if not patient_match and not need_match:
            reasons.append("no_patient_or_knowledge_need_match")
    passed = not reasons
    return {
        "card_id": card_id,
        "applicability_gate_pass": passed,
        "applicability_gate_reasons": reasons,
        "query_purpose": purpose,
        "selected_branch": branch,
        "required_evidence_concepts": sorted(required_any | required_all),
        "reviewer_status": card.get("reviewer_status", "pending_review"),
        "usage_boundary_present": bool(str(card.get("usage_boundary", "")).strip()),
    }


def assess_rag_card_applicability_v2_6(
    card: dict[str, Any],
    *,
    patient_factor_ids: set[str],
    ecg_concept_ids: set[str],
    cohort_prior_factor_ids: set[str],
) -> dict[str, Any]:
    text = " ".join(str(card.get(key, "")) for key in (
        "card_id", "card_group", "card_title", "key_evidence_summary",
        "agent_action_hint", "blocked_action", "usage_boundary",
    )).lower()
    patient_match = any(factor.lower() in text for factor in patient_factor_ids)
    ecg_match = any(concept.lower() in text for concept in ecg_concept_ids)
    cohort_match = any(factor.lower() in text for factor in cohort_prior_factor_ids)
    externally_supported = bool(card.get("source_excerpt_ids") or card.get("confidence_level"))
    applicable = bool((patient_match or ecg_match) and externally_supported)
    return {
        "card_id": str(card.get("card_id", "")),
        "retrieved": True,
        "patient_factor_matched": patient_match,
        "ecg_pattern_matched": ecg_match,
        "cohort_prior_supported": cohort_match,
        "externally_supported": externally_supported,
        "case_applicable": applicable,
        "cited": False,
        "decision_support": applicable and str(card.get("blocked_action", "")).lower() != "veto",
        "decision_veto": applicable and str(card.get("blocked_action", "")).lower() == "veto",
    }


def build_rag_reasoning_payload(
    context: dict[str, Any],
    agent1_output: dict[str, Any],
    agent2_output: dict[str, Any],
    retrieved_cards: list[dict[str, Any]],
    triage: dict[str, Any],
) -> dict[str, Any]:
    """Build the only Agent3 LLM input: compact evidence plus compact RAG cards.

    Query construction, gate decisions, score handling, cohort context and
    per-round trace records remain local Runtime data.  They are not repeated
    to the Agent3 LLM.
    """
    cards = [
        {
            "card_id": str(card.get("card_id", "")),
            "key_evidence_summary": str(card.get("key_evidence_summary", card.get("content", "")))[:500],
            "usage_boundary": str(card.get("usage_boundary", ""))[:260],
            "source_excerpt_reference": list(card.get("source_excerpt_ids", []))[:3],
            "safety_level": str(card.get("safety_level", card.get("confidence_level", ""))),
        }
        for card in retrieved_cards if isinstance(card, dict)
    ][:5]
    payload = {
        "case_id": str(context.get("case_id", "")),
        "ecg_baseline_state": build_anchor_packet_for_agent3(context).get("ecg_baseline_state"),
        "selected_discrepancy_head": agent2_output.get("selected_discrepancy_head"),
        "agent1_compact_evidence": build_agent1_packet_for_agent3(agent1_output),
        "agent2_compact_evidence": build_agent2_packet_for_agent3(agent2_output),
        "retrieved_cards": cards,
        "rag_boundary": {
            "retrieval_is_optional": True,
            "cards_are_external_context_not_patient_facts": True,
            "only_applicable_cards_may_be_supporting_or_veto_cards": True,
            "do_not_output_a_correction_action_or_numeric_score": True,
        },
    }
    forbidden = find_forbidden_v2_6_fields(payload)
    if forbidden:
        raise ValueError(f"forbidden Agent3 payload fields: {forbidden}")
    return payload


class RAGDiscordanceMechanismAgent(LLMAgentBase):
    """Real v2.6 LLM Agent3: RAG-aware proposal, never final authority."""

    agent_id = "agent3"
    agent_name = "Agent3_CrossModal_Discrepancy_Arbitration_RAG_Agent"
    prompt_key = "agent3"
    schema_key = "agent3"

    def build_payload(self, context: dict[str, Any]) -> dict[str, Any]:
        return build_rag_reasoning_payload(
            context,
            context["agent1_output"],
            context["agent2_output"],
            list(context.get("retrieved_cards", [])),
            dict(context.get("triage", {})),
        )

    def run_with_trace(self, context: dict[str, Any]):
        result = super().run_with_trace(context)
        enriched = enrich_agent3_output_with_rag_aware_fields_v2_7(
            result.output,
            agent1=context.get("agent1_output", {}),
            agent2=context.get("agent2_output", {}),
            ecg_anchor_context=context.get("ecg_anchor_context", {}),
            retrieved_cards=list(context.get("retrieved_cards", [])),
            canonical_relation_v3_core_enabled=False,
            canonical_relation_v3_1_enabled=True,
        )
        plan = dict(context.get("local_retrieval_plan", {}))
        retrieved_ids = [
            str(card.get("card_id", ""))
            for card in context.get("retrieved_cards", [])
            if isinstance(card, dict) and card.get("card_id")
        ]
        enriched.update({
            "schema_version": V2_6_SCHEMA_VERSION,
            "agent_name": self.agent_name,
            "case_id": str(context.get("case_id", "")),
            "safety_boundary_check": V2_6_SAFE_BOUNDARY_CHECK,
            "global_consistency_status": str(context.get("triage", {}).get("global_consistency_status", "INSUFFICIENT_EVIDENCE")),
            "retrieval_decision": str(plan.get("retrieval_decision", "SKIP")),
            "retrieval_reason": str(plan.get("retrieval_reason", "NONE")),
            "retrieval_executed": bool(context.get("retrieved_cards")),
            "retrieved_card_ids": retrieved_ids,
            "retrieval_round_index": int(context.get("retrieval_round_index", 0)),
            "selected_discrepancy_head": context.get("agent2_output", {}).get("selected_discrepancy_head", ""),
            "canonical_relation_source": "DETERMINISTIC_AGENT3_EVIDENCE_ADJUDICATION",
        })
        validate_v2_6_agent_output("agent3", enriched)
        trace = dict(result.trace)
        trace["parsed_output"] = enriched
        trace["normalized_json"] = enriched
        trace["rag_aware_not_dependent_revision"] = {
            "patient_evidence_direction": enriched.get("patient_evidence_direction"),
            "patient_evidence_strength": enriched.get("patient_evidence_strength"),
            "rag_evidence_status": enriched.get("rag_evidence_status"),
            "rag_should_affect_agent4_policy": enriched.get("rag_should_affect_agent4_policy"),
        }
        self.last_trace = trace
        return type(result)(output=enriched, trace=trace)
