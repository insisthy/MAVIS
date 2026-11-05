"""v2.6 Agent schema contracts and runtime validators."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

V2_6_SCHEMA_VERSION = "ecg_correction_v1"
V2_6_SAFE_BOUNDARY_CHECK = {
    "no_target_label_access": True,
    "no_cag_prediction": True,
    "no_final_probability_output": True,
    "no_numeric_score_override": True,
    "no_treatment_advice": True,
    "rag_not_used_as_patient_fact": True,
    "unknown_not_treated_as_absent": True,
    "status": "pass",
}
V2_6_COMMON = {"schema_version", "agent_name", "case_id", "safety_boundary_check"}
V2_6_AGENT_SCHEMAS = {
    "agent1": {
        "agent_name": "Agent1_ECG_Baseline_Audit_Agent",
        "required": V2_6_COMMON | {
            "ecg_evidence_pattern", "ecg_directional_strength",
            "ecg_evidence_informativeness", "ecg_signal_quality",
            "ecg_uncertainty", "ischemic_evidence_ids",
            "non_specific_or_confounding_evidence_ids",
            "hidden_ischemia_warning", "false_positive_mimic_warning",
            "retrieval_concepts", "blind_reasoning_summary",
            "ecg_self_score_consistency",
            "ecg_local_concern", "ecg_audit_confidence",
            "ecg_model_explanation_consistency", "ecg_audit_reason_code",
            "evidence_pass_hash", "score_consistency_pass_hash",
        },
        "enums": {
            "schema_version": [V2_6_SCHEMA_VERSION],
            "ecg_evidence_pattern": [
                "ISCHEMIA_SUPPORTIVE", "NO_CLEAR_ISCHEMIC_EVIDENCE",
                "NON_SPECIFIC_OR_CONFOUNDED", "MIXED", "NOT_EVALUABLE",
            ],
            "ecg_directional_strength": ["NONE", "WEAK", "MODERATE", "STRONG"],
            "ecg_evidence_informativeness": ["LOW", "MEDIUM", "HIGH"],
            "ecg_signal_quality": ["LOW", "MEDIUM", "HIGH"],
            "ecg_uncertainty": ["LOW", "MEDIUM", "HIGH"],
            "ecg_self_score_consistency": [
                "LOCAL_CONCORDANT", "LOCAL_SCORE_POSSIBLY_OVERSTATED",
                "LOCAL_SCORE_POSSIBLY_UNDERSTATED", "LOCAL_MIXED",
                "LOCAL_NOT_EVALUABLE",
            ],
            "ecg_local_concern": [
                "NONE", "POSSIBLE_UNDERESTIMATION",
                "POSSIBLE_OVERESTIMATION", "UNRESOLVED",
            ],
            "ecg_audit_confidence": ["LOW", "MEDIUM", "HIGH"],
            "ecg_model_explanation_consistency": [
                "CONSISTENT", "PARTIALLY_CONSISTENT", "INCONSISTENT", "UNAVAILABLE",
            ],
        },
        "optional": set(),
        "prompt_excluded": set(),
        "booleans": [
            "hidden_ischemia_warning", "false_positive_mimic_warning",
        ],
        "lists": [
            "ischemic_evidence_ids", "non_specific_or_confounding_evidence_ids",
            "retrieval_concepts",
        ],
    },
    "agent2": {
        "agent_name": "Agent2_Clinical_Discrepancy_Agent",
        "required": {
            "schema_version",
            "selected_discrepancy_head", "head_score", "risk_band", "alert_state",
            "evidence_stance", "supporting_factor_ids", "contradicting_factor_ids",
            "confounder_factor_ids", "critical_missing_factor_ids",
            "evidence_sufficiency", "concise_rationale", "patient_fact_loss_count",
            "provenance_status",
        },
        "enums": {
            "schema_version": [V2_6_SCHEMA_VERSION],
            "selected_discrepancy_head": ["UNDERESTIMATION", "OVERESTIMATION"],
            "risk_band": ["LOW", "INTERMEDIATE", "HIGH"],
            "alert_state": ["ALERT", "NO_ALERT"],
            "evidence_stance": ["SUPPORTS_DISCREPANCY", "OPPOSES_DISCREPANCY", "MIXED", "INSUFFICIENT"],
            "evidence_sufficiency": ["SUFFICIENT", "LIMITED", "INSUFFICIENT"],
            "provenance_status": ["LOCKED_MODEL_AND_VALIDATED_LLM"],
        },
        "numbers": ["head_score", "patient_fact_loss_count"],
        "lists": [
            "supporting_factor_ids", "contradicting_factor_ids",
            "confounder_factor_ids", "critical_missing_factor_ids",
        ],
    },
    "agent3": {
        "agent_name": "Agent3_CrossModal_Discrepancy_Arbitration_RAG_Agent",
        "required": V2_6_COMMON | {
            "evidence_alignment", "evidence_sufficiency", "overall_confidence",
            "supporting_evidence_ids", "contradicting_evidence_ids",
            "remaining_uncertainty", "applicable_card_ids",
            "supporting_card_ids", "veto_card_ids",
            "brief_arbitration_rationale",
        },
        "optional": {
            "agent3_stageA_executed", "agent3_stageB_executed",
            "retrieved_card_ids", "retrieval_round_index",
            "global_consistency_status", "retrieval_decision", "retrieval_reason",
            "rag_trigger", "rag_used", "retrieval_executed",
            "retrieval_query_count", "retrieved_card_count",
            "agent3b_skipped_reason", "cohort_prior_used",
            "cohort_prior_rerank_used", "cohort_prior_context_used",
            "cohort_prior_query_expansion_used",
            "cohort_prior_enters_supporting_cards",
            "cohort_prior_enters_applicable_card_ids",
            "agent3_uncertainty_reasons",
            "patient_evidence_direction", "patient_evidence_strength",
            "patient_evidence_basis", "rag_evidence_status",
            "rag_evidence_strength", "rag_should_affect_agent4_policy",
            "absolute_clinical_risk_direction",
            "correction_relation_relative_to_ecg",
            "correction_evidence_grade",
            "correction_confidence_relative_to_ecg",
            "directional_signal_present",
            "correction_relation_basis",
            "mechanism_context_card_ids", "safety_warning_card_ids",
            "correction_supporting_card_ids", "review_only_card_ids",
            "contradictory_card_ids", "card_support_direction",
            "why_card_is_not_supporting",
            "deterministic_rag_gate_override",
            "deterministic_rag_gate_override_reason",
            "quantitative_support_status",
            "semantic_support_status",
            "ecg_support_status",
            "safety_block_status",
            "canonical_relation_source",
            "relation_adjudication_trace",
            "relation_equality_check_pass",
            "ecg_only_cag_prediction_class",
            "selected_discrepancy_head",
            "risk_band",
            "risk_alert",
            "overwrite_attempt_count",
        },
        "enums": {
            "schema_version": [V2_6_SCHEMA_VERSION],
            "global_consistency_status": [
                "CONSISTENT_CONFIDENT", "CONSISTENT_UNCERTAIN",
                "ECG_HIGH_CLINICAL_LOW", "ECG_LOW_CLINICAL_HIGH",
                "MODAL_EVIDENCE_SCORE_CONFLICT", "COMPLEX_DISCORDANCE",
                "INSUFFICIENT_EVIDENCE",
            ],
            "proposal_direction": ["PRESERVE", "REVIEW_ONLY"],
            "proposal_strength": ["NONE", "WEAK", "MODERATE", "STRONG"],
            "proposal_confidence": ["LOW", "MEDIUM", "HIGH"],
            "rag_mode": ["STANDARD_RAG"],
            "patient_evidence_direction": ["UP", "DOWN", "PRESERVE", "UNCERTAIN"],
            "patient_evidence_strength": ["NONE", "WEAK", "MODERATE", "STRONG"],
            "patient_evidence_basis": [
                "ECG_DOMINANT", "CLINICAL_DOMINANT", "CONCORDANT_MULTIMODAL",
                "CONFLICTING_MULTIMODAL", "SCORE_MARGIN_ONLY", "INSUFFICIENT",
            ],
            "rag_evidence_status": [
                "RAG_UNAVAILABLE", "RAG_CONTEXT_ONLY", "RAG_MECHANISM_SUPPORT",
                "RAG_CORRECTION_SUPPORT", "RAG_SAFETY_WARNING", "RAG_CONTRADICTORY",
            ],
            "rag_evidence_strength": ["NONE", "CONTEXT_ONLY", "WEAK_SUPPORT", "MODERATE_SUPPORT"],
            "absolute_clinical_risk_direction": [
                "HIGHER_CAG_RISK_CONTEXT",
                "LOWER_CAG_RISK_CONTEXT",
                "NEUTRAL_CAG_RISK_CONTEXT",
                "INDETERMINATE_CAG_RISK_CONTEXT",
            ],
            "correction_relation_relative_to_ecg": [
                "ECG_BASELINE_SUPPORTED",
                "ECG_BASELINE_POSSIBLY_UNDERSTATED",
                "ECG_BASELINE_POSSIBLY_OVERSTATED",
                "ECG_BASELINE_RELATION_UNRESOLVED",
            ],
            "correction_evidence_grade": ["NONE", "WEAK", "MODERATE", "STRONG"],
            "correction_confidence_relative_to_ecg": ["LOW", "MEDIUM", "HIGH"],
            "evidence_sufficiency": ["SUFFICIENT", "PARTIAL", "INSUFFICIENT"],
            "modality_alignment": ["ALIGNED", "DISAGREEMENT", "UNCERTAIN"],
            "overall_confidence": ["HIGH", "MODERATE", "LOW"],
            "retrieval_decision": ["SKIP", "RETRIEVE"],
            "retrieval_reason": [
                "NONE", "INSUFFICIENT_EVIDENCE", "LOW_CONFIDENCE",
                "MODALITY_DISAGREEMENT", "COMBINED_UNCERTAINTY",
                "ECG_ANCHOR_DISCREPANCY_ALERT",
            ],
            "ecg_baseline_relation": [
                "ECG_BASELINE_SUPPORTED",
                "ECG_BASELINE_POSSIBLY_UNDERSTATED",
                "ECG_BASELINE_POSSIBLY_OVERSTATED",
                "ECG_BASELINE_RELATION_UNRESOLVED",
            ],
        },
        "numbers": [],
        "optional_numbers": ["retrieval_query_count", "retrieved_card_count", "overwrite_attempt_count"],
        "optional_booleans": [
            "agent3_stageA_executed", "agent3_stageB_executed",
            "rag_used", "retrieval_executed", "cohort_prior_used",
            "cohort_prior_rerank_used", "cohort_prior_context_used",
            "cohort_prior_query_expansion_used",
            "cohort_prior_enters_supporting_cards",
            "cohort_prior_enters_applicable_card_ids",
            "rag_should_affect_agent4_policy",
            "directional_signal_present",
            "deterministic_rag_gate_override",
            "relation_equality_check_pass",
        ],
        "optional_lists": [
            "agent3_uncertainty_reasons", "mechanism_context_card_ids",
            "safety_warning_card_ids", "correction_supporting_card_ids",
            "review_only_card_ids", "contradictory_card_ids",
            "card_support_direction", "why_card_is_not_supporting",
            "deterministic_rag_gate_override_reason",
            "relation_adjudication_trace",
        ],
        "lists": [
            "supporting_evidence_ids", "contradicting_evidence_ids",
            "remaining_uncertainty", "applicable_card_ids",
            "supporting_card_ids", "veto_card_ids",
        ],
    },
    "agent4": {
        "agent_name": "Agent4_Safety_Constrained_ECG_Correction_Policy_Agent",
        "required": V2_6_COMMON | {
            "agent4_action_raw",
            "correction_confidence", "evidence_sufficiency",
            "safety_constraint", "hidden_risk_flag",
            "rag_misuse_flag", "medical_overclaim_flag",
            "supporting_evidence_ids", "contradicting_evidence_ids",
            "final_policy_rationale",
        },
        "optional": {
            # Controller-owned fields are intentionally absent from Agent4 Raw.
            "absolute_clinical_risk_direction",
            "correction_relation_relative_to_ecg",
            "correction_evidence_grade",
            "action_confidence",
            "directional_signal_present",
            "supporting_evidence_groups",
            "contradicting_evidence_groups",
            "critical_safety_conflict",
        },
        "prompt_excluded": {
            "unsafe_downward_correction_flag", "reflection_pass",
            # Directional relation, grade and safety fields below are
            # deterministic/committed views and must not be redundantly
            # regenerated by the LLM.
            "absolute_clinical_risk_direction",
            "correction_relation_relative_to_ecg",
            "correction_evidence_grade",
            "action_confidence", "directional_signal_present",
            "supporting_evidence_groups", "contradicting_evidence_groups",
            "critical_safety_conflict",
        },
        "enums": {
            "schema_version": [V2_6_SCHEMA_VERSION],
            "agent4_action_raw": ["PRESERVE", "UP", "DOWN", "REVIEW_REQUIRED"],
            "correction_confidence": ["LOW", "MODERATE", "HIGH"],
            "action_confidence": ["LOW", "MEDIUM", "HIGH"],
            "absolute_clinical_risk_direction": [
                "HIGHER_CAG_RISK_CONTEXT",
                "LOWER_CAG_RISK_CONTEXT",
                "NEUTRAL_CAG_RISK_CONTEXT",
                "INDETERMINATE_CAG_RISK_CONTEXT",
            ],
            "correction_relation_relative_to_ecg": [
                "ECG_BASELINE_SUPPORTED",
                "ECG_BASELINE_POSSIBLY_UNDERSTATED",
                "ECG_BASELINE_POSSIBLY_OVERSTATED",
                "ECG_BASELINE_RELATION_UNRESOLVED",
            ],
            "correction_evidence_grade": ["NONE", "WEAK", "MODERATE", "STRONG"],
            "evidence_sufficiency": ["SUFFICIENT", "PARTIAL", "INSUFFICIENT"],
            "safety_constraint": ["BOUNDED_ALLOWED", "NO_DOWNWARD_CORRECTION", "PRESERVE_ONLY"],
        },
        "booleans": [
            "hidden_risk_flag", "rag_misuse_flag", "medical_overclaim_flag",
        ],
        "optional_booleans": [
            "directional_signal_present", "critical_safety_conflict",
        ],
        "optional_numbers": [
            "repair_attempt_count",
        ],
        "lists": [
            "supporting_evidence_ids",
            "contradicting_evidence_ids",
        ],
        "optional_lists": [
            "supporting_evidence_groups", "contradicting_evidence_groups",
            "critical_safety_conflict_sources", "raw_duplicate_directional_fields",
        ],
    },
}

AGENT1_EVIDENCE_PASS_FIELDS = V2_6_COMMON | {
    "ecg_evidence_pattern", "ecg_directional_strength",
    "ecg_evidence_informativeness", "ecg_signal_quality",
    "ecg_uncertainty", "ischemic_evidence_ids",
    "non_specific_or_confounding_evidence_ids",
    "hidden_ischemia_warning", "false_positive_mimic_warning",
    "retrieval_concepts", "blind_reasoning_summary",
}
AGENT2_EVIDENCE_PASS_FIELDS = {
    "evidence_stance", "supporting_factor_ids", "contradicting_factor_ids",
    "confounder_factor_ids", "critical_missing_factor_ids",
    "evidence_sufficiency", "concise_rationale",
}

V2_6_AGENT_STAGE_SCHEMAS = {
    "agent1.evidence_pass": {
        "agent_id": "agent1",
        "stage_name": "Agent1-A ECG Evidence Pass",
        "agent_name": V2_6_AGENT_SCHEMAS["agent1"]["agent_name"],
        "llm_allowed": True,
        "required": AGENT1_EVIDENCE_PASS_FIELDS,
        "enums": {
            field: values
            for field, values in V2_6_AGENT_SCHEMAS["agent1"]["enums"].items()
            if field in AGENT1_EVIDENCE_PASS_FIELDS
        },
        "booleans": [
            field for field in V2_6_AGENT_SCHEMAS["agent1"].get("booleans", [])
            if field in AGENT1_EVIDENCE_PASS_FIELDS
        ],
        "lists": [
            field for field in V2_6_AGENT_SCHEMAS["agent1"].get("lists", [])
            if field in AGENT1_EVIDENCE_PASS_FIELDS
        ],
        "numbers": [],
        "deterministic_only_fields": sorted(set(V2_6_AGENT_SCHEMAS["agent1"]["required"]) - AGENT1_EVIDENCE_PASS_FIELDS),
        "prompt_schema_source": "agent1.evidence_pass",
        "repair_schema_source": "agent1.evidence_pass",
        "validator_schema_source": "agent1.evidence_pass",
    },
    "agent1.committed_output": {
        "agent_id": "agent1",
        "stage_name": "Agent1 committed output",
        "agent_name": V2_6_AGENT_SCHEMAS["agent1"]["agent_name"],
        "llm_allowed": False,
        **V2_6_AGENT_SCHEMAS["agent1"],
        "deterministic_only_fields": sorted(set(V2_6_AGENT_SCHEMAS["agent1"]["required"]) - AGENT1_EVIDENCE_PASS_FIELDS),
        "prompt_schema_source": "",
        "repair_schema_source": "",
        "validator_schema_source": "agent1.committed_output",
    },
    "agent2.evidence_pass": {
        "agent_id": "agent2",
        "stage_name": "Agent2-A Clinical Evidence Pass",
        "agent_name": V2_6_AGENT_SCHEMAS["agent2"]["agent_name"],
        "llm_allowed": True,
        "required": AGENT2_EVIDENCE_PASS_FIELDS,
        "enums": {field: values for field, values in V2_6_AGENT_SCHEMAS["agent2"]["enums"].items() if field in AGENT2_EVIDENCE_PASS_FIELDS},
        "booleans": [],
        "lists": [
            field for field in V2_6_AGENT_SCHEMAS["agent2"].get("lists", [])
            if field in AGENT2_EVIDENCE_PASS_FIELDS
        ],
        "numbers": [],
        "deterministic_only_fields": sorted(set(V2_6_AGENT_SCHEMAS["agent2"]["required"]) - AGENT2_EVIDENCE_PASS_FIELDS),
        "prompt_schema_source": "agent2.evidence_pass",
        "repair_schema_source": "agent2.evidence_pass",
        "validator_schema_source": "agent2.evidence_pass",
    },
    "agent2.committed_output": {
        "agent_id": "agent2",
        "stage_name": "Agent2 committed output",
        "agent_name": V2_6_AGENT_SCHEMAS["agent2"]["agent_name"],
        "llm_allowed": False,
        **V2_6_AGENT_SCHEMAS["agent2"],
        "deterministic_only_fields": sorted(set(V2_6_AGENT_SCHEMAS["agent2"]["required"]) - AGENT2_EVIDENCE_PASS_FIELDS),
        "prompt_schema_source": "",
        "repair_schema_source": "",
        "validator_schema_source": "agent2.committed_output",
    },
    "agent3.rag_reasoning": {
        "agent_id": "agent3", "stage_name": "Agent3 RAG reasoning", "llm_allowed": True,
        **V2_6_AGENT_SCHEMAS["agent3"],
        "deterministic_only_fields": [],
        "prompt_schema_source": "agent3.rag_reasoning",
        "repair_schema_source": "agent3.rag_reasoning",
        "validator_schema_source": "agent3.rag_reasoning",
    },
    "agent4.safety_review": {
        "agent_id": "agent4", "stage_name": "Agent4 safety review", "llm_allowed": True,
        **V2_6_AGENT_SCHEMAS["agent4"],
        "deterministic_only_fields": [],
        "prompt_schema_source": "agent4.safety_review",
        "repair_schema_source": "agent4.safety_review",
        "validator_schema_source": "agent4.safety_review",
    },
}

# Retrieval construction is local Runtime work, not Agent3 medical-semantic
# output.  These fields are intentionally excluded from both the LLM Raw and
# the persisted Agent3 committed contracts.  They belong only to the separate
# per-round retrieval trace artifact.
AGENT3_RETRIEVAL_TRACE_ONLY_FIELDS = frozenset({
    "neutral_query", "focused_query", "expanded_query", "query_terms",
    "query_variants", "query_plan", "retrieval_plan", "retrieval_query",
    "retrieval_query_contract", "query_bundle", "multi_query_bundle",
    "initial_query", "current_query", "rag_query", "retrieval_metadata",
    "retrieval_trace", "round_trace", "query_hash", "top_k",
    "filter_metadata", "candidate_card_ids", "retrieval_scores",
    "candidate_card_count", "applicability_gate_rejections",
    "applicability_gate_rejection_count", "retrieval_source",
    "knowledge_base_path", "knowledge_base_hash",
    "runtime_active_kb_retrieval_used", "payload_retrieved_card_count",
    "metadata_rerank_consumed", "technical_retry_count",
})

# Agent3's committed contract is independently named even where its currently
# allowed semantic fields match the raw contract.  This makes the Raw →
# deterministic committed boundary explicit in provenance and prevents a
# retrieval-trace field from being silently accepted later.
V2_6_AGENT_STAGE_SCHEMAS["agent3.committed_output"] = {
    "agent_id": "agent3",
    "stage_name": "Agent3 deterministic committed output",
    "llm_allowed": False,
    **deepcopy(V2_6_AGENT_SCHEMAS["agent3"]),
    "deterministic_only_fields": [],
    "prompt_schema_source": "",
    "repair_schema_source": "",
    "validator_schema_source": "agent3.committed_output",
    "trace_only_fields": sorted(AGENT3_RETRIEVAL_TRACE_ONLY_FIELDS),
}


def build_agent_raw_schema_registry() -> dict[str, dict[str, Any]]:
    """Authoritative LLM-produced schemas, before deterministic enrichment."""
    stage_keys = {
        "agent1": "agent1.evidence_pass",
        "agent2": "agent2.evidence_pass",
        "agent3": "agent3.rag_reasoning",
        "agent4": "agent4.safety_review",
    }
    return {agent_id: deepcopy(V2_6_AGENT_STAGE_SCHEMAS[stage_key]) for agent_id, stage_key in stage_keys.items()}


def build_agent_committed_schema_registry() -> dict[str, dict[str, Any]]:
    """Authoritative post-validation schemas consumed by later runtime stages."""
    stage_keys = {
        "agent1": "agent1.committed_output",
        "agent2": "agent2.committed_output",
        "agent3": "agent3.committed_output",
        "agent4": "agent4.safety_review",
    }
    return {agent_id: deepcopy(V2_6_AGENT_STAGE_SCHEMAS[stage_key]) for agent_id, stage_key in stage_keys.items()}
FORBIDDEN_V2_6_FIELDS = {
    "Target_CAG_Binary", "CAG_label", "group_label", "test_label", "y_true",
    "ground_truth", "final_score", "final_corrected_score", "adjusted_score",
    "v2_6_adjusted_score", "score_adjustment_delta", "delta", "raw_delta",
    "constrained_delta", "mas_corrected_score",
    "formal_residual_model_ready", "numerical_execution_authorized",
    "CAG_prediction", "CAG_diagnosis", "treatment_advice",
    "A", "B", "C", "D", "ABCD",
}


def find_forbidden_v2_6_fields(value: Any) -> list[str]:
    found: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if str(key) in FORBIDDEN_V2_6_FIELDS:
                    found.add(str(key))
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return sorted(found)


def normalize_v2_6_safety_boundary_check(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized = deepcopy(data)
    boundary = normalized.get("safety_boundary_check")
    if not isinstance(boundary, dict) or set(boundary) != set(V2_6_SAFE_BOUNDARY_CHECK):
        raise ValueError("invalid v2.6 safety_boundary_check shape")
    canonical = {}
    for key, expected in V2_6_SAFE_BOUNDARY_CHECK.items():
        value = boundary[key]
        if isinstance(expected, bool):
            if value not in (True, "true", "True", 1):
                raise ValueError(f"unsafe boundary value: {key}")
            canonical[key] = True
        else:
            if str(value).lower() != "pass":
                raise ValueError("unsafe boundary status")
            canonical[key] = "pass"
    normalized["safety_boundary_check"] = canonical
    return normalized, {"safety_boundary_normalized": canonical != boundary}


def _as_str_set(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {str(value) for value in values}


def _continuous_factor_ids_from_statuses(data: dict[str, Any]) -> set[str]:
    """Patient factors with visible continuous values from Agent2 deterministic status rows."""
    ids: set[str] = set()
    for item in data.get("patient_factor_statuses", []):
        if not isinstance(item, dict):
            continue
        if str(item.get("patient_factor_status")) != "continuous_value_available":
            continue
        factor_id = str(item.get("patient_factor_id") or "").strip()
        if factor_id:
            ids.add(factor_id)
    return ids


ALLOWED_CLAIM_SUPPORT_STATUS = {
    "fully_supported", "partially_supported", "patient_only",
    "external_only", "conflicting", "unsupported",
}

V2_6_NESTED_SCHEMAS = {
    "agent3.claim_evidence_chain.item": {
        "required": [
            "claim_id",
            "claim_type",
            "claim_text",
            "patient_evidence_ids",
            "model_evidence_ids",
            "cohort_prior_factor_ids",
            "external_card_ids",
            "support_status",
            "counterevidence_card_ids",
            "uncertainty",
        ],
        "lists": [
            "patient_evidence_ids",
            "model_evidence_ids",
            "cohort_prior_factor_ids",
            "external_card_ids",
            "counterevidence_card_ids",
        ],
        "enums": {
            "support_status": sorted(ALLOWED_CLAIM_SUPPORT_STATUS),
        },
    },
}


def build_nested_schema_contract(nested_schema_id: str) -> str:
    schema = V2_6_NESTED_SCHEMAS[nested_schema_id]
    lines = [
        f"Nested schema: {nested_schema_id}",
        "claim_evidence_chain must be a non-empty list.",
        "Each item must contain exactly the required keys.",
        "The list fields must always be JSON arrays, even when empty.",
        "Do not output strings for list fields.",
        "Do not omit empty list fields.",
        "Do not add extra keys.",
        "Required item keys: " + " / ".join(schema["required"]),
        "List fields: " + " / ".join(schema["lists"]),
    ]
    for field, allowed in schema.get("enums", {}).items():
        lines.append(f"{field} allowed values: " + " / ".join(allowed))
    lines.extend([
        "Generic example:",
        '{"claim_evidence_chain":[{"claim_id":"claim_1","claim_type":"patient_correction","claim_text":"Brief claim supported by available patient evidence and applicable RAG cards.","patient_evidence_ids":[],"model_evidence_ids":[],"cohort_prior_factor_ids":[],"external_card_ids":[],"support_status":"partially_supported","counterevidence_card_ids":[],"uncertainty":"Brief uncertainty statement."}]}',
    ])
    return "\n".join(lines)


def collect_claim_evidence_chain_errors(chain: Any) -> list[dict[str, Any]]:
    schema = V2_6_NESTED_SCHEMAS["agent3.claim_evidence_chain.item"]
    required = set(schema["required"])
    list_fields = set(schema["lists"])
    allowed_status = set(schema["enums"]["support_status"])
    errors: list[dict[str, Any]] = []
    if not isinstance(chain, list):
        return [_error(
            "invalid_nested_type",
            "claim_evidence_chain",
            "Agent3 claim_evidence_chain must be a non-empty list",
            actual_type=type(chain).__name__,
        )]
    if not chain:
        return [_error(
            "empty_nested_list",
            "claim_evidence_chain",
            "Agent3 claim_evidence_chain must be a non-empty list",
        )]
    for index, claim in enumerate(chain):
        prefix = f"claim_evidence_chain[{index}]"
        if not isinstance(claim, dict):
            errors.append(_error(
                "invalid_nested_item_type",
                prefix,
                "Agent3 claim_evidence_chain item must be an object",
                actual_type=type(claim).__name__,
            ))
            continue
        missing = sorted(required - set(claim))
        extra = sorted(set(claim) - required)
        if missing:
            errors.append(_error(
                "missing_nested_field",
                prefix,
                "Agent3 claim_evidence_chain item missing required keys",
                missing_keys=missing,
            ))
        if extra:
            errors.append(_error(
                "extra_nested_field",
                prefix,
                "Agent3 claim_evidence_chain item has extra keys",
                extra_keys=extra,
            ))
        for field in sorted(list_fields):
            if field in claim and not isinstance(claim[field], list):
                errors.append(_error(
                    "invalid_nested_list_field",
                    f"{prefix}.{field}",
                    f"Agent3 claim {field} must be a list",
                    actual_type=type(claim[field]).__name__,
                ))
        if "support_status" in claim and str(claim["support_status"]) not in allowed_status:
            errors.append(_error(
                "invalid_nested_enum",
                f"{prefix}.support_status",
                "Agent3 invalid claim support_status",
                allowed=sorted(allowed_status),
                actual=claim.get("support_status"),
            ))
        if (
            claim.get("support_status") == "external_only"
            and str(claim.get("claim_type")) == "patient_correction"
        ):
            errors.append(_error(
                "nested_evidence_boundary",
                f"{prefix}.support_status",
                "external-only evidence cannot support a patient correction conclusion",
            ))
        if claim.get("cohort_prior_factor_ids") and not (
            claim.get("patient_evidence_ids") or claim.get("external_card_ids")
        ):
            errors.append(_error(
                "nested_evidence_boundary",
                f"{prefix}.cohort_prior_factor_ids",
                "cohort prior alone cannot support a patient conclusion",
            ))
    return errors


def validate_claim_evidence_chain(chain: Any) -> None:
    errors = collect_claim_evidence_chain_errors(chain)
    if errors:
        compact = "; ".join(
            f"{error['code']}:{error['field_path']}:{error['message']}"
            for error in errors
        )
        raise ValueError(compact)


def build_agent_enum_inventory(agent_id: str) -> dict[str, list[str]]:
    return deepcopy(V2_6_AGENT_SCHEMAS[agent_id].get("enums", {}))


def declared_schema_fields(schema: dict[str, Any]) -> set[str]:
    """Return every field explicitly declared by an agent schema.

    This is the single source used by full-agent validators and contract
    builders. A field declared in any typed bucket is legal for the full
    committed agent output; undeclared fields are still rejected.
    """
    return (
        set(schema.get("required", set()))
        | set(schema.get("optional", set()))
        | set(schema.get("enums", {}))
        | set(schema.get("booleans", []))
        | set(schema.get("optional_booleans", []))
        | set(schema.get("numbers", []))
        | set(schema.get("optional_numbers", []))
        | set(schema.get("lists", []))
        | set(schema.get("optional_lists", []))
    )


def build_agent_raw_prompt_schema(agent_id: str) -> dict[str, Any]:
    """Return the schema surface that the LLM is actually asked to emit.

    Full agent schemas also describe deterministic committed fields. Technical
    Raw repair must not re-expose those fields after the primary prompt has
    excluded them.
    """
    schema = V2_6_AGENT_SCHEMAS[agent_id]
    excluded = set(schema.get("prompt_excluded", set()))
    exposed = declared_schema_fields(schema) - excluded
    raw_schema: dict[str, Any] = {
        "agent_name": schema["agent_name"],
        "required": sorted(set(schema.get("required", set())) & exposed),
        "optional": sorted(
            (set(schema.get("optional", set())) & exposed)
            - set(schema.get("required", set()))
        ),
    }
    for bucket in (
        "booleans", "optional_booleans", "numbers", "optional_numbers",
        "lists", "optional_lists",
    ):
        raw_schema[bucket] = sorted(set(schema.get(bucket, [])) & exposed)
    raw_schema["enums"] = {
        field: list(values)
        for field, values in schema.get("enums", {}).items()
        if field in exposed
    }
    return raw_schema


# Agent3 Raw keeps the compact 14-field required arbitration contract. The
# only conditional field exposed to the LLM is contradictory_card_ids because
# classifying an external card as contradicting is semantic arbitration work.
# All other declared non-required fields remain legal in the locally enriched
# committed output, but are excluded from the Raw LLM prompt contract. Derive
# this set from the canonical declaration inventory so enum-only legacy fields
# cannot accidentally leak back into the prompt.
_agent3_schema = V2_6_AGENT_SCHEMAS["agent3"]
_agent3_schema["prompt_excluded"] = (
    declared_schema_fields(_agent3_schema)
    - set(_agent3_schema["required"])
    - {"contradictory_card_ids"}
)


def build_agent_field_type_inventory(agent_id: str) -> dict[str, str]:
    schema = V2_6_AGENT_SCHEMAS[agent_id]
    inventory: dict[str, str] = {
        field: "string_or_object"
        for field in declared_schema_fields(schema)
    }
    for field in schema.get("enums", {}):
        inventory[field] = "enum"
    for field in schema.get("booleans", []):
        inventory[field] = "boolean"
    for field in schema.get("optional_booleans", []):
        inventory[field] = "boolean"
    for field in schema.get("lists", []):
        inventory[field] = "list"
    for field in schema.get("optional_lists", []):
        inventory[field] = "list"
    for field in schema.get("numbers", []):
        inventory[field] = "number"
    for field in schema.get("optional_numbers", []):
        inventory[field] = "number"
    inventory["safety_boundary_check"] = "object"
    return inventory


def build_agent_stage_field_type_inventory(stage_schema_key: str) -> dict[str, str]:
    schema = V2_6_AGENT_STAGE_SCHEMAS[stage_schema_key]
    inventory: dict[str, str] = {field: "string_or_object" for field in schema["required"]}
    for field in schema.get("enums", {}):
        inventory[field] = "enum"
    for field in schema.get("booleans", []):
        inventory[field] = "boolean"
    for field in schema.get("lists", []):
        inventory[field] = "list"
    for field in schema.get("numbers", []):
        inventory[field] = "number"
    inventory["safety_boundary_check"] = "object"
    return inventory


def build_agent_stage_schema_contract(stage_schema_key: str) -> str:
    schema = V2_6_AGENT_STAGE_SCHEMAS[stage_schema_key]
    type_inventory = build_agent_stage_field_type_inventory(stage_schema_key)
    lines = [
        f"Internal runtime stage contract: {stage_schema_key}",
        "Do not output stage_schema_key; it is routing metadata and is not a JSON response field.",
        f"Agent name must be exactly: {schema['agent_name']}",
        f"schema_version must be exactly: {V2_6_SCHEMA_VERSION}",
        "Return exactly one JSON object with exactly these required fields:",
    ]
    contract_fields = (
        set(schema["required"]) | set(schema.get("optional", set()))
        | set(schema.get("enums", {}))
        | set(schema.get("booleans", [])) | set(schema.get("optional_booleans", []))
        | set(schema.get("lists", [])) | set(schema.get("optional_lists", []))
        | set(schema.get("numbers", [])) | set(schema.get("optional_numbers", []))
    )
    for field in sorted(contract_fields):
        field_type = type_inventory.get(field, "string_or_object")
        allowed = schema.get("enums", {}).get(field)
        if allowed:
            detail = "allowed = " + " / ".join(allowed)
        elif field_type == "boolean":
            detail = "type = boolean; use native JSON true or false only"
        elif field_type == "list":
            detail = "type = list/array"
        elif field_type == "number":
            detail = "type = number"
        elif field == "safety_boundary_check":
            detail = f"must equal {V2_6_SAFE_BOUNDARY_CHECK}"
        else:
            detail = "type = string/object as appropriate; do not add forbidden fields"
        lines.append(f"- {field}: {detail}")
    deterministic = list(schema.get("deterministic_only_fields", []))
    if deterministic:
        lines.append("Do not output these deterministic-only fields:")
        for field in deterministic:
            lines.append(f"- {field}")
    return "\n".join(lines)


def build_agent_stage_json_schema(stage_schema_key: str) -> dict[str, Any]:
    schema = V2_6_AGENT_STAGE_SCHEMAS[stage_schema_key]
    type_inventory = build_agent_stage_field_type_inventory(stage_schema_key)
    properties: dict[str, Any] = {}
    contract_fields = (
        set(schema["required"]) | set(schema.get("optional", set()))
        | set(schema.get("enums", {}))
        | set(schema.get("booleans", [])) | set(schema.get("optional_booleans", []))
        | set(schema.get("lists", [])) | set(schema.get("optional_lists", []))
        | set(schema.get("numbers", [])) | set(schema.get("optional_numbers", []))
    )
    for field in sorted(contract_fields):
        field_type = type_inventory.get(field, "string_or_object")
        if field in schema.get("enums", {}):
            prop = {"enum": list(schema["enums"][field])}
        elif field_type == "boolean":
            prop = {"type": "boolean"}
        elif field_type == "list":
            prop = {"type": "array"}
        elif field_type == "number":
            prop = {"type": "number"}
        elif field == "safety_boundary_check":
            prop = {"type": "object", "additionalProperties": False}
        else:
            prop = {}
        properties[field] = prop
    return {
        "stage_schema_key": stage_schema_key,
        "type": "object",
        "additionalProperties": False,
        "required": sorted(schema["required"]),
        "properties": properties,
    }


def collect_stage_schema_errors(stage_schema_key: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    schema = V2_6_AGENT_STAGE_SCHEMAS[stage_schema_key]
    errors: list[dict[str, Any]] = []
    if not isinstance(data, dict):
        return [_error("type", "$", "stage output must be an object")]
    required = set(schema["required"])
    for field in sorted(required - set(data)):
        errors.append(_error("missing_field", field, f"missing required field: {field}"))
    declared = declared_schema_fields(schema)
    for field in sorted(set(data) - declared):
        code = "stage_schema_violation"
        message = f"extra field not allowed in {stage_schema_key}: {field}"
        if field in set(schema.get("deterministic_only_fields", [])):
            message = f"FULL_AGENT_SCHEMA_RETURNED_IN_EVIDENCE_PASS: {field}"
        errors.append(_error(code, field, message))
    if "agent_name" in required and data.get("agent_name") != schema["agent_name"]:
        errors.append(_error("agent_name", "agent_name", "agent_name mismatch", expected=schema["agent_name"], actual=data.get("agent_name")))
    for field, allowed in schema.get("enums", {}).items():
        if field in data and data.get(field) not in allowed:
            errors.append(_error("invalid_enum", field, f"invalid enum {field}: {data.get(field)!r}", allowed=list(allowed), actual=data.get(field)))
    for field in schema.get("booleans", []):
        if field in data and not isinstance(data.get(field), bool):
            errors.append(_error("invalid_boolean", field, f"{field} must be boolean", actual_type=type(data.get(field)).__name__, actual=data.get(field)))
    for field in schema.get("numbers", []):
        if field in data and (isinstance(data.get(field), bool) or not isinstance(data.get(field), (int, float))):
            errors.append(_error("invalid_number", field, f"{field} must be numeric", actual_type=type(data.get(field)).__name__))
    for field in schema.get("lists", []):
        if field in data and not isinstance(data.get(field), list):
            errors.append(_error("invalid_list", field, f"{field} must be a list", actual_type=type(data.get(field)).__name__))
    if "safety_boundary_check" in required and data.get("safety_boundary_check") != V2_6_SAFE_BOUNDARY_CHECK:
        errors.append(_error("safety_boundary", "safety_boundary_check", "safety boundary did not pass"))
    return errors


def validate_agent_stage_output(stage_schema_key: str, data: dict[str, Any]) -> None:
    errors = collect_stage_schema_errors(stage_schema_key, data)
    if errors:
        compact = "; ".join(f"{e['code']}:{e['field_path']}:{e['message']}" for e in errors)
        raise ValueError(compact)


def build_agent_json_schema(agent_id: str) -> dict[str, Any]:
    schema = V2_6_AGENT_SCHEMAS[agent_id]
    type_inventory = build_agent_field_type_inventory(agent_id)
    properties: dict[str, Any] = {}
    # JSON Schema describes the full legal committed output, while prompt
    # contracts deliberately list raw required fields only.
    contract_fields = declared_schema_fields(schema)
    for field in sorted(contract_fields):
        field_type = type_inventory.get(field, "string_or_object")
        prop: dict[str, Any]
        if field_type == "enum":
            prop = {"type": "string", "enum": list(schema["enums"][field])}
        elif field_type == "boolean":
            prop = {"type": "boolean"}
        elif field_type == "list":
            prop = {"type": "array"}
        elif field_type == "number":
            prop = {"type": "number"}
        elif field == "safety_boundary_check":
            prop = {"type": "object", "additionalProperties": False}
        else:
            prop = {}
        properties[field] = prop
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(schema["required"]),
        "properties": properties,
    }


def build_agent_schema_contract(agent_id: str) -> str:
    schema = V2_6_AGENT_SCHEMAS[agent_id]
    type_inventory = build_agent_field_type_inventory(agent_id)
    lines = [
        f"Agent name must be exactly: {schema['agent_name']}",
        f"schema_version must be exactly: {V2_6_SCHEMA_VERSION}",
        "Return exactly one JSON object with exactly these required fields:",
    ]
    excluded = set(schema.get("prompt_excluded", set()))
    required_fields = set(schema["required"]) - excluded
    contract_fields = declared_schema_fields(schema) - excluded

    def append_field(field: str) -> None:
        field_type = type_inventory.get(field, "string_or_object")
        allowed = schema.get("enums", {}).get(field)
        if allowed:
            detail = "allowed = " + " / ".join(allowed)
        elif field_type == "boolean":
            detail = "type = boolean; use native JSON true or false only"
        elif field_type == "list":
            detail = "type = list/array"
        elif field_type == "number":
            detail = "type = number"
        elif field == "safety_boundary_check":
            detail = f"must equal {V2_6_SAFE_BOUNDARY_CHECK}"
        else:
            detail = "type = string/object as appropriate; do not add forbidden fields"
        lines.append(f"- {field}: {detail}")

    for field in sorted(required_fields):
        append_field(field)
    conditional_fields = contract_fields - required_fields
    if conditional_fields:
        lines.append("Conditional optional fields: omit them unless the role instructions explicitly require them:")
        for field in sorted(conditional_fields):
            append_field(field)
    return "\n".join(lines)


def _error(code: str, field_path: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"code": code, "field_path": field_path, "message": message, **extra}


def _collect_agent1_relation_errors(data: dict[str, Any]) -> list[dict[str, Any]]:
    allowed = {
        "ISCHEMIA_SUPPORTIVE": {"WEAK", "MODERATE", "STRONG"},
        "NO_CLEAR_ISCHEMIC_EVIDENCE": {"NONE"},
        "NON_SPECIFIC_OR_CONFOUNDED": {"NONE", "WEAK"},
        "MIXED": {"WEAK", "MODERATE"},
        "NOT_EVALUABLE": {"NONE"},
    }
    pattern = data.get("ecg_evidence_pattern")
    strength = data.get("ecg_directional_strength")
    if pattern in allowed and strength not in allowed[pattern]:
        return [_error(
            "cross_field_relation",
            "ecg_evidence_pattern/ecg_directional_strength",
            f"Agent1 incompatible pattern/strength: {pattern}+{strength}",
            allowed_strength=sorted(allowed[pattern]),
        )]
    return []


def validate_v2_6_agent_output_collect_errors(agent_id: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    schema = V2_6_AGENT_SCHEMAS[agent_id]
    errors: list[dict[str, Any]] = []
    if not isinstance(data, dict):
        return [_error("type", "$", "agent output must be an object")]
    for field in find_forbidden_v2_6_fields(data):
        errors.append(_error("forbidden_field", field, f"forbidden field present: {field}"))
    required = set(schema["required"])
    declared = declared_schema_fields(schema)
    for field in sorted(required - set(data)):
        errors.append(_error("missing_field", field, f"missing required field: {field}"))
    for field in sorted(set(data) - declared):
        errors.append(_error("extra_field", field, f"extra field not allowed: {field}"))
    if "agent_name" in required and data.get("agent_name") != schema["agent_name"]:
        errors.append(_error("agent_name", "agent_name", "agent_name mismatch", expected=schema["agent_name"], actual=data.get("agent_name")))
    for field, allowed in schema.get("enums", {}).items():
        if field in data and data.get(field) not in allowed:
            errors.append(_error("invalid_enum", field, f"invalid enum {field}: {data.get(field)!r}", allowed=list(allowed), actual=data.get(field)))
    for field in schema.get("booleans", []):
        if field in data and not isinstance(data.get(field), bool):
            errors.append(_error("invalid_boolean", field, f"{field} must be boolean", actual_type=type(data.get(field)).__name__, actual=data.get(field)))
    for field in schema.get("numbers", []):
        if field in data and (isinstance(data.get(field), bool) or not isinstance(data.get(field), (int, float))):
            errors.append(_error("invalid_number", field, f"{field} must be numeric", actual_type=type(data.get(field)).__name__))
    for field in schema.get("optional_numbers", []):
        if field in data and (isinstance(data.get(field), bool) or not isinstance(data.get(field), (int, float))):
            errors.append(_error("invalid_number", field, f"{field} must be numeric", actual_type=type(data.get(field)).__name__))
    for field in schema.get("lists", []):
        if field in data and not isinstance(data.get(field), list):
            errors.append(_error("invalid_list", field, f"{field} must be a list", actual_type=type(data.get(field)).__name__))
    for field in schema.get("optional_lists", []):
        if field in data and not isinstance(data.get(field), list):
            errors.append(_error("invalid_list", field, f"{field} must be a list", actual_type=type(data.get(field)).__name__))
    for field in schema.get("optional_booleans", []):
        if field in data and not isinstance(data.get(field), bool):
            errors.append(_error("invalid_boolean", field, f"{field} must be boolean", actual_type=type(data.get(field)).__name__, actual=data.get(field)))
    if "safety_boundary_check" in required and data.get("safety_boundary_check") != V2_6_SAFE_BOUNDARY_CHECK:
        errors.append(_error("safety_boundary", "safety_boundary_check", "safety boundary did not pass"))
    if not errors:
        try:
            if agent_id == "agent3":
                retrieved = _as_str_set(data.get("retrieved_card_ids"))
                applicable = _as_str_set(data.get("applicable_card_ids"))
                if retrieved and not applicable.issubset(retrieved):
                    errors.append(_error("cross_field_relation", "applicable_card_ids", "Agent3 applicable_card_ids must be a subset of retrieved_card_ids"))
                for field in ("supporting_card_ids", "veto_card_ids"):
                    if not _as_str_set(data.get(field)).issubset(applicable):
                        errors.append(_error("cross_field_relation", field, f"Agent3 {field} must be a subset of applicable_card_ids"))
            elif agent_id == "agent4":
                action_for_validation = data.get("agent4_action_raw")
                if data.get("evidence_sufficiency") == "INSUFFICIENT" and action_for_validation in {"UP", "DOWN"}:
                    errors.append(_error("cross_field_relation", "evidence_sufficiency", "Agent4 insufficient evidence cannot trigger UP/DOWN"))
                if data.get("safety_constraint") == "PRESERVE_ONLY" and action_for_validation in {"UP", "DOWN"}:
                    errors.append(_error("cross_field_relation", "safety_constraint", "Agent4 PRESERVE_ONLY blocks UP/DOWN"))
                if action_for_validation == "DOWN" and (
                    data.get("safety_constraint") == "NO_DOWNWARD_CORRECTION"
                    or data.get("hidden_risk_flag")
                    or data.get("unsafe_downward_correction_flag")
                ):
                    errors.append(_error("cross_field_relation", "agent4_action_raw", "Agent4 safety boundary blocks DOWN"))
                if action_for_validation == "DOWN" and data.get("rag_misuse_flag"):
                    errors.append(_error("cross_field_relation", "rag_misuse_flag", "Agent4 RAG misuse blocks DOWN"))
                if action_for_validation == "DOWN" and data.get("medical_overclaim_flag"):
                    errors.append(_error("cross_field_relation", "medical_overclaim_flag", "Agent4 medical overclaim blocks DOWN"))
                if action_for_validation == "UP" and data.get("medical_overclaim_flag"):
                    errors.append(_error("cross_field_relation", "medical_overclaim_flag", "Agent4 medical overclaim blocks UP"))
            elif agent_id == "agent2":
                present = _as_str_set(data.get("present_clinical_factor_ids"))
                continuous_for_query = _as_str_set(data.get("continuous_query_expansion_factor_ids"))
                continuous_available = continuous_for_query | _continuous_factor_ids_from_statuses(data)
                if not _as_str_set(data.get("cohort_supported_patient_factor_ids")).issubset(present | continuous_available):
                    errors.append(_error(
                        "cross_field_relation",
                        "cohort_supported_patient_factor_ids",
                        "Agent2 cohort_supported_patient_factor_ids must be present or continuous available",
                    ))
                if not _as_str_set(data.get("query_expansion_candidate_factor_ids")).issubset(present | continuous_for_query):
                    errors.append(_error(
                        "cross_field_relation",
                        "query_expansion_candidate_factor_ids",
                        "Agent2 query_expansion_candidate_factor_ids must be present or continuous interpretable",
                    ))
            elif agent_id == "agent1":
                errors.extend(_collect_agent1_relation_errors(data))
        except Exception as exc:
            errors.append(_error("cross_field_relation", "$", str(exc)))
    return errors


def validate_v2_6_agent_output(agent_id: str, data: dict[str, Any]) -> None:
    errors = validate_v2_6_agent_output_collect_errors(agent_id, data)
    if errors:
        compact = "; ".join(f"{e['code']}:{e['field_path']}:{e['message']}" for e in errors)
        raise ValueError(compact)


def _validate_agent3_trace_boundary(data: dict[str, Any], *, boundary: str) -> None:
    trace_fields = sorted(set(data) & AGENT3_RETRIEVAL_TRACE_ONLY_FIELDS)
    if trace_fields:
        raise ValueError(
            f"agent3_{boundary}_contains_retrieval_trace_fields:" + ",".join(trace_fields)
        )


def validate_agent3_raw_output(data: dict[str, Any]) -> None:
    """Validate only LLM-produced Agent3 medical-semantic fields."""
    _validate_agent3_trace_boundary(data, boundary="raw")
    validate_v2_6_agent_output("agent3", data)


def validate_agent3_committed_output(data: dict[str, Any]) -> None:
    """Validate Agent3's downstream semantic object after local enrichment."""
    _validate_agent3_trace_boundary(data, boundary="committed")
    validate_v2_6_agent_output("agent3", data)
