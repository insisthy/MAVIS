from __future__ import annotations

import hashlib
import time
from typing import Any

from .ecg_audit_agent import AgentRunResult, LLMAgentBase, stable_hash
from .missingness import classify_dataset_v4_missingness
from src.llm.json_parser import parse_json_object
from src.llm.structured_output import (
    TYPE_NORMALIZER_VERSION,
    normalize_llm_structured_output_types,
    normalize_runtime_owned_agent_identity,
    normalize_schema_enum_values,
)

from .schemas import (
    V2_6_SAFE_BOUNDARY_CHECK,
    V2_6_SCHEMA_VERSION,
    find_forbidden_v2_6_fields,
    normalize_v2_6_safety_boundary_check,
    validate_v2_6_agent_output,
)
from src.runtime.directional_policy import (
    allowed_actions_for_relation,
    normalize_agent4_action_for_directional_contract,
)


def _as_string_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value if str(item)}
    return {str(value)} if str(value) else set()


def evaluate_cohort_context_safety_v2_6(agent3_payload_or_output: dict[str, Any]) -> dict[str, Any]:
    """Deterministic fail-closed checks for v2.6 cohort background separation."""
    issues: list[str] = []
    cohort_context = list(agent3_payload_or_output.get("cohort_factor_context") or agent3_payload_or_output.get("cohort_background_context") or [])
    patient_factor_support = list(agent3_payload_or_output.get("patient_specific_factor_support", []))
    patient_topk = set(map(str, agent3_payload_or_output.get("patient_specific_retrieved_card_ids", [])))
    cohort_cards = set(map(str, agent3_payload_or_output.get("cohort_background_card_ids", agent3_payload_or_output.get("linked_cohort_external_card_ids", []))))
    text = str(agent3_payload_or_output)
    for item in cohort_context:
        if item.get("patient_fact_allowed") is True:
            issues.append("cohort_factor_written_as_patient_fact")
        if str(item.get("patient_match_status", "")) == "not_evaluated" and any(
            phrase in text
            for phrase in ("该患者高龄", "该患者HbA1c", "该患者hba1c", "patient has high age", "patient has elevated HbA1c")
        ):
            issues.append("not_evaluated_factor_patient_assertion")
        if (
            str(item.get("data_association_role", "")) == "C_ASSOCIATED"
            and str(item.get("allowed_agent3_role", "")) == "HIDDEN_RISK_COHORT_SUPPORT"
        ):
            issues.append("c_associated_factor_written_as_hidden_support")
    if patient_factor_support:
        issues.append("patient_specific_factor_support_not_empty_for_unthresholded_cohort_factor")
    if patient_topk & cohort_cards:
        issues.append("cohort_background_card_in_patient_specific_topk")
    if "patient probability" in text or "患者概率" in text:
        issues.append("cohort_coefficient_as_patient_probability")
    if "raw unit odds ratio" in text or "原始单位OR" in text:
        issues.append("standardized_coefficient_as_raw_unit_or")
    if any(str(item.get("source_scope", "")).lower() in {"mixed", "unknown", "test"} for item in cohort_context):
        issues.append("mixed_or_unknown_split_factor_in_runtime")
    return {
        "cohort_context_safety_pass": not issues,
        "cohort_context_safety_issues": sorted(set(issues)),
        "fail_closed": bool(issues),
        "preserve_score": True,
        "risk_delta": 0,
        "evidence_sufficiency": "partial_or_insufficient" if issues else "not_applicable",
    }


def build_rag_aware_consumption_guidance_v2_7(agent3_output: dict[str, Any]) -> dict[str, Any]:
    rag_status = str(agent3_output.get("rag_evidence_status", "RAG_UNAVAILABLE"))
    patient_direction = str(agent3_output.get("patient_evidence_direction", "UNCERTAIN"))
    patient_strength = str(agent3_output.get("patient_evidence_strength", "NONE"))
    return {
        "patient_evidence_is_primary": True,
        "rag_is_auxiliary_context_safety_or_contradiction_support": True,
        "rag_not_supporting_no_longer_auto_penalizes": rag_status in {
            "RAG_UNAVAILABLE", "RAG_CONTEXT_ONLY", "RAG_MECHANISM_SUPPORT",
        },
        "retrieved_but_not_supporting_is_not_lower_confidence_signal": True,
        "mild_up_allowed_from_patient_evidence_without_rag_support": (
            patient_direction == "UP"
            and patient_strength in {"MODERATE", "STRONG"}
            and rag_status in {"RAG_UNAVAILABLE", "RAG_CONTEXT_ONLY", "RAG_MECHANISM_SUPPORT"}
        ),
        "down_requires_stronger_patient_and_safety_support": True,
        "down_safety_constraints_preserved": True,
        "cohort_prior_alone_block_preserved": True,
        "rag_alone_block_preserved": True,
        "agent3_patient_evidence_direction": patient_direction,
        "agent3_patient_evidence_strength": patient_strength,
        "agent3_rag_evidence_status": rag_status,
        "agent3_rag_should_affect_agent4_policy": bool(agent3_output.get("rag_should_affect_agent4_policy", False)),
    }


AGENT4_SAFE_TRIAGE_KEYS = (
    "evidence_alignment",
    "local_model_evidence_conflict_present",
    "evidence_sufficiency",
    "overall_confidence",
    "remaining_uncertainty",
    "rag_status",
)

AGENT4_FORBIDDEN_TRIAGE_TOKEN_FRAGMENTS = (
    "score", "threshold", "margin", "baseline", "locked_", "high_risk",
    "low_risk", "above_threshold", "below_threshold", "probability",
)
AGENT4_FORBIDDEN_SAFE_VIEW_KEYS = {
    "score", "threshold", "margin", "baseline", "locked", "risk_level",
    "risk_bucket", "prediction", "model_evidence_ids", "proposal_direction",
    "proposal_strength", "proposal_confidence", "proposal_reason",
    "global_consistency_status", "discordance_type", "score_alignment",
    "fusion_relation",
}

def _assert_no_forbidden_structured_keys(value: Any, *, path: str = "") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_l = str(key).lower()
            if any(token in key_l for token in AGENT4_FORBIDDEN_SAFE_VIEW_KEYS):
                raise ValueError(f"Agent4 safe view forbidden key: {path}.{key}")
            _assert_no_forbidden_structured_keys(child, path=f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_forbidden_structured_keys(child, path=f"{path}[{index}]")


def _compact_agent4_list(values: Any, *, limit: int = 8) -> list[Any]:
    """Bound list-like Agent4 prompt fields without changing their semantics."""

    if not isinstance(values, list):
        return []
    return list(values[:limit])


def _compact_agent4_text(value: Any, *, limit: int = 320) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:limit] + "...[truncated_for_agent4_prompt_budget]"


def build_agent1_safe_view_for_agent4(agent1_output: dict[str, Any]) -> dict[str, Any]:
    """Redact Agent1 output to patient-level ECG evidence and non-numeric audit state."""

    safe = {
        "ecg_evidence_pattern": agent1_output.get("ecg_evidence_pattern"),
        "ecg_directional_strength": agent1_output.get("ecg_directional_strength"),
        "ecg_evidence_informativeness": agent1_output.get("ecg_evidence_informativeness"),
        "ecg_signal_quality": agent1_output.get("ecg_signal_quality"),
        "ecg_uncertainty": agent1_output.get("ecg_uncertainty"),
        "hidden_ischemia_warning": bool(agent1_output.get("hidden_ischemia_warning", False)),
        "false_positive_mimic_warning": bool(agent1_output.get("false_positive_mimic_warning", False)),
        "ischemic_evidence_ids": _compact_agent4_list(agent1_output.get("ischemic_evidence_ids", []) or []),
        "non_specific_or_confounding_evidence_ids": _compact_agent4_list(
            agent1_output.get("non_specific_or_confounding_evidence_ids", []) or []
        ),
        "ecg_local_concern": agent1_output.get("ecg_local_concern", "UNRESOLVED"),
        "ecg_audit_confidence": agent1_output.get("ecg_audit_confidence", "LOW"),
        "ecg_model_explanation_consistency": agent1_output.get("ecg_model_explanation_consistency", "UNAVAILABLE"),
        "ecg_audit_reason_code": _compact_agent4_text(agent1_output.get("ecg_audit_reason_code", ""), limit=160),
    }
    _assert_no_forbidden_structured_keys(safe)
    return safe


def build_agent2_safe_view_for_agent4(agent2_output: dict[str, Any]) -> dict[str, Any]:
    """Pass only the Agent2 minimal committed evidence packet to Agent4."""

    missingness = classify_dataset_v4_missingness(agent2_output)
    safe = {
        "selected_discrepancy_head": agent2_output.get("selected_discrepancy_head"),
        "risk_band": agent2_output.get("risk_band"),
        "alert_state": agent2_output.get("alert_state"),
        "evidence_stance": agent2_output.get("evidence_stance"),
        "supporting_factor_ids": _compact_agent4_list(agent2_output.get("supporting_factor_ids", []) or []),
        "contradicting_factor_ids": _compact_agent4_list(agent2_output.get("contradicting_factor_ids", []) or []),
        "confounder_factor_ids": _compact_agent4_list(agent2_output.get("confounder_factor_ids", []) or []),
        "reported_missing_factor_ids": _compact_agent4_list(missingness["reported_missing_factor_ids"]),
        "structural_missing_factor_ids": _compact_agent4_list(missingness["structural_missing_factor_ids"]),
        "action_blocking_missing_factor_ids": _compact_agent4_list(missingness["action_blocking_missing_factor_ids"]),
        "missingness_action_policy": missingness["missingness_action_policy"],
        "evidence_sufficiency": agent2_output.get("evidence_sufficiency"),
        "concise_rationale": _compact_agent4_text(agent2_output.get("concise_rationale", ""), limit=240),
    }
    _assert_no_forbidden_structured_keys(safe)
    return safe


def build_agent3_safe_view_for_agent4(agent3_output: dict[str, Any]) -> dict[str, Any]:
    """Return Agent3's compact committed arbitration view for Agent4.

    Raw Agent3 text may mention every retrieved card, including cards that the
    deterministic committed layer later classifies as review-only.  Those raw
    mentions must not re-introduce a veto after policy relevance was rejected.
    """
    rag_affects_policy = bool(agent3_output.get("rag_should_affect_agent4_policy", False))
    if rag_affects_policy:
        supporting_card_ids = agent3_output.get(
            "correction_supporting_card_ids", agent3_output.get("supporting_card_ids", [])
        )
        veto_card_ids = agent3_output.get(
            "safety_warning_card_ids", agent3_output.get("veto_card_ids", [])
        )
        applicable_card_ids = list(
            dict.fromkeys([*(supporting_card_ids or []), *(veto_card_ids or [])])
        )
    else:
        supporting_card_ids = []
        veto_card_ids = []
        applicable_card_ids = []

    committed_confidence = agent3_output.get(
        "correction_confidence_relative_to_ecg",
        agent3_output.get("overall_confidence", "LOW"),
    )
    relation = str(agent3_output.get(
        "correction_relation_relative_to_ecg", "ECG_BASELINE_RELATION_UNRESOLVED"
    ))
    committed_alignment = (
        "COMMITTED_DIRECTIONAL_SUPPORT"
        if relation in {
            "ECG_BASELINE_POSSIBLY_UNDERSTATED",
            "ECG_BASELINE_POSSIBLY_OVERSTATED",
        }
        else agent3_output.get("evidence_alignment", "UNCERTAIN")
    )
    committed_uncertainty = (
        []
        if relation in {
            "ECG_BASELINE_POSSIBLY_UNDERSTATED",
            "ECG_BASELINE_POSSIBLY_OVERSTATED",
        }
        else agent3_output.get("remaining_uncertainty", [])
    )
    committed_rationale = committed_alignment
    safe = {
        "correction_relation_relative_to_ecg": relation,
        "correction_evidence_grade": agent3_output.get("correction_evidence_grade", "NONE"),
        "evidence_alignment": committed_alignment,
        "evidence_sufficiency": agent3_output.get("evidence_sufficiency", "INSUFFICIENT"),
        "overall_confidence": committed_confidence,
        "supporting_evidence_ids": _compact_agent4_list(agent3_output.get("supporting_evidence_ids", []), limit=8),
        "contradicting_evidence_ids": _compact_agent4_list(agent3_output.get("contradicting_evidence_ids", []), limit=8),
        "applicable_card_ids": _compact_agent4_list(applicable_card_ids, limit=8),
        "supporting_card_ids": _compact_agent4_list(supporting_card_ids, limit=8),
        "veto_card_ids": _compact_agent4_list(veto_card_ids, limit=8),
        "remaining_uncertainty": _compact_agent4_list(committed_uncertainty, limit=6),
        "brief_arbitration_rationale": _compact_agent4_text(committed_rationale, limit=500),
    }
    _assert_no_forbidden_structured_keys(safe)
    return safe


def build_agent4_safe_triage(context: dict[str, Any]) -> dict[str, Any]:
    """Build the only triage object allowed into Agent4's LLM payload."""
    agent3 = context.get("agent3_output", {}) or {}
    relation = str(agent3.get("correction_relation_relative_to_ecg", ""))
    alignment = (
        "COMMITTED_DIRECTIONAL_SUPPORT"
        if relation in {
            "ECG_BASELINE_POSSIBLY_UNDERSTATED",
            "ECG_BASELINE_POSSIBLY_OVERSTATED",
        }
        else agent3.get("evidence_alignment")
        or agent3.get("modality_alignment")
        or "UNCERTAIN"
    )
    conflict = alignment in {"DISAGREEMENT", "UNCERTAIN"}
    committed_uncertainty = (
        []
        if relation in {
            "ECG_BASELINE_POSSIBLY_UNDERSTATED",
            "ECG_BASELINE_POSSIBLY_OVERSTATED",
        }
        else agent3.get("remaining_uncertainty", [])
    )
    safe = {
        "evidence_alignment": str(alignment),
        "local_model_evidence_conflict_present": bool(conflict),
        "evidence_sufficiency": str(agent3.get("evidence_sufficiency", "INSUFFICIENT")),
        "overall_confidence": str(
            agent3.get("correction_confidence_relative_to_ecg", agent3.get("overall_confidence", "LOW"))
        ),
        "remaining_uncertainty": _compact_agent4_list(committed_uncertainty, limit=6),
        "rag_status": (
            "RAG_POLICY_RELEVANT"
            if agent3.get("rag_should_affect_agent4_policy", False)
            else "RAG_NOT_POLICY_RELEVANT"
            if agent3.get("retrieved_card_ids")
            else "RAG_UNAVAILABLE"
        ),
    }
    _assert_no_forbidden_structured_keys(safe)
    return {key: safe[key] for key in AGENT4_SAFE_TRIAGE_KEYS}


def _agent4_evidence_id_allowlist(context: dict[str, Any]) -> set[str]:
    """Return only IDs already visible to the Agent4 safety workflow.

    This is deliberately a provenance allowlist, not a medical inference
    mechanism.  It only collects existing Agent1/2 evidence IDs and Agent3
    patient/card/safety IDs that were available before Agent4 was called.
    """

    agent1 = context.get("agent1_output", {}) or {}
    agent2 = context.get("agent2_output", {}) or {}
    agent3 = context.get("agent3_output", {}) or {}
    allowed: set[str] = set()
    for source, keys in (
        (agent1, ("ischemic_evidence_ids", "non_specific_or_confounding_evidence_ids", "possible_ecg_mimic_terms")),
        (agent2, ("supporting_factor_ids", "contradicting_factor_ids", "confounder_factor_ids", "critical_missing_factor_ids")),
        (agent3, ("patient_evidence_ids", "applicable_card_ids", "supporting_card_ids", "veto_card_ids", "external_knowledge_card_ids", "cited_card_ids", "safety_warning_card_ids")),
    ):
        for key in keys:
            allowed |= _as_string_set(source.get(key))
    for chain in agent3.get("claim_evidence_chain", []) or []:
        if isinstance(chain, dict):
            allowed |= _as_string_set(chain.get("patient_evidence_ids"))
            allowed |= _as_string_set(chain.get("external_card_ids"))
            allowed |= _as_string_set(chain.get("counterevidence_card_ids"))
    return allowed


def _map_action_confidence(correction_confidence: Any) -> str:
    return {
        "LOW": "LOW",
        "MODERATE": "MEDIUM",
        "HIGH": "HIGH",
    }.get(str(correction_confidence or "").upper(), "LOW")


def _derive_critical_safety_conflict(
    raw_output: dict[str, Any],
    agent3_output: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Aggregate already-effective safety signals without creating new rules."""

    sources: list[str] = []
    if bool(raw_output.get("hidden_risk_flag")):
        sources.append("hidden_risk_flag")
    if bool(raw_output.get("unsafe_downward_correction_flag")):
        sources.append("unsafe_downward_correction_flag")
    if bool(raw_output.get("rag_misuse_flag")):
        sources.append("rag_misuse_flag")
    if bool(raw_output.get("medical_overclaim_flag")):
        sources.append("medical_overclaim_flag")
    if str(agent3_output.get("safety_block_status", "NONE")).upper() != "NONE":
        sources.append("agent3_safety_block_status")
    return bool(sources), sources


class SafetyReflectionAgent(LLMAgentBase):
    """Agent4 safety reviewer that emits v2.7 ordinal correction policy only."""

    agent_id = "agent4"
    agent_name = "Agent4_Safety_Constrained_ECG_Correction_Policy_Agent"
    prompt_key = "agent4"
    schema_key = "agent4"
    maximum_agent4_generation_attempts = 2
    maximum_empty_content_regeneration_attempts = 1

    def __init__(
        self,
        client: Any | None = None,
        *,
        maximum_agent4_generation_attempts: int | None = None,
        maximum_empty_content_regeneration_attempts: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(client, **kwargs)
        self.maximum_agent4_generation_attempts = max(
            1,
            int(maximum_agent4_generation_attempts or type(self).maximum_agent4_generation_attempts),
        )
        self.maximum_empty_content_regeneration_attempts = max(
            0,
            int(maximum_empty_content_regeneration_attempts
                if maximum_empty_content_regeneration_attempts is not None
                else type(self).maximum_empty_content_regeneration_attempts),
        )
        if self.maximum_empty_content_regeneration_attempts >= self.maximum_agent4_generation_attempts:
            self.maximum_empty_content_regeneration_attempts = self.maximum_agent4_generation_attempts - 1


    def run_with_trace(self, context: dict[str, Any]) -> AgentRunResult:
        if self.client is None:
            raise RuntimeError(f"{self.__class__.__name__} requires an injected LLM client")
        payload = self.build_payload(context)
        self.validate_payload(payload)
        user_payload = {
            "agent_id": self.agent_id,
            "case_id": str(context.get("case_id", "")),
            "runtime_version": "v2.8",
            "schema_version": V2_6_SCHEMA_VERSION,
            "schema_hash": stable_hash(self.schema),
            "input": payload,
        }
        raw = ""
        parsed: dict[str, Any] = {}
        parsed_before_type_normalization: dict[str, Any] = {}
        normalization: dict[str, Any] = {}
        empty_content_count = 0
        errors: list[str] = []
        started_all = time.perf_counter()
        for attempt in range(1, self.maximum_agent4_generation_attempts + 1):
            attempt_payload = dict(user_payload)
            if empty_content_count:
                attempt_payload["agent4_empty_content_regeneration"] = {
                    "enabled": True,
                    "reason": "AGENT4_EMPTY_FINAL_CONTENT",
                    "instruction": (
                        "Return only the required compact JSON object. Do not include markdown, "
                        "analysis, chain-of-thought, or any non-JSON text."
                    ),
                    "same_safe_view": True,
                    "new_medical_information_added": False,
                }
            try:
                raw = self.client.complete(self.prompt, attempt_payload)
            except Exception as exc:
                error_code = getattr(exc, "error_code", "") or type(exc).__name__
                errors.append(f"{error_code}:{exc}")
                if (
                    error_code == "AGENT4_EMPTY_FINAL_CONTENT"
                    and empty_content_count < self.maximum_empty_content_regeneration_attempts
                    and attempt < self.maximum_agent4_generation_attempts
                ):
                    empty_content_count += 1
                    continue
                raise
            if not str(raw or "").strip():
                errors.append("AGENT4_EMPTY_FINAL_CONTENT:empty raw response")
                if (
                    empty_content_count < self.maximum_empty_content_regeneration_attempts
                    and attempt < self.maximum_agent4_generation_attempts
                ):
                    empty_content_count += 1
                    continue
                raise ValueError("AGENT4_EMPTY_FINAL_CONTENT")
            parsed, parse_error = parse_json_object(raw)
            if parse_error:
                raise ValueError("Agent4 non-empty content is not valid JSON")
            parsed_before_type_normalization = dict(parsed)
            parsed, agent_identity_actions = normalize_runtime_owned_agent_identity(
                self.agent_id, parsed
            )
            parsed, type_actions = normalize_llm_structured_output_types(
                parsed,
                self.schema.get("booleans", []),
            )
            parsed, enum_actions = normalize_schema_enum_values(self.agent_id, parsed)
            parsed, normalization = normalize_v2_6_safety_boundary_check(parsed)
            normalization["type_normalizer_version"] = TYPE_NORMALIZER_VERSION
            normalization["type_normalization_actions"] = (
                agent_identity_actions + type_actions + enum_actions
            )
            normalization["parsed_json_before_type_normalization"] = parsed_before_type_normalization
            validate_v2_6_agent_output(self.agent_id, parsed)
            break
        else:
            raise RuntimeError("Agent4 generation loop exhausted")

        # Agent4 ends at a Raw semantic recommendation.  The single controller
        # owns all normalization, safety vetoes, tiers, and numerical execution.
        committed = dict(parsed)
        audit = {"independent_precommit_removed": True, "raw_response_preserved": True}
        trace = {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "case_id": str(context.get("case_id", "")),
            "schema_version": V2_6_SCHEMA_VERSION,
            "prompt_hash": hashlib.sha256(self.prompt.encode("utf-8")).hexdigest(),
            "model_inference_config_hash": self.model_inference_config_hash(),
            "transport_execution_config_hash": self.transport_execution_config_hash(),
            "attempt_count": 1 + empty_content_count,
            "retry_request_count": empty_content_count,
            "maximum_agent4_generation_attempts": self.maximum_agent4_generation_attempts,
            "maximum_empty_content_regeneration_attempts": self.maximum_empty_content_regeneration_attempts,
            "empty_content_regeneration_count": empty_content_count,
            "empty_content_errors": errors,
            "raw_response": raw,
            "parsed_output_raw": dict(parsed),
            "parsed_output": committed,
            "normalization": normalization,
            "parsed_json_before_normalization": parsed_before_type_normalization,
            "normalization_actions": normalization.get("type_normalization_actions", []),
            "normalized_json": committed,
            "safety_boundary_check": V2_6_SAFE_BOUNDARY_CHECK,
            "single_controller_handoff": audit,
            "elapsed_seconds": round(time.perf_counter() - started_all, 6),
            "nested_retry_possible": False,
            "empty_content_not_transport_retry": True,
            "empty_content_not_schema_repair": True,
        }
        self.last_trace = trace
        if self.trace_sink:
            self.trace_sink(trace)
        return AgentRunResult(output=committed, trace=trace)

    def build_payload(self, context: dict[str, Any]) -> dict[str, Any]:
        safe_triage = build_agent4_safe_triage(context)
        payload = {
            "case_id": str(context.get("case_id", "")),
            "agent1_safe_view": build_agent1_safe_view_for_agent4(context["agent1_output"]),
            "agent2_safe_view": build_agent2_safe_view_for_agent4(context["agent2_output"]),
            "agent3_safe_view": build_agent3_safe_view_for_agent4(context["agent3_output"]),
            "ecg_anchor_context_redaction": {
                "base_score_values_hidden_from_agent4_llm": True,
                "threshold_values_hidden_from_agent4_llm": True,
                "margin_values_hidden_from_agent4_llm": True,
                "baseline_prediction_hidden_from_agent4_llm": True,
                "deterministic_controller_receives_locked_ecg_anchor_context_after_agent4": True,
            },
            "agent4_safe_triage": safe_triage,
            "review_boundary": {
                "agent4_review_is_non_executing": True,
                "may_not_directly_execute_or_determine_eligibility": True,
                "enumerated_safety_findings_may_be_consumed_by_controller": True,
                "may_not_output_delta_or_score": True,
                "may_only_output_ordinal_correction_policy": True,
                "validation_policy_is_not_yet_locked_so_controller_applies_zero_delta": not bool(
                    context.get("runtime_correction_policy_locked", False)
                ),
                "single_agent4_review_only": True,
                "additional_retrieval_or_review_pass_prohibited": True,
                "may_not_invent_patient_facts": True,
                "cohort_prior_is_not_patient_fact_or_causal_claim": True,
                "cohort_background_card_not_patient_topk": True,
                "not_evaluated_factor_no_patient_assertion": True,
                "standardized_coefficient_not_raw_unit_or": True,
                "patient_evidence_primary_rag_auxiliary": True,
                "rag_not_supporting_not_a_confidence_penalty": True,
                "retrieved_but_not_supporting_not_a_review_required_trigger": True,
                "mild_up_may_use_patient_evidence_without_rag_support": True,
                "downward_correction_still_requires_safety_clearance": True,
                "v2_8_neutral_rag_gate_required": True,
                "agent4_llm_score_blind": True,
                "ecg_baseline_relation_is_agent4_visible_but_raw_score_hidden": True,
                "final_single_controller_is_the_only_executor": True,
                "absolute_clinical_risk_direction_not_executable": True,
                "correction_relation_relative_to_ecg_drives_action_legality": True,
                "weak_moderate_strong_directional_candidates_allowed_without_rag_requirement": True,
                "overestimation_branch_forbids_up_underestimation_branch_forbids_down": True,
            },
            "agent4_transport_budget_contract": {
                "payload_safe_view_compacted": True,
                "raw_score_and_threshold_values_hidden": True,
                "medical_action_set_unchanged": True,
                "agent4_may_not_output_delta_or_final_score": True,
                "downstream_controller_remains_deterministic_executor": True,
            },
        }
        forbidden = find_forbidden_v2_6_fields(payload)
        if forbidden:
            raise ValueError(f"forbidden Agent4 payload fields: {forbidden}")
        return payload
