from __future__ import annotations

import csv
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from src.llm.json_parser import parse_json_object
from src.llm.structured_output import normalize_schema_enum_values, remove_matching_runtime_stage_metadata
from src.models.clinical_discrepancy_model import (
    assert_agent2_discrepancy_models_locked,
    predict_agent2_discrepancy_from_locked_artifacts,
)

from .ecg_audit_agent import AgentRunResult, LLMAgentBase, stable_hash
from .prompts import V2_6_AGENT_EVIDENCE_PROMPTS
from .schemas import (
    V2_6_AGENT_STAGE_SCHEMAS,
    V2_6_SAFE_BOUNDARY_CHECK,
    V2_6_SCHEMA_VERSION,
    find_forbidden_v2_6_fields,
    validate_agent_stage_output,
    validate_v2_6_agent_output,
)

CLINICAL_KEYS = {
    "age", "age_years", "sex", "sex_standardized", "hypertension",
    "hypertension_history", "hypertension_duration_years", "hypertension_control_status",
    "diabetes", "diabetes_history", "diabetes_duration_years", "diabetes_control_status", "smoking",
    "smoking_history", "drinking", "drinking_history", "chest_pain_type",
    "smoking_amount_per_day", "smoking_years",
    "chest_pain_subtype", "symptom_attack_status", "stroke_history",
    "pad_history", "stroke_or_pad_history", "chd_family_history",
    "digoxin_use", "antiarrhythmic_use", "tricyclic_antidepressant_use",
    "st_t_affecting_drug_use", "medication_summary", "TC", "TG", "HDL_C",
    "LDL_C", "GLU", "HbA1c", "HGB", "CREA", "eGFR", "CRP", "TSH",
    "T3", "T4", "FT3", "FT4", "lab_indicator_summary",
    "clinical_history_summary", "clinical_risk_factor_summary",
    "clinical_hidden_fn_like_factors", "clinical_false_positive_like_factors",
    "agent2_clinical_evidence_summary",
    "patient_level_feature_contributions", "feature_contribution_method",
}

FORMAL_CLINICAL_FEATURES = (
    "age_years", "sex_standardized", "hypertension_history",
    "hypertension_duration_years", "hypertension_control_status",
    "diabetes_history", "diabetes_duration_years", "diabetes_control_status",
    "stroke_or_pad_history", "chest_pain_type", "chest_pain_subtype",
    "chd_family_history", "symptom_attack_status", "TC", "TG", "HDL_C",
    "LDL_C", "GLU", "HbA1c", "HGB", "CREA", "eGFR", "CRP", "TSH",
    "FT3", "FT4",
)

CANONICAL_CLINICAL_FACTOR_MAP = {
    "age": ("age", "age_years"),
    "age_years": ("age", "age_years"),
    "sex": ("sex", "sex"),
    "sex_standardized": ("sex", "sex"),
    "hypertension": ("hypertension", "hypertension_history"),
    "hypertension_history": ("hypertension", "hypertension_history"),
    "hypertension_duration_years": ("hypertension_duration", "hypertension_duration_years"),
    "hypertension_control_status": ("hypertension_control", "hypertension_control_status"),
    "diabetes": ("diabetes", "diabetes_history"),
    "diabetes_history": ("diabetes", "diabetes_history"),
    "diabetes_duration_years": ("diabetes_duration", "diabetes_duration_years"),
    "diabetes_control_status": ("diabetes_control", "diabetes_control_status"),
    "smoking": ("smoking", "smoking_history"),
    "smoking_history": ("smoking", "smoking_history"),
    "smoking_amount_per_day": ("smoking_amount_per_day", "smoking_amount_per_day"),
    "smoking_years": ("smoking_years", "smoking_years"),
    "drinking": ("drinking", "drinking_history"),
    "drinking_history": ("drinking", "drinking_history"),
    "stroke_history": ("stroke_history", "stroke_history"),
    "pad_history": ("peripheral_artery_disease", "pad_history"),
    "stroke_or_pad_history": ("stroke_or_peripheral_artery_disease", "stroke_or_pad_history"),
    "chd_family_history": ("family_history_chd", "chd_family_history"),
    "chest_pain_type": ("chest_pain_type", "chest_pain_type"),
    "chest_pain_subtype": ("chest_pain_subtype", "chest_pain_subtype"),
    "symptom_attack_status": ("symptom_attack_status", "symptom_attack_status"),
    "digoxin_use": ("digoxin_use", "digoxin_use"),
    "antiarrhythmic_use": ("antiarrhythmic_use", "antiarrhythmic_use"),
    "tricyclic_antidepressant_use": ("tricyclic_antidepressant_use", "tricyclic_antidepressant_use"),
    "st_t_affecting_drug_use": ("st_t_affecting_drug_use", "st_t_affecting_drug_use"),
    "TC": ("total_cholesterol", "TC"),
    "TG": ("triglycerides", "TG"),
    "HDL_C": ("hdl_cholesterol", "HDL_C"),
    "LDL_C": ("ldl_cholesterol", "LDL_C"),
    "GLU": ("glucose", "GLU"),
    "HbA1c": ("hba1c", "HbA1c"),
    "HGB": ("hemoglobin", "HGB"),
    "CREA": ("creatinine", "CREA"),
    "eGFR": ("egfr", "eGFR"),
    "CRP": ("crp", "CRP"),
    "TSH": ("tsh", "TSH"),
    "T3": ("t3", "T3"),
    "T4": ("t4", "T4"),
    "FT3": ("ft3", "FT3"),
    "FT4": ("ft4", "FT4"),
}

CONTINUOUS_FACTOR_IDS = {
    "age", "hypertension_duration", "diabetes_duration",
    "smoking_amount_per_day", "smoking_years",
    "total_cholesterol", "triglycerides", "hdl_cholesterol",
    "ldl_cholesterol", "glucose", "hba1c", "hemoglobin",
    "creatinine", "egfr", "crp", "tsh", "t3", "t4", "ft3", "ft4",
}

CONTINUOUS_NUMERIC_DIRECTION_WORDS = {
    "high", "higher", "low", "lower", "elevated", "reduced", "decreased",
    "increased", "abnormal", "normal", "markedly", "poor", "worse",
}

CONTINUOUS_NUMERIC_ALIASES = {
    "age_years": ["age", "older", "elderly"],
    "hypertension_duration_years": ["hypertension duration"],
    "diabetes_duration_years": ["diabetes duration"],
    "smoking_amount_per_day": ["smoking amount"],
    "smoking_years": ["smoking years"],
    "TC": ["total cholesterol", "tc"],
    "TG": ["triglyceride", "tg"],
    "HDL_C": ["hdl", "hdl-c", "hdl_c"],
    "LDL_C": ["ldl", "ldl-c", "ldl_c"],
    "GLU": ["glu", "glucose"],
    "HbA1c": ["hba1c", "a1c"],
    "HGB": ["hgb", "hemoglobin"],
    "CREA": ["crea", "creatinine"],
    "eGFR": ["egfr"],
    "CRP": ["crp", "c-reactive"],
    "TSH": ["tsh"],
    "T3": [" t3 ", "triiodothyronine"],
    "T4": [" t4 ", "thyroxine"],
    "FT3": ["ft3", "free t3"],
    "FT4": ["ft4", "free t4"],
}

NOMINAL_CATEGORICAL_FACTOR_IDS = {
    "sex", "chest_pain_type", "chest_pain_subtype",
    "symptom_attack_status", "hypertension_control",
    "diabetes_control",
}

# High-confidence patient-factor direction rules used only to verify the
# polarity of factor IDs already selected by Agent2-A.  These rules do not add
# evidence, interpret continuous values, or replace the locked discrepancy
# model.  "HIGHER" and "LOWER" describe the patient's clinical risk context;
# the selected discrepancy head determines whether that context supports or
# contradicts the discrepancy hypothesis.
BRANCH_POLARITY_BINARY_RISK_FACTOR_IDS = {
    "hypertension",
    "diabetes",
    "smoking",
    "stroke_history",
    "peripheral_artery_disease",
    "stroke_or_peripheral_artery_disease",
    "family_history_chd",
}

BRANCH_POLARITY_CATEGORICAL_VALUES = {
    "chest_pain_type": {
        "typical_angina": "HIGHER",
        "atypical_chest_pain": "LOWER",
        "asymptomatic": "LOWER",
    },
    "chest_pain_subtype": {
        "typical_angina": "HIGHER",
        "anginal_equivalent": "HIGHER",
        "atypical_chest_pain": "LOWER",
        "atypical_context": "LOWER",
        "activity_unrelated_atypical": "LOWER",
        "no_chest_pain_or_asymptomatic": "LOWER",
        "non_cardiac_context_excluded": "LOWER",
    },
    "symptom_attack_status": {
        "acute_attack": "HIGHER",
        "non_acute_or_routine": "LOWER",
    },
    # Controlled disease is not treated as absence of disease.  Only the
    # unambiguously adverse control states are assigned a fixed direction.
    "hypertension_control": {
        "poor_control": "HIGHER",
        "untreated_or_irregular": "HIGHER",
    },
    "diabetes_control": {
        "poor_control": "HIGHER",
        "untreated_or_irregular": "HIGHER",
    },
}

# These pairs originate from related source fields and may occasionally carry
# the exact same normalized category.  Exact equality is deduplicated only in
# the LLM semantic evidence lists; the locked Agent2 model feature vector is
# untouched.  The first factor is the canonical evidence pointer when both are
# present with the same value.
EXACT_DUPLICATE_CLINICAL_EVIDENCE_GROUPS = (
    ("chest_pain_type", "chest_pain_subtype"),
)

CLINICAL_UNIT_LOCK_SOURCE = "USER_CONFIRMED_FROM_CLEANED_HOSPITAL_ORIGINAL_LAB_SHEETS"
CLINICAL_DATA_PROVENANCE_SOURCE = "CLEANED_HOSPITAL_ORIGINAL_LAB_SHEETS"

CLINICAL_CONFIRMED_UNITS = {
    "TC": "mmol/L",
    "TG": "mmol/L",
    "HDL_C": "mmol/L",
    "LDL_C": "mmol/L",
    "GLU": "mmol/L",
    "HbA1c": "%",
    "HGB": "g/L",
    "CREA": "μmol/L",
    "eGFR": "mL/min/1.73 m²",
    "CRP": "mg/L",
    "TSH": "mIU/L",
    "T3": "nmol/L",
    "T4": "nmol/L",
    "FT3": "pmol/L",
    "FT4": "pmol/L",
}

UNIT_CONFIRMED_FACTOR_IDS = {
    CANONICAL_CLINICAL_FACTOR_MAP[column][0]
    for column in CLINICAL_CONFIRMED_UNITS
}

MISSING_VALUES = {"", "na", "n/a", "nan", "none", "null", "unknown", "missing", "未记录", "未知"}
PRESENT_VALUES = {"1", "true", "yes", "y", "present", "positive", "有", "是", "阳性"}
ABSENT_VALUES = {"0", "false", "no", "n", "absent", "negative", "无", "否", "阴性"}
ALLOWED_PRIOR_RUNTIME_USE = {"query_expansion", "retrieval_rerank", "offline_analysis_only", "none"}
ALLOWED_RUNTIME_PRIOR_STATUS = {
    "ELIGIBLE_QUERY_EXPANSION", "ELIGIBLE_RERANK_ONLY", "SUPPORTIVE_OFFLINE_ONLY", "EXCLUDED",
}


def _status_for_value(factor_id: str, value: Any) -> str:
    if value is None:
        return "missing"
    text = str(value).strip()
    if text.lower() in MISSING_VALUES:
        return "missing"
    if factor_id in CONTINUOUS_FACTOR_IDS:
        try:
            float(text)
            return "continuous_value_available"
        except ValueError:
            return "unknown"
    lowered = text.lower()
    if lowered in PRESENT_VALUES:
        return "present"
    if lowered in ABSENT_VALUES:
        return "absent"
    if factor_id in NOMINAL_CATEGORICAL_FACTOR_IDS:
        return "categorical_value_available"
    return "unknown"


def build_patient_factor_statuses(packet: dict[str, Any]) -> list[dict[str, Any]]:
    factors: dict[str, dict[str, Any]] = {}
    for source_column, (factor_id, canonical_name) in CANONICAL_CLINICAL_FACTOR_MAP.items():
        raw = packet.get(source_column)
        status = _status_for_value(factor_id, raw)
        unit_confirmed = source_column in CLINICAL_CONFIRMED_UNITS
        confirmed_unit = CLINICAL_CONFIRMED_UNITS.get(source_column, "")
        existing = factors.get(factor_id)
        if existing and existing.get("value_available") and status in {"missing", "unknown"}:
            continue
        factors[factor_id] = {
            "patient_factor_id": factor_id,
            "canonical_factor_name": canonical_name,
            "source_column": source_column,
            "patient_factor_status": status,
            "patient_value": raw,
            "value_available": status not in {"missing", "unknown"},
            "missing": status == "missing",
            "source_value_state": status,
            "source_field_ids": [source_column],
            "source_table": str(packet.get("_source_table", "data/processed/v2_9/agent2_input_v5.csv")),
            "projection_rule_id": f"v2_6_clinical_{source_column}_state_v1",
            "patient_specific": True,
            "score_derived": False,
            "label_derived": False,
            "unit_confirmation_status": "CONFIRMED" if unit_confirmed else "NOT_APPLICABLE",
            "confirmed_unit": confirmed_unit,
            "unit_confirmation_source": CLINICAL_UNIT_LOCK_SOURCE if unit_confirmed else "",
            "clinical_data_provenance_source": CLINICAL_DATA_PROVENANCE_SOURCE if unit_confirmed else "",
            "clinical_data_provenance_confirmed": unit_confirmed,
            "hospital_reference_range_confirmed": False if unit_confirmed else None,
            "reference_range_confirmation_status": "NOT_CONFIRMED" if unit_confirmed else "NOT_APPLICABLE",
            "hospital_range_claim_allowed": False,
            "agent_interpretation_allowed": True,
        }
    return [factors[key] for key in sorted(factors)]


def load_cohort_factor_prior_registry(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    prior_path = Path(path)
    if not prior_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with prior_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("train_only", "")).strip().lower() != "true":
                continue
            if str(row.get("causal_claim_allowed", "")).strip().lower() == "true":
                continue
            runtime_use = str(row.get("runtime_use", "")).strip()
            runtime_status = str(row.get("runtime_prior_status", "")).strip()
            if runtime_use not in ALLOWED_PRIOR_RUNTIME_USE:
                continue
            if runtime_status and runtime_status not in ALLOWED_RUNTIME_PRIOR_STATUS:
                continue
            rows.append(dict(row))
    return rows


def match_patient_factors_to_cohort_priors(
    patient_factors: list[dict[str, Any]],
    cohort_priors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    factor_by_id = {str(item.get("patient_factor_id")): item for item in patient_factors}
    matches: list[dict[str, Any]] = []
    for prior in cohort_priors:
        factor_id = str(prior.get("factor_id") or prior.get("canonical_factor_name") or "")
        patient = factor_by_id.get(factor_id)
        if not patient:
            continue
        status = str(patient.get("patient_factor_status"))
        runtime_status = str(prior.get("runtime_prior_status", "")).strip()
        runtime_use = str(prior.get("runtime_use", "none"))
        if not runtime_status:
            runtime_status = "ELIGIBLE_QUERY_EXPANSION" if runtime_use == "query_expansion" else "ELIGIBLE_RERANK_ONLY" if runtime_use == "retrieval_rerank" else "EXCLUDED"
        present_or_allowed_continuous = status == "present"
        if status == "continuous_value_available" and str(prior.get("continuous_state_rule", "")).strip():
            present_or_allowed_continuous = True
        eligible_query = (
            present_or_allowed_continuous
            and runtime_status == "ELIGIBLE_QUERY_EXPANSION"
            and runtime_use == "query_expansion"
        )
        eligible_rerank = (
            (status == "present" or status == "continuous_value_available")
            and runtime_status in {"ELIGIBLE_QUERY_EXPANSION", "ELIGIBLE_RERANK_ONLY"}
            and runtime_use in {"query_expansion", "retrieval_rerank"}
        )
        matches.append({
            "factor_id": factor_id,
            "branch_scope": prior.get("branch_scope", "global_cag"),
            "patient_factor_status": status,
            "cohort_association_available": True,
            "cohort_association_direction": prior.get("association_direction", ""),
            "effect_summary": prior.get("effect_size", ""),
            "model_importance_available": bool(prior.get("model_importance") or prior.get("importance_rank")),
            "importance_summary": prior.get("model_importance") or prior.get("importance_rank") or "",
            "source_artifact": prior.get("source_artifact", ""),
            "runtime_prior_status": runtime_status,
            "runtime_use": runtime_use,
            "eligible_for_query_expansion": eligible_query,
            "eligible_for_retrieval_rerank": eligible_rerank,
        })
    return matches


def build_safe_clinical_packet(context: dict[str, Any]) -> dict[str, Any]:
    source = context.get("agent2_clinical_evidence") or context.get("clinical_evidence_packet") or context
    source_forbidden = find_forbidden_v2_6_fields(source)
    if source_forbidden:
        raise ValueError(f"forbidden clinical source fields: {source_forbidden}")
    packet = {key: source.get(key) for key in CLINICAL_KEYS if key in source}
    packet["case_id"] = str(context.get("case_id", ""))
    patient_factors = build_patient_factor_statuses(packet)
    priors = load_cohort_factor_prior_registry(context.get("cohort_factor_prior_registry_path"))
    matches = match_patient_factors_to_cohort_priors(patient_factors, priors)
    present_ids = [
        str(item["patient_factor_id"]) for item in patient_factors
        if item.get("patient_factor_status") == "present"
    ]
    continuous_ids = [
        str(item["patient_factor_id"]) for item in patient_factors
        if item.get("patient_factor_status") == "continuous_value_available"
    ]
    query_ids = [
        str(item["factor_id"]) for item in matches
        if item.get("eligible_for_query_expansion")
    ]
    rerank_ids = [
        str(item["factor_id"]) for item in matches
        if item.get("eligible_for_retrieval_rerank")
    ]
    packet["patient_factor_statuses"] = patient_factors
    packet["clinical_evidence_items"] = [
        {
            "evidence_id": f"CLINICAL:{item['source_column']}",
            "evidence_domain": "CLINICAL",
            "canonical_concept_id": item["patient_factor_id"],
            "status": item["patient_factor_status"],
            "source_field_ids": list(item["source_field_ids"]),
            "source_table": item["source_table"],
            "projection_rule_id": item["projection_rule_id"],
            "patient_specific": True,
            "score_derived": False,
            "label_derived": False,
        }
        for item in patient_factors
    ]
    packet["patient_factor_matches"] = matches
    packet["present_clinical_factor_ids"] = sorted(set(present_ids))
    packet["absent_clinical_factor_ids"] = sorted({
        str(item["patient_factor_id"]) for item in patient_factors
        if item.get("patient_factor_status") == "absent"
    })
    packet["unknown_clinical_factor_ids"] = sorted({
        str(item["patient_factor_id"]) for item in patient_factors
        if item.get("patient_factor_status") == "unknown"
    })
    packet["missing_clinical_factor_ids"] = sorted({
        str(item["patient_factor_id"]) for item in patient_factors
        if item.get("patient_factor_status") == "missing"
    })
    packet["continuous_value_available_ids"] = sorted(set(continuous_ids))
    packet["categorical_value_available_ids"] = sorted({
        str(item["patient_factor_id"]) for item in patient_factors
        if item.get("patient_factor_status") == "categorical_value_available"
    })
    packet["unit_unconfirmed_feature_ids"] = sorted({
        str(item["patient_factor_id"]) for item in patient_factors
        if item.get("unit_confirmation_status") != "NOT_APPLICABLE"
        and item.get("agent_interpretation_allowed") is False
        and item.get("value_available") is True
    })
    packet["clinical_data_provenance_confirmed"] = True
    packet["clinical_data_provenance_source"] = CLINICAL_DATA_PROVENANCE_SOURCE
    packet["hospital_reference_range_confirmed"] = False
    packet["hospital_range_claim_allowed"] = False
    packet["patient_specific_confounder_ids"] = sorted({
        factor for factor in present_ids
        if factor in {"digoxin_use", "antiarrhythmic_use", "tricyclic_antidepressant_use", "st_t_affecting_drug_use"}
    })
    packet["hidden_risk_signal_ids"] = sorted({
        factor for factor in present_ids
        if factor in {"diabetes", "hypertension", "smoking", "stroke_history", "peripheral_artery_disease", "family_history_chd"}
    })
    available_count = sum(
        1 for item in patient_factors
        if item["source_column"] in FORMAL_CLINICAL_FEATURES and item["value_available"]
    )
    packet["clinical_payload_projection_coverage"] = round(available_count / len(FORMAL_CLINICAL_FEATURES), 6)
    packet["clinical_payload_projection_limitations"] = (
        [] if available_count == len(FORMAL_CLINICAL_FEATURES)
        else ["one_or_more_formal_clinical_fields_missing"]
    )
    packet["continuous_query_expansion_factor_ids"] = sorted(set(continuous_ids) & set(query_ids))
    packet["cohort_supported_patient_factor_ids"] = sorted(set(query_ids) | set(rerank_ids))
    packet["query_expansion_candidate_factor_ids"] = sorted(set(query_ids))
    packet["cohort_prior_limitations"] = [
        "cohort priors are train-only retrieval priors, not patient facts or causal claims"
    ] if matches else ["no eligible train-only cohort prior matched this packet"]
    forbidden = find_forbidden_v2_6_fields(packet)
    if forbidden:
        raise ValueError(f"forbidden clinical packet fields: {forbidden}")
    return packet


AGENT2_STAGE_A_MODEL_CONTEXT_KEYS = {
    "patient_level_feature_contributions",
    "feature_contribution_method",
    "clinical_hidden_fn_like_factors",
    "clinical_false_positive_like_factors",
    "agent2_clinical_evidence_summary",
    "patient_factor_matches",
    "cohort_supported_patient_factor_ids",
    "query_expansion_candidate_factor_ids",
    "continuous_query_expansion_factor_ids",
    "cohort_prior_limitations",
    "hidden_risk_signal_ids",
}


def _agent2_factor_status_sets(packet: dict[str, Any]) -> dict[str, set[str]]:
    present = set(map(str, packet.get("present_clinical_factor_ids", [])))
    absent = set(map(str, packet.get("absent_clinical_factor_ids", [])))
    missing = set(map(str, packet.get("missing_clinical_factor_ids", [])))
    unknown = set(map(str, packet.get("unknown_clinical_factor_ids", [])))
    continuous = set(map(str, packet.get("continuous_value_available_ids", [])))
    categorical = set(map(str, packet.get("categorical_value_available_ids", [])))
    if not categorical:
        categorical = {
            str(item.get("patient_factor_id"))
            for item in packet.get("patient_factor_statuses", [])
            if str(item.get("patient_factor_status")) == "categorical_value_available"
        }
    directional_continuous = set(map(str, packet.get("continuous_query_expansion_factor_ids", [])))
    confounders = set(map(str, packet.get("patient_specific_confounder_ids", [])))
    all_visible = present | absent | categorical | continuous | missing | unknown
    return {
        "binary_present_ids": present,
        "binary_absent_ids": absent,
        "categorical_available_ids": categorical,
        "continuous_available_ids": continuous,
        "continuous_direction_interpretable_ids": directional_continuous,
        "missing_ids": missing,
        "unknown_ids": unknown,
        "missing_unknown_ids": missing | unknown,
        "patient_specific_confounder_ids": confounders,
        "all_patient_visible_factor_ids": all_visible,
    }


def build_agent2_stage_a_blind_packet(context: dict[str, Any]) -> dict[str, Any]:
    """Patient clinical/lab evidence packet for Agent2-A.

    Agent2-A is the blind clinical evidence pass. Cohort-prior matches,
    model-attribution fields, and downstream retrieval expansion hints are
    retained for deterministic post-pass context, but are not shown to the LLM.
    """
    packet = build_safe_clinical_packet(context)
    facts = []
    for item in packet.get("patient_factor_statuses", []):
        if not isinstance(item, dict):
            continue
        fact = {
            "factor_id": str(item.get("patient_factor_id", "")),
            "status": str(item.get("patient_factor_status", "unknown")),
        }
        if item.get("value_available"):
            fact["value"] = item.get("patient_value")
        if item.get("unit"):
            fact["unit"] = item.get("unit")
        facts.append(fact)
    return {
        "patient_facts": facts,
        "clinical_payload_projection_coverage": packet.get("clinical_payload_projection_coverage"),
        "clinical_payload_projection_limitations": list(packet.get("clinical_payload_projection_limitations", [])),
    }


def _branch_relative_factor_directions(packet: dict[str, Any]) -> dict[str, str]:
    """Return only high-confidence HIGHER/LOWER directions from patient facts."""
    directions: dict[str, str] = {}
    for item in packet.get("patient_factor_statuses", []):
        if not isinstance(item, dict):
            continue
        factor_id = str(item.get("patient_factor_id", ""))
        status = str(item.get("patient_factor_status", ""))
        if factor_id in BRANCH_POLARITY_BINARY_RISK_FACTOR_IDS:
            if status == "present":
                directions[factor_id] = "HIGHER"
            elif status == "absent":
                directions[factor_id] = "LOWER"
            continue
        value_map = BRANCH_POLARITY_CATEGORICAL_VALUES.get(factor_id)
        if status == "categorical_value_available" and value_map:
            value = str(item.get("patient_value", "")).strip().lower()
            if value in value_map:
                directions[factor_id] = value_map[value]
    return directions


def _expected_discrepancy_evidence_field(head: str, clinical_direction: str) -> str | None:
    # Only factors with an explicit, high-confidence patient-level direction
    # may be moved between the supporting and contradicting lists. Contextual
    # factors such as sex deliberately have no hard-coded polarity and must
    # remain where the clinical evidence auditor placed them.
    if clinical_direction not in {"HIGHER", "LOWER"}:
        return None
    if head == "UNDERESTIMATION":
        return "supporting_factor_ids" if clinical_direction == "HIGHER" else "contradicting_factor_ids"
    if head == "OVERESTIMATION":
        return "contradicting_factor_ids" if clinical_direction == "HIGHER" else "supporting_factor_ids"
    return None


def _stance_from_directional_lists(supporting: list[str], contradicting: list[str]) -> str:
    if supporting and contradicting:
        return "MIXED"
    if supporting:
        return "SUPPORTS_DISCREPANCY"
    if contradicting:
        return "OPPOSES_DISCREPANCY"
    return "INSUFFICIENT"


def _patient_factor_value_map(packet: dict[str, Any]) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in packet.get("patient_factor_statuses", []):
        if not isinstance(item, dict):
            continue
        if str(item.get("patient_factor_status", "")) != "categorical_value_available":
            continue
        factor_id = str(item.get("patient_factor_id", ""))
        value = str(item.get("patient_value", "")).strip().lower()
        if factor_id and value:
            values[factor_id] = value
    return values


def _deduplicate_exact_semantic_evidence(
    out: dict[str, Any],
    packet: dict[str, Any],
) -> list[dict[str, str]]:
    """Remove only exact same-value aliases from directional evidence lists."""
    values = _patient_factor_value_map(packet)
    deduplications: list[dict[str, str]] = []
    for canonical_id, alias_id in EXACT_DUPLICATE_CLINICAL_EVIDENCE_GROUPS:
        canonical_value = values.get(canonical_id, "")
        alias_value = values.get(alias_id, "")
        if not canonical_value or canonical_value != alias_value:
            continue
        canonical_field = next(
            (field for field in ("supporting_factor_ids", "contradicting_factor_ids") if canonical_id in out[field]),
            "",
        )
        alias_field = next(
            (field for field in ("supporting_factor_ids", "contradicting_factor_ids") if alias_id in out[field]),
            "",
        )
        if not canonical_field or not alias_field:
            continue
        if canonical_field != alias_field:
            # The same patient fact cannot safely support and contradict the
            # same hypothesis.  Remove both instead of choosing a direction.
            out[canonical_field] = [item for item in out[canonical_field] if item != canonical_id]
            out[alias_field] = [item for item in out[alias_field] if item != alias_id]
            deduplications.append({
                "canonical_factor_id": canonical_id,
                "alias_factor_id": alias_id,
                "normalized_value": canonical_value,
                "action": "remove_both_conflicting_exact_duplicates",
                "reason": "same_patient_fact_opposite_direction_conflict",
            })
            continue
        out[alias_field] = [item for item in out[alias_field] if item != alias_id]
        deduplications.append({
            "canonical_factor_id": canonical_id,
            "alias_factor_id": alias_id,
            "normalized_value": canonical_value,
            "action": f"retain_{canonical_id}_remove_{alias_id}",
            "reason": "exact_same_value_semantic_duplicate",
        })
    return deduplications


def _sanitize_minimal_agent2_raw(
    raw: dict[str, Any],
    packet: dict[str, Any],
    selected_discrepancy_head: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the seven-field Raw contract against visible patient facts.

    This is deliberately local: factor legality, duplicate removal and list
    limits are not a second LLM task.
    """
    sets = _agent2_factor_status_sets(packet)
    visible = sets["all_patient_visible_factor_ids"]
    allowed = {
        # Binary present/absent status is not intrinsically supporting or
        # contradicting: its role reverses between UNDER and OVER review.
        "supporting_factor_ids": sets["binary_present_ids"] | sets["binary_absent_ids"] | sets["categorical_available_ids"] | sets["continuous_direction_interpretable_ids"],
        "contradicting_factor_ids": sets["binary_present_ids"] | sets["binary_absent_ids"] | sets["categorical_available_ids"] | sets["continuous_direction_interpretable_ids"],
        "confounder_factor_ids": sets["patient_specific_confounder_ids"],
        "critical_missing_factor_ids": sets["missing_unknown_ids"],
    }
    out = copy.deepcopy(raw)
    removed: list[dict[str, str]] = []
    for field, allowed_ids in allowed.items():
        kept: list[str] = []
        for item in raw.get(field, []) or []:
            factor = str(item)
            if factor in visible and factor in allowed_ids and factor not in kept and len(kept) < 5:
                kept.append(factor)
            else:
                removed.append({"field": field, "factor_id": factor, "reason": "not_visible_not_allowed_duplicate_or_over_limit"})
        out[field] = kept
    overlap = set(out["supporting_factor_ids"]) & set(out["contradicting_factor_ids"])
    if overlap:
        out["contradicting_factor_ids"] = [x for x in out["contradicting_factor_ids"] if x not in overlap]
        removed.extend({"field": "contradicting_factor_ids", "factor_id": x, "reason": "supporting_contradicting_overlap"} for x in sorted(overlap))

    head = str(selected_discrepancy_head).strip().upper()
    polarity_directions = _branch_relative_factor_directions(packet)
    polarity_normalizations: list[dict[str, str]] = []
    if head in {"UNDERESTIMATION", "OVERESTIMATION"}:
        original_supporting = list(out["supporting_factor_ids"])
        original_contradicting = list(out["contradicting_factor_ids"])
        normalized = {"supporting_factor_ids": [], "contradicting_factor_ids": []}
        for source_field, factors in (
            ("supporting_factor_ids", original_supporting),
            ("contradicting_factor_ids", original_contradicting),
        ):
            for factor in factors:
                expected_field = _expected_discrepancy_evidence_field(head, polarity_directions.get(factor, ""))
                target_field = expected_field or source_field
                if factor not in normalized[target_field] and len(normalized[target_field]) < 5:
                    normalized[target_field].append(factor)
                elif factor not in normalized[target_field]:
                    removed.append({"field": source_field, "factor_id": factor, "reason": "branch_polarity_target_over_limit"})
                if expected_field and expected_field != source_field:
                    polarity_normalizations.append({
                        "factor_id": factor,
                        "clinical_direction": polarity_directions[factor],
                        "selected_discrepancy_head": head,
                        "from_field": source_field,
                        "to_field": expected_field,
                        "reason": "high_confidence_branch_relative_polarity",
                    })
        out.update(normalized)
        out["evidence_stance"] = _stance_from_directional_lists(
            out["supporting_factor_ids"], out["contradicting_factor_ids"]
        )
        if not out["supporting_factor_ids"] and not out["contradicting_factor_ids"]:
            out["evidence_sufficiency"] = "INSUFFICIENT"
        if polarity_normalizations:
            out["concise_rationale"] = (
                f"Branch-relative polarity normalization retained "
                f"{len(out['supporting_factor_ids'])} supporting and "
                f"{len(out['contradicting_factor_ids'])} contradicting patient factors "
                f"for the locked {head} review."
            )
    semantic_deduplications = _deduplicate_exact_semantic_evidence(out, packet)
    out["evidence_stance"] = _stance_from_directional_lists(
        out["supporting_factor_ids"], out["contradicting_factor_ids"]
    )
    if not out["supporting_factor_ids"] and not out["contradicting_factor_ids"]:
        out["evidence_sufficiency"] = "INSUFFICIENT"
    rationale = " ".join(str(out.get("concise_rationale", "")).split())
    out["concise_rationale"] = rationale[:320]
    return out, {
        "removed_items": removed,
        "polarity_normalizations": polarity_normalizations,
        "semantic_deduplications": semantic_deduplications,
        "branch_polarity_rule_count": len(polarity_directions),
        "selected_discrepancy_head": head,
        "visible_patient_fact_count": len(visible),
        "patient_fact_loss_count": 0,
    }


def _normalize_minimal_agent2_deterministic_confounder_field(
    raw: dict[str, Any],
    packet: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Restore only the locally owned Agent2 confounder list when omitted.

    ``confounder_factor_ids`` is not an open-ended LLM judgment: its complete
    legal value is derived from the patient-status packet's explicit present
    drug/confounder flags.  Some JSON models omit an empty array even when the
    prompt requires it.  Treating that harmless omission as a whole-case
    failure makes batch execution brittle, while asking the model to repair it
    would add a redundant network request.

    No other Raw field is defaulted here.  Missing supporting/contradicting
    evidence, stance, sufficiency, missingness, or rationale still fails the
    strict Raw contract because those fields carry semantic judgment.
    """

    normalized = copy.deepcopy(raw)
    if "confounder_factor_ids" in normalized and normalized["confounder_factor_ids"] is not None:
        return normalized, []

    deterministic_value = sorted(
        _agent2_factor_status_sets(packet)["patient_specific_confounder_ids"]
    )[:5]
    raw_value = normalized.get("confounder_factor_ids")
    normalized["confounder_factor_ids"] = deterministic_value
    return normalized, [{
        "field_path": "confounder_factor_ids",
        "raw_value": raw_value,
        "normalized_value": deterministic_value,
        "normalization_rule": "restore_locally_owned_patient_specific_confounder_ids",
    }]


def _minimal_agent2_committed_output(
    *, context: dict[str, Any], raw: dict[str, Any], prediction: dict[str, Any], fact_loss_count: int,
) -> dict[str, Any]:
    head = str(prediction.get("selected_discrepancy_head", ""))
    alert = bool(prediction.get("underestimation_alert")) if head == "UNDERESTIMATION" else bool(prediction.get("overestimation_alert"))
    return {
        "schema_version": V2_6_SCHEMA_VERSION,
        "selected_discrepancy_head": head,
        "head_score": float(prediction.get("discrepancy_risk_score")),
        "risk_band": str(prediction.get("discrepancy_risk_band", "")),
        "alert_state": "ALERT" if alert else "NO_ALERT",
        "evidence_stance": str(raw["evidence_stance"]),
        "supporting_factor_ids": list(raw["supporting_factor_ids"]),
        "contradicting_factor_ids": list(raw["contradicting_factor_ids"]),
        "confounder_factor_ids": list(raw["confounder_factor_ids"]),
        "critical_missing_factor_ids": list(raw["critical_missing_factor_ids"]),
        "evidence_sufficiency": str(raw["evidence_sufficiency"]),
        "concise_rationale": str(raw["concise_rationale"]),
        "patient_fact_loss_count": int(fact_loss_count),
        "provenance_status": "LOCKED_MODEL_AND_VALIDATED_LLM",
    }


class ClinicalLabEvidenceAgent(LLMAgentBase):
    """Agent2: clinical discrepancy agent."""

    agent_id = "agent2"
    agent_name = "Agent2_Clinical_Discrepancy_Agent"
    prompt_key = "agent2"
    schema_key = "agent2"

    def build_payload(self, context: dict[str, Any]) -> dict[str, Any]:
        return {"clinical_evidence_packet": build_agent2_stage_a_blind_packet(context)}

    def _run_minimal_contract(self, context: dict[str, Any]) -> AgentRunResult:
        if self.client is None:
            raise RuntimeError(f"{self.__class__.__name__} requires an injected LLM client")
        assert_agent2_discrepancy_models_locked(context.get("agent2_discrepancy_model_config"))
        packet = build_safe_clinical_packet(context)
        stage_packet = build_agent2_stage_a_blind_packet(context)
        prediction = predict_agent2_discrepancy_from_locked_artifacts(
            {**context, "agent2_clinical_evidence": packet, "clinical_evidence_packet": packet},
            context.get("agent2_discrepancy_model_config"),
        )
        head = str(prediction.get("selected_discrepancy_head", ""))
        alert = bool(prediction.get("underestimation_alert")) if head == "UNDERESTIMATION" else bool(prediction.get("overestimation_alert"))
        stage_schema_key = "agent2.evidence_pass"
        stage_schema = V2_6_AGENT_STAGE_SCHEMAS[stage_schema_key]
        user_payload = {
            "agent_id": self.agent_id,
            "case_id": str(context.get("case_id", "")),
            "runtime_version": "v2.9",
            "schema_version": V2_6_SCHEMA_VERSION,
            "schema_hash": stable_hash(stage_schema),
            "selected_discrepancy_head": head,
            "locked_model_summary": {"risk_band": prediction.get("discrepancy_risk_band"), "alert_state": "ALERT" if alert else "NO_ALERT"},
            "input": {"clinical_evidence_packet": stage_packet},
        }
        started = time.perf_counter()
        raw_text, attempt_count, network_retry_errors = self._complete_with_network_retry(V2_6_AGENT_EVIDENCE_PROMPTS["agent2"], user_payload)
        raw, parse_error = parse_json_object(raw_text)
        if parse_error:
            raise ValueError("Agent2 minimal Raw contract returned invalid JSON")
        raw_before_normalization = copy.deepcopy(raw)
        raw, runtime_metadata_actions = remove_matching_runtime_stage_metadata(
            raw,
            expected_stage_schema_key=stage_schema_key,
        )
        raw, enum_actions = normalize_schema_enum_values("agent2", raw)
        raw, deterministic_confounder_actions = _normalize_minimal_agent2_deterministic_confounder_field(
            raw,
            packet,
        )
        validate_agent_stage_output(stage_schema_key, raw)
        sanitized, sanitation = _sanitize_minimal_agent2_raw(raw, packet, head)
        validate_agent_stage_output(stage_schema_key, sanitized)
        result = _minimal_agent2_committed_output(context=context, raw=sanitized, prediction=prediction, fact_loss_count=sanitation["patient_fact_loss_count"])
        validate_v2_6_agent_output("agent2", result)
        trace = {
            "agent_id": "agent2", "agent_name": self.agent_name, "case_id": str(context.get("case_id", "")),
            "active_contract_version": "agent2_minimal_scientific_contract_v1", "stage_schema_key": stage_schema_key,
            "prompt_hash": hashlib.sha256(V2_6_AGENT_EVIDENCE_PROMPTS["agent2"].encode("utf-8")).hexdigest(),
            "schema_hash": stable_hash(stage_schema), "model_inference_config_hash": self.model_inference_config_hash(), "transport_execution_config_hash": self.transport_execution_config_hash(),
            "attempt_count": attempt_count, "retry_request_count": attempt_count - 1, "network_retry_errors": network_retry_errors,
            "raw_response": raw_text, "stage_a_user_payload": user_payload, "raw_semantic_output": raw,
            "parsed_json_before_normalization": raw_before_normalization,
            "normalization_actions": runtime_metadata_actions + enum_actions + deterministic_confounder_actions,
            "sanitized_raw_semantic_output": sanitized, "raw_sanitation": sanitation,
            "parsed_output": result, "schema_repair_attempted": False, "schema_repair_count": 0,
            "semantic_factor_llm_clarification_disabled": True, "latency_seconds": round(time.perf_counter() - started, 6),
        }
        self.last_trace = trace
        if self.trace_sink:
            self.trace_sink(trace)
        return AgentRunResult(output=result, trace=trace)

    def run_with_trace(self, context: dict[str, Any]) -> AgentRunResult:
        return self._run_minimal_contract(context)
