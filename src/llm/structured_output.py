from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from src.agents.schemas import (
    V2_6_AGENT_SCHEMAS,
    V2_6_NESTED_SCHEMAS,
    build_nested_schema_contract,
    build_agent_stage_schema_contract,
)


TYPE_NORMALIZER_VERSION = "v2_6_exact_boolean_string_normalizer_v1"
ECG_PATTERN_ALIAS_NORMALIZER_VERSION = "v2_6_agent1_ecg_pattern_alias_normalizer_v1"
FIELD_NAME_NORMALIZER_VERSION = "v2_8_exact_nonsemantic_field_name_normalizer_v1"
RUNTIME_METADATA_NORMALIZER_VERSION = "v2_8_exact_runtime_metadata_normalizer_v1"
RUNTIME_AGENT_IDENTITY_NORMALIZER_VERSION = "v2_8_runtime_owned_agent_identity_v1"
FIELD_LEVEL_SCHEMA_REPAIR_PROMPT = """
You are a field-level JSON enum repair step.
Return only one legal value for the requested field.
Do not return a specific ECG finding.
Do not modify any other field.
Do not add explanations.
Return exactly one JSON object in this format:
{"field_path":"ecg_evidence_pattern","corrected_value":"NON_SPECIFIC_OR_CONFOUNDED"}
"""
AGENT1_PATTERN_STRENGTH_RELATION_REPAIR_PROMPT = """
You are an Agent1 cross-field schema relation repair step.
Repair only ecg_evidence_pattern and/or ecg_directional_strength.
Use only the active Agent1 schema contract and the related evidence-id fields.
Do not add medical evidence.
Do not remove medical evidence.
Do not use CAG labels, outcomes, scores, or case-specific rules.
Allowed compatibility:
- ISCHEMIA_SUPPORTIVE: ecg_directional_strength must be WEAK, MODERATE, or STRONG.
- NO_CLEAR_ISCHEMIC_EVIDENCE: ecg_directional_strength must be NONE.
- NON_SPECIFIC_OR_CONFOUNDED: ecg_directional_strength must be NONE or WEAK.
- MIXED: ecg_directional_strength must be WEAK or MODERATE.
- NOT_EVALUABLE: ecg_directional_strength must be NONE.
If ecg_evidence_pattern is NO_CLEAR_ISCHEMIC_EVIDENCE and ischemic_evidence_ids is empty, repair ecg_directional_strength to NONE.
If ischemic_evidence_ids is not empty and strength is directional, prefer repairing the pattern to ISCHEMIA_SUPPORTIVE rather than erasing the directional evidence.
Return exactly one JSON object:
{"ecg_evidence_pattern":"NO_CLEAR_ISCHEMIC_EVIDENCE","ecg_directional_strength":"NONE"}
"""
TECHNICAL_SCHEMA_REPAIR_PROMPT = """
You are a technical JSON schema repair step.
Correct only JSON syntax, field names and field types.
Do not add new medical evidence.
Do not remove medical evidence.
Do not change the substantive clinical meaning.
Return exactly one JSON object.
Boolean fields must be native JSON true/false, not strings, numbers, yes/no, or prose.
For Agent3 claim_evidence_chain, correct only the nested structure. Each item must contain claim_id, claim_type, claim_text, patient_evidence_ids, model_evidence_ids, cohort_prior_factor_ids, external_card_ids, support_status, counterevidence_card_ids, and uncertainty. Do not change proposal_direction or proposal_strength.
Do not wrap the JSON in Markdown code fences.
Do not add any text before or after the JSON.
"""


def normalize_llm_structured_output_types(
    parsed_json: dict[str, Any],
    boolean_fields: Iterable[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Conservatively normalize exact boolean strings for active schema fields only.

    This function deliberately does not guess yes/no, 1/0, null, empty strings,
    missing fields, or explanatory prose. It only converts exact true/false
    strings after trimming surrounding whitespace.
    """
    normalized = deepcopy(parsed_json)
    actions: list[dict[str, Any]] = []
    for field in boolean_fields:
        if field not in normalized:
            continue
        value = normalized[field]
        if isinstance(value, bool):
            continue
        if isinstance(value, str):
            stripped = value.strip()
            lowered = stripped.lower()
            if lowered in {"true", "false"}:
                converted = lowered == "true"
                normalized[field] = converted
                actions.append({
                    "field_path": field,
                    "from_type": "string",
                    "from_value": value,
                    "to_type": "boolean",
                    "to_value": converted,
                    "rule": "EXACT_BOOLEAN_STRING_NORMALIZATION",
                    "normalizer_version": TYPE_NORMALIZER_VERSION,
                })
    return normalized, actions


_EXACT_NONSEMANTIC_FIELD_ALIASES = {
    # Agent3 distinguishes patient evidence (contradicting_evidence_ids) from
    # external knowledge cards (contradictory_card_ids).  The provider has
    # emitted the grammatically parallel but non-canonical card-key spelling
    # below.  Renaming this exact key preserves the list verbatim and performs
    # no medical inference.
    ("agent3", "contradicting_card_ids"): "contradictory_card_ids",
}


def normalize_exact_nonsemantic_field_names(
    agent_id: str,
    parsed_json: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Rename exact, audited field aliases without changing their values.

    If both spellings are present with different values, leave the object
    untouched so the strict validator fails closed instead of guessing which
    value is authoritative.
    """

    normalized = deepcopy(parsed_json)
    actions: list[dict[str, Any]] = []
    for (owner, alias), canonical in _EXACT_NONSEMANTIC_FIELD_ALIASES.items():
        if owner != agent_id or alias not in normalized:
            continue
        if canonical in normalized and normalized[canonical] != normalized[alias]:
            continue
        value = normalized.pop(alias)
        normalized.setdefault(canonical, value)
        actions.append({
            "field_path": alias,
            "normalized_field_path": canonical,
            "normalization_rule": "EXACT_NONSEMANTIC_FIELD_NAME_ALIAS",
            "normalizer_version": FIELD_NAME_NORMALIZER_VERSION,
        })
    return normalized, actions


def remove_matching_runtime_stage_metadata(
    parsed_json: dict[str, Any],
    *,
    expected_stage_schema_key: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Remove an exact routing-key echo that is not part of medical output.

    A mismatching value is deliberately retained and rejected by the strict
    schema validator.  Only an exact echo of the runtime-owned stage key is
    removed.
    """

    normalized = deepcopy(parsed_json)
    if normalized.get("stage_schema_key") != expected_stage_schema_key:
        return normalized, []
    normalized.pop("stage_schema_key")
    return normalized, [{
        "field_path": "stage_schema_key",
        "raw_value": expected_stage_schema_key,
        "normalization_rule": "REMOVE_MATCHING_RUNTIME_ROUTING_METADATA",
        "normalizer_version": RUNTIME_METADATA_NORMALIZER_VERSION,
    }]


def normalize_runtime_owned_agent_identity(
    agent_id: str,
    parsed_json: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Set the non-medical agent identity from the trusted runtime route.

    ``agent_name`` is static envelope metadata: it identifies the Python agent
    instance that made the request and is not a medical conclusion produced by
    the model.  Providers may shorten or restyle this string even when the
    semantic JSON is otherwise valid.  Preserve the original response in the
    raw trace, but make the validated object use the canonical schema identity.

    This function deliberately does not alter ``case_id``, ``schema_version``,
    or any clinical/action field.  Those fields retain their existing strict
    validation semantics.
    """

    if agent_id not in V2_6_AGENT_SCHEMAS:
        raise KeyError(f"unknown agent_id for runtime identity: {agent_id}")
    normalized = deepcopy(parsed_json)
    canonical = V2_6_AGENT_SCHEMAS[agent_id]["agent_name"]
    raw_value = normalized.get("agent_name")
    if raw_value == canonical:
        return normalized, []
    normalized["agent_name"] = canonical
    return normalized, [{
        "field_path": "agent_name",
        "raw_value": raw_value,
        "normalized_value": canonical,
        "normalization_rule": "INJECT_RUNTIME_OWNED_AGENT_IDENTITY",
        "normalizer_version": RUNTIME_AGENT_IDENTITY_NORMALIZER_VERSION,
        "medical_semantics_changed": False,
    }]


_ECG_EVIDENCE_PATTERN_CANONICAL = {
    "ischemia_supportive": "ISCHEMIA_SUPPORTIVE",
    "no_clear_ischemic_evidence": "NO_CLEAR_ISCHEMIC_EVIDENCE",
    "non_specific_or_confounded": "NON_SPECIFIC_OR_CONFOUNDED",
    "mixed": "MIXED",
    "not_evaluable": "NOT_EVALUABLE",
}

_ECG_EVIDENCE_PATTERN_ALIASES = {
    "nonspecific_st_t_change": "NON_SPECIFIC_OR_CONFOUNDED",
    "non_specific_st_t_change": "NON_SPECIFIC_OR_CONFOUNDED",
    "nonspecific_st_t_abnormality": "NON_SPECIFIC_OR_CONFOUNDED",
    "non_specific_st_t_abnormality": "NON_SPECIFIC_OR_CONFOUNDED",
    "nonspecific_or_confounded": "NON_SPECIFIC_OR_CONFOUNDED",
    "non_specific_or_confounding": "NON_SPECIFIC_OR_CONFOUNDED",
}

_ECG_PATTERN_ALIAS_EVIDENCE_IDS = {
    "nonspecific_st_t_change": "nonspecific_st_t_change",
    "non_specific_st_t_change": "nonspecific_st_t_change",
    "nonspecific_st_t_abnormality": "nonspecific_st_t_change",
    "non_specific_st_t_abnormality": "nonspecific_st_t_change",
}

_FIELD_ENUM_ALIASES = {
    ("agent1", "ecg_directional_strength"): {
        "none": "NONE",
        "neutral": "NONE",
        "no_directional_strength": "NONE",
        "no_direction": "NONE",
        "weak": "WEAK",
        "moderate": "MODERATE",
        "strong": "STRONG",
    },
    ("agent1", "ecg_evidence_pattern"): _ECG_EVIDENCE_PATTERN_ALIASES,
}

_AGENT1_PATTERN_STRENGTH_ALLOWED = {
    "ISCHEMIA_SUPPORTIVE": {"WEAK", "MODERATE", "STRONG"},
    "NO_CLEAR_ISCHEMIC_EVIDENCE": {"NONE"},
    "NON_SPECIFIC_OR_CONFOUNDED": {"NONE", "WEAK"},
    "MIXED": {"WEAK", "MODERATE"},
    "NOT_EVALUABLE": {"NONE"},
}


def _canonicalize_enum_token(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def normalize_schema_enum_values(
    agent_id: str,
    parsed_json: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Normalize enum fields using active schema values and exact field aliases."""
    normalized = deepcopy(parsed_json)
    actions: list[dict[str, Any]] = []
    schema = V2_6_AGENT_SCHEMAS[agent_id]
    for field, allowed_values in schema.get("enums", {}).items():
        raw_value = normalized.get(field)
        if not isinstance(raw_value, str):
            continue
        token = _canonicalize_enum_token(raw_value)
        allowed_by_token = {
            _canonicalize_enum_token(value): value
            for value in allowed_values
        }
        mapped = allowed_by_token.get(token)
        rule = "SCHEMA_ENUM_CASE_NORMALIZATION"
        if mapped is None:
            mapped = _FIELD_ENUM_ALIASES.get((agent_id, field), {}).get(token)
            rule = "FIELD_SPECIFIC_ENUM_ALIAS_NORMALIZATION"
        if mapped is None or mapped == raw_value:
            continue
        normalized[field] = mapped
        if (
            agent_id == "agent1"
            and field == "ecg_evidence_pattern"
            and rule == "FIELD_SPECIFIC_ENUM_ALIAS_NORMALIZATION"
            and mapped == "NON_SPECIFIC_OR_CONFOUNDED"
        ):
            evidence_ids = normalized.get("non_specific_or_confounding_evidence_ids")
            if not isinstance(evidence_ids, list):
                evidence_ids = []
            if token not in evidence_ids:
                evidence_ids.append(token)
            normalized["non_specific_or_confounding_evidence_ids"] = evidence_ids
        actions.append({
            "field_path": field,
            "raw_value": raw_value,
            "normalized_value": mapped,
            "normalization_rule": rule,
        })
    return normalized, actions


def normalize_agent1_ecg_evidence_pattern(
    parsed_json: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Normalize field-specific Agent1 coarse ECG pattern aliases.

    This only maps a small exact whitelist. It does not infer from broad terms
    such as st_depression, t_wave_inversion, ischemia, abnormal_ecg, or normal.
    """
    normalized = deepcopy(parsed_json)
    actions: list[dict[str, Any]] = []
    raw_value = normalized.get("ecg_evidence_pattern")
    if not isinstance(raw_value, str):
        return normalized, actions
    token = _canonicalize_enum_token(raw_value)
    mapped = _ECG_EVIDENCE_PATTERN_CANONICAL.get(token) or _ECG_EVIDENCE_PATTERN_ALIASES.get(token)
    if not mapped:
        return normalized, actions
    if raw_value == mapped:
        return normalized, actions
    normalized["ecg_evidence_pattern"] = mapped
    evidence_id = _ECG_PATTERN_ALIAS_EVIDENCE_IDS.get(token, "")
    if evidence_id:
        existing = normalized.get("non_specific_or_confounding_evidence_ids")
        if isinstance(existing, list):
            if evidence_id not in existing:
                existing = list(existing) + [evidence_id]
            normalized["non_specific_or_confounding_evidence_ids"] = existing
    actions.append({
        "field_path": "ecg_evidence_pattern",
        "raw_value": raw_value,
        "normalized_value": mapped,
        "normalization_rule": "EXACT_ECG_PATTERN_ALIAS_NORMALIZATION",
        "evidence_id_transferred": evidence_id,
        "normalizer_version": ECG_PATTERN_ALIAS_NORMALIZER_VERSION,
    })
    return normalized, actions


def build_agent1_pattern_strength_relation_repair_payload(
    *,
    parsed_json: dict[str, Any],
    relation_error: str,
) -> dict[str, Any]:
    """Build an auditable payload for Agent1 pattern/strength relation repair."""
    schema = V2_6_AGENT_SCHEMAS["agent1"]
    return {
        "repair_scope": "agent1_pattern_strength_relation",
        "relation_error": relation_error,
        "allowed_ecg_evidence_pattern": list(schema["enums"]["ecg_evidence_pattern"]),
        "allowed_ecg_directional_strength": list(schema["enums"]["ecg_directional_strength"]),
        "compatibility_matrix": {
            key: sorted(value)
            for key, value in _AGENT1_PATTERN_STRENGTH_ALLOWED.items()
        },
        "current_values": {
            "ecg_evidence_pattern": parsed_json.get("ecg_evidence_pattern"),
            "ecg_directional_strength": parsed_json.get("ecg_directional_strength"),
            "ischemic_evidence_ids": parsed_json.get("ischemic_evidence_ids"),
            "non_specific_or_confounding_evidence_ids": parsed_json.get("non_specific_or_confounding_evidence_ids"),
            "hidden_ischemia_warning": parsed_json.get("hidden_ischemia_warning"),
            "false_positive_mimic_warning": parsed_json.get("false_positive_mimic_warning"),
        },
        "repair_constraints": [
            "Repair only ecg_evidence_pattern and/or ecg_directional_strength.",
            "Do not add or remove medical evidence IDs.",
            "Do not use scores, labels, outcomes, or case-specific rules.",
            "NO_CLEAR_ISCHEMIC_EVIDENCE requires ecg_directional_strength=NONE.",
            "If ischemic_evidence_ids is empty, do not keep a directional STRONG/MODERATE/WEAK strength for NO_CLEAR_ISCHEMIC_EVIDENCE.",
        ],
    }


def repair_agent1_pattern_strength_relation(
    parsed_json: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Deterministically repair Agent1 pattern/strength compatibility.

    The repair is intentionally conservative and auditable. It only touches the
    two relation fields and only when the pair violates the active schema
    matrix. It never invents evidence IDs and never uses scores or labels.
    """
    repaired = deepcopy(parsed_json)
    pattern = repaired.get("ecg_evidence_pattern")
    strength = repaired.get("ecg_directional_strength")
    allowed_strengths = _AGENT1_PATTERN_STRENGTH_ALLOWED.get(pattern)
    if allowed_strengths is None or strength in allowed_strengths:
        return repaired, []

    ischemic_ids = repaired.get("ischemic_evidence_ids")
    if not isinstance(ischemic_ids, list):
        ischemic_ids = []
    non_specific_ids = repaired.get("non_specific_or_confounding_evidence_ids")
    if not isinstance(non_specific_ids, list):
        non_specific_ids = []

    reason = "AGENT1_PATTERN_STRENGTH_RELATION_REPAIR"
    if pattern == "NO_CLEAR_ISCHEMIC_EVIDENCE":
        if ischemic_ids and strength in {"WEAK", "MODERATE", "STRONG"}:
            repaired["ecg_evidence_pattern"] = "ISCHEMIA_SUPPORTIVE"
            repaired["ecg_directional_strength"] = strength
            rule = "NO_CLEAR_WITH_ISCHEMIC_IDS_REPAIR_PATTERN_TO_ISCHEMIA_SUPPORTIVE"
        else:
            repaired["ecg_directional_strength"] = "NONE"
            rule = "NO_CLEAR_WITHOUT_ISCHEMIC_IDS_REPAIR_STRENGTH_TO_NONE"
    elif pattern == "NOT_EVALUABLE":
        repaired["ecg_directional_strength"] = "NONE"
        rule = "NOT_EVALUABLE_REPAIR_STRENGTH_TO_NONE"
    elif pattern == "NON_SPECIFIC_OR_CONFOUNDED":
        repaired["ecg_directional_strength"] = "WEAK" if non_specific_ids or repaired.get("false_positive_mimic_warning") else "NONE"
        rule = "NON_SPECIFIC_REPAIR_STRENGTH_TO_ALLOWED_NONE_OR_WEAK"
    elif pattern == "MIXED":
        repaired["ecg_directional_strength"] = "MODERATE" if ischemic_ids and non_specific_ids else "WEAK"
        rule = "MIXED_REPAIR_STRENGTH_TO_ALLOWED_WEAK_OR_MODERATE"
    elif pattern == "ISCHEMIA_SUPPORTIVE":
        repaired["ecg_directional_strength"] = "WEAK"
        rule = "ISCHEMIA_SUPPORTIVE_REPAIR_STRENGTH_TO_WEAK"
    else:
        return repaired, []

    changed_fields = []
    for field in ("ecg_evidence_pattern", "ecg_directional_strength"):
        if repaired.get(field) != parsed_json.get(field):
            changed_fields.append({
                "field_path": field,
                "raw_value": parsed_json.get(field),
                "normalized_value": repaired.get(field),
            })
    if not changed_fields:
        return repaired, []
    return repaired, [{
        "field_path": "ecg_evidence_pattern/ecg_directional_strength",
        "raw_pattern": pattern,
        "raw_strength": strength,
        "normalized_pattern": repaired.get("ecg_evidence_pattern"),
        "normalized_strength": repaired.get("ecg_directional_strength"),
        "ischemic_evidence_ids": list(ischemic_ids),
        "non_specific_or_confounding_evidence_ids": list(non_specific_ids),
        "normalization_rule": rule,
        "repair_reason": reason,
        "changed_fields": changed_fields,
    }]


def normalize_agent3_claim_evidence_chain(
    parsed_json: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Deterministically repair legacy Agent3 claim_evidence_chain shapes.

    This function only converts shape. It does not add evidence IDs, does not
    change proposal fields, and does not invent external-card support.
    """
    normalized = deepcopy(parsed_json)
    chain = normalized.get("claim_evidence_chain")
    if chain is None:
        return normalized, []
    actions: list[dict[str, Any]] = []
    if isinstance(chain, dict):
        chain = [chain]
        actions.append({
            "field_path": "claim_evidence_chain",
            "raw_value_type": "dict",
            "normalized_value_type": "list",
            "normalization_rule": "WRAP_SINGLE_CLAIM_OBJECT",
        })
    if not isinstance(chain, list):
        return normalized, actions
    item_schema = V2_6_NESTED_SCHEMAS["agent3.claim_evidence_chain.item"]
    list_fields = set(item_schema["lists"])
    allowed_status = set(item_schema["enums"]["support_status"])
    repaired_chain = []
    changed = False
    for index, item in enumerate(chain, start=1):
        if isinstance(item, str):
            repaired = {
                "claim_id": f"claim_{index}",
                "claim_type": "structured_claim",
                "claim_text": item,
                "patient_evidence_ids": [],
                "model_evidence_ids": [],
                "cohort_prior_factor_ids": [],
                "external_card_ids": [],
                "support_status": "unsupported",
                "counterevidence_card_ids": [],
                "uncertainty": "",
            }
            repaired_chain.append(repaired)
            changed = True
            actions.append({
                "field_path": f"claim_evidence_chain[{index - 1}]",
                "raw_value_type": "str",
                "normalized_value_type": "object",
                "normalization_rule": "LEGACY_STRING_CLAIM_TO_NESTED_OBJECT",
            })
            continue
        if not isinstance(item, dict):
            repaired_chain.append(item)
            continue
        original = deepcopy(item)
        repaired = {
            "claim_id": str(item.get("claim_id") or f"claim_{index}"),
            "claim_type": str(item.get("claim_type") or "structured_claim"),
            "claim_text": str(item.get("claim_text") or item.get("claim") or ""),
            "patient_evidence_ids": item.get("patient_evidence_ids"),
            "model_evidence_ids": item.get("model_evidence_ids"),
            "cohort_prior_factor_ids": item.get("cohort_prior_factor_ids"),
            "external_card_ids": item.get("external_card_ids"),
            "support_status": item.get("support_status"),
            "counterevidence_card_ids": item.get("counterevidence_card_ids"),
            "uncertainty": str(item.get("uncertainty") or item.get("evidence") or ""),
        }
        for field in list_fields:
            field_was_present = field in item
            value = repaired[field]
            if isinstance(value, list):
                continue
            if isinstance(value, str) and value.strip().lower() in {"", "none"}:
                repaired[field] = []
            elif value is None and not field_was_present:
                repaired[field] = []
            else:
                repaired[field] = value
        status = repaired.get("support_status")
        if isinstance(status, str):
            status_token = status.strip().lower().replace(" ", "_").replace("-", "_")
            repaired["support_status"] = status_token if status_token in allowed_status else status
        if repaired.get("support_status") not in allowed_status:
            repaired["support_status"] = "unsupported"
        if repaired != original:
            changed = True
            actions.append({
                "field_path": f"claim_evidence_chain[{index - 1}]",
                "raw_value_type": "object",
                "normalized_value_type": "object",
                "normalization_rule": "LEGACY_CLAIM_OBJECT_TO_NESTED_SCHEMA",
            })
        repaired_chain.append(repaired)
    if changed:
        normalized["claim_evidence_chain"] = repaired_chain
    return normalized, actions


def build_schema_repair_payload(
    *,
    parsed_json: dict[str, Any],
    schema_error: str,
    target_schema: dict[str, Any] | None = None,
    required_output_fields: list[str] | None = None,
    boolean_fields: list[str] | None = None,
    stage_schema_key: str | None = None,
) -> dict[str, Any]:
    payload = {
        "repair_scope": "technical_schema_repair",
        "schema_error": schema_error,
        "parsed_json_before_repair": parsed_json,
        "target_schema": target_schema or {},
        "required_output_fields": required_output_fields or [],
        "boolean_fields": boolean_fields or [],
        "stage_schema_key": stage_schema_key,
        "constraints": [
            "Correct only JSON syntax, field names and field types.",
            "Do not add new medical evidence.",
            "Do not remove medical evidence.",
            "Do not change the substantive clinical meaning.",
            "Return exactly one JSON object.",
        ],
    }
    if stage_schema_key:
        payload["stage_schema_contract"] = build_agent_stage_schema_contract(stage_schema_key)
    if stage_schema_key == "agent1.evidence_pass":
        payload["constraints"].extend([
            "Correct only Agent1-A Evidence Pass JSON.",
            "Do not add Agent1-B audit fields.",
            "Do not add score consistency fields.",
            "Do not return full Agent1 committed schema.",
            "Return exactly the evidence-pass JSON object.",
        ])
    if stage_schema_key == "agent2.evidence_pass":
        payload["constraints"].extend([
            "Correct only Agent2-A Evidence Pass JSON.",
            "Do not add Agent2-B audit fields.",
            "Do not add score consistency fields.",
            "Do not return full Agent2 committed schema.",
            "Return exactly the evidence-pass JSON object.",
        ])
    payload["repair_constraints"] = list(payload["constraints"])
    return payload


def build_field_level_repair_payload(
    *,
    field_path: str,
    invalid_value: Any,
    allowed_values: list[str],
    field_definition: str,
) -> dict[str, Any]:
    return {
        "repair_scope": "field_level_schema_repair",
        "field_path": field_path,
        "invalid_value": invalid_value,
        "allowed_values": allowed_values,
        "field_definition": field_definition,
        "constraints": [
            "Return only one legal value for this field.",
            "Do not return a specific ECG finding.",
            "Do not modify any other field.",
            "Do not add explanation.",
        ],
    }
