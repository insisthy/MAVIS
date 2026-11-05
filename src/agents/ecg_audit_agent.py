from __future__ import annotations

import hashlib
import json
import math
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from src.llm.json_parser import parse_json_object
from src.runtime.runtime_fingerprints import build_runtime_fingerprint_components
from src.llm.structured_output import (
    FIELD_LEVEL_SCHEMA_REPAIR_PROMPT,
    TECHNICAL_SCHEMA_REPAIR_PROMPT,
    TYPE_NORMALIZER_VERSION,
    build_field_level_repair_payload,
    build_agent1_pattern_strength_relation_repair_payload,
    build_schema_repair_payload,
    normalize_agent1_ecg_evidence_pattern,
    normalize_exact_nonsemantic_field_names,
    normalize_llm_structured_output_types,
    normalize_runtime_owned_agent_identity,
    normalize_schema_enum_values,
    remove_matching_runtime_stage_metadata,
    repair_agent1_pattern_strength_relation,
)

from .prompts import V2_6_AGENT_EVIDENCE_PROMPTS, V2_6_AGENT_PROMPTS
from .schemas import (
    V2_6_AGENT_SCHEMAS,
    V2_6_AGENT_STAGE_SCHEMAS,
    V2_6_SAFE_BOUNDARY_CHECK,
    V2_6_SCHEMA_VERSION,
    build_agent_raw_prompt_schema,
    find_forbidden_v2_6_fields,
    normalize_v2_6_safety_boundary_check,
    validate_agent_stage_output,
    validate_v2_6_agent_output,
)


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


EXPLICIT_ECG_MIMIC_ALLOWLIST = {
    "left_ventricular_hypertrophy_mimic",
    "lvh_strain_secondary_st_t",
    "bundle_branch_block_mimic",
    "drug_or_electrolyte_st_t_mimic",
}


def has_explicit_patient_level_ecg_mimic_support(evidence: dict[str, Any]) -> bool:
    """Return true only for traceable patient-level ECG false-positive mimic evidence.

    Non-specific ECG changes, absent ischemic evidence, missing evidence, or
    uncertainty alone must not be interpreted as a DOWN/overstated signal.
    """

    if evidence.get("false_positive_mimic_warning") is not True:
        return False
    candidate_ids = {
        str(item)
        for item in (
            list(evidence.get("possible_ecg_mimic_terms", []) or [])
            + list(evidence.get("non_specific_or_confounding_evidence_ids", []) or [])
        )
        if str(item)
    }
    return bool(candidate_ids & EXPLICIT_ECG_MIMIC_ALLOWLIST)


@dataclass(frozen=True)
class AgentRunResult:
    output: dict[str, Any]
    trace: dict[str, Any]


class LLMAgentBase:
    """v2.6 通用 LLM 执行器；不包含 ECG、临床、RAG 或 eligibility 医学规则。"""

    agent_id = ""
    agent_name = ""
    prompt_key = ""
    schema_key = ""

    def __init__(
        self,
        client: Any | None = None,
        *,
        max_retries: int = 3,
        retry_backoff_seconds: float = 5.0,
        retry_backoff_multiplier: float = 2.0,
        trace_sink: Any | None = None,
    ) -> None:
        self.client = client
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self.retry_backoff_multiplier = max(1.0, float(retry_backoff_multiplier))
        self.trace_sink = trace_sink
        self.last_trace: dict[str, Any] = {}

    @property
    def prompt(self) -> str:
        return V2_6_AGENT_PROMPTS[self.prompt_key]

    @property
    def schema(self) -> dict[str, Any]:
        return V2_6_AGENT_SCHEMAS[self.schema_key]

    def build_payload(self, context: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def validate_payload(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dict")
        forbidden = find_forbidden_v2_6_fields(payload)
        if forbidden:
            raise ValueError(f"forbidden payload fields: {forbidden}")

    def _runtime_fingerprint_components(self) -> dict[str, Any]:
        config = getattr(self.client, "config", None)
        execution = {
            "connect_timeout_seconds": getattr(config, "connect_timeout_seconds", None),
            "read_timeout_seconds": getattr(config, "read_timeout_seconds", None),
            "max_technical_retries_per_call": getattr(self.client, "max_technical_retries_per_call", 0),
            "technical_retry_backoff_seconds": getattr(self.client, "retry_backoff_seconds", []),
            "agent1_3_streaming_enabled": getattr(self.client, "agent1_3_streaming_enabled", False),
            "agent1_3_max_stream_duration_seconds": getattr(self.client, "agent1_3_max_stream_duration_seconds", None),
            "agent1_3_thinking_mode": getattr(self.client, "agent1_3_thinking_mode", None),
            "agent1_3_reasoning_effort": getattr(self.client, "agent1_3_reasoning_effort", None),
            "agent4_streaming_enabled": getattr(self.client, "agent4_streaming_enabled", False),
            "agent4_max_output_tokens": getattr(self.client, "agent4_max_output_tokens", None),
            "agent4_max_total_attempts": getattr(self.client, "agent4_max_total_attempts", None),
            "agent4_thinking_mode": getattr(self.client, "agent4_thinking_mode", None),
        }
        wrapped = getattr(self.client, "client", self.client)
        return build_runtime_fingerprint_components(config, execution, client_class=type(wrapped).__name__, transport_adapter=type(self.client).__name__)

    def model_inference_config_hash(self) -> str:
        return str(self._runtime_fingerprint_components()["model_inference_config_hash"])

    def transport_execution_config_hash(self) -> str:
        return str(self._runtime_fingerprint_components()["transport_execution_config_hash"])

    def _complete_with_network_retry(self, prompt: str, payload: dict[str, Any]) -> tuple[str, int, list[str]]:
        """Call the LLM client with transport retry only.

        Schema repair remains a separate, at-most-once downstream step.  This
        helper is intentionally small so Agent1/Agent2 evidence-pass calls use
        the same retry semantics as the shared base path without changing
        prompts, provider, model, temperature, or timeout.
        """
        # RawResponsePersistingClient is the single technical-retry owner in
        # formal runtime.  Keep this fallback only for unwrapped local clients.
        retry_limit = 0 if getattr(self.client, "owns_transport_retries", False) else self.max_retries
        attempts = 0
        errors: list[str] = []
        for attempt in range(retry_limit + 1):
            attempts = attempt + 1
            try:
                return self.client.complete(prompt, payload), attempts, errors
            except Exception as exc:
                errors.append(f"{type(exc).__name__}:{exc}")
                if attempt >= retry_limit:
                    raise
                delay = self.retry_backoff_seconds * self.retry_backoff_multiplier**attempt
                if delay:
                    time.sleep(delay)
        raise RuntimeError("LLM completion retry loop exhausted unexpectedly")

    def run_with_trace(self, context: dict[str, Any]) -> AgentRunResult:
        if self.client is None:
            raise RuntimeError(f"{self.__class__.__name__} requires an injected LLM client")
        payload = self.build_payload(context)
        self.validate_payload(payload)
        user_payload = {
            "agent_id": self.agent_id,
            "case_id": str(context.get("case_id", "")),
            "runtime_version": "v2.6",
            "schema_version": V2_6_SCHEMA_VERSION,
            "schema_hash": stable_hash(self.schema),
            "input": payload,
        }
        raw = ""
        parsed: dict[str, Any] = {}
        normalization: dict[str, Any] = {}
        parsed_before_type_normalization: dict[str, Any] = {}
        schema_errors_before_repair: list[str] = []
        schema_repair_attempted = False
        schema_repair_count = 0
        repaired_json: dict[str, Any] | None = None
        attempts = 0
        started_all = time.perf_counter()
        retry_limit = 0 if getattr(self.client, "owns_transport_retries", False) else self.max_retries
        for attempt in range(retry_limit + 1):
            attempts = attempt + 1
            try:
                raw = self.client.complete(self.prompt, user_payload)
                parsed, parse_error = parse_json_object(raw)
                if parse_error:
                    raise ValueError("LLM output is not valid JSON")
                parsed_before_type_normalization = dict(parsed)
                parsed, agent_identity_actions = normalize_runtime_owned_agent_identity(
                    self.agent_id, parsed
                )
                parsed, field_name_actions = normalize_exact_nonsemantic_field_names(
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
                    agent_identity_actions + field_name_actions + type_actions + enum_actions
                )
                normalization["parsed_json_before_type_normalization"] = parsed_before_type_normalization
                try:
                    validate_v2_6_agent_output(self.agent_id, parsed)
                except Exception as validation_exc:
                    schema_errors_before_repair.append(str(validation_exc))
                    schema_repair_attempted = True
                    schema_repair_count = 1
                    raw_repair_schema = build_agent_raw_prompt_schema(self.agent_id)
                    repaired_raw = self.client.complete(
                        TECHNICAL_SCHEMA_REPAIR_PROMPT,
                        build_schema_repair_payload(
                            parsed_json=parsed_before_type_normalization,
                            schema_error=str(validation_exc),
                            target_schema=raw_repair_schema,
                            required_output_fields=list(raw_repair_schema["required"]),
                            boolean_fields=list(raw_repair_schema.get("booleans", [])),
                        ),
                    )
                    repaired_json, repaired_parse_error = parse_json_object(repaired_raw)
                    if repaired_parse_error:
                        raise ValueError("technical schema repair returned invalid JSON") from validation_exc
                    repaired_json, repair_identity_actions = normalize_runtime_owned_agent_identity(
                        self.agent_id, repaired_json
                    )
                    repaired_json, repair_actions = normalize_llm_structured_output_types(
                        repaired_json,
                        self.schema.get("booleans", []),
                    )
                    repaired_json, repair_field_name_actions = normalize_exact_nonsemantic_field_names(
                        self.agent_id, repaired_json
                    )
                    repaired_json, repair_enum_actions = normalize_schema_enum_values(self.agent_id, repaired_json)
                    nested_actions = []
                    repaired_json, repair_boundary = normalize_v2_6_safety_boundary_check(repaired_json)
                    normalization["type_normalization_actions"].extend(repair_actions)
                    normalization["type_normalization_actions"].extend(repair_identity_actions)
                    normalization["type_normalization_actions"].extend(repair_field_name_actions)
                    normalization["type_normalization_actions"].extend(repair_enum_actions)
                    normalization["type_normalization_actions"].extend(nested_actions)
                    normalization["schema_repair_boundary_normalization"] = repair_boundary
                    validate_v2_6_agent_output(self.agent_id, repaired_json)
                    parsed = repaired_json
                break
            except Exception:
                if attempt >= retry_limit:
                    raise
                delay = self.retry_backoff_seconds * self.retry_backoff_multiplier**attempt
                if delay:
                    time.sleep(delay)
        trace = {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "case_id": str(context.get("case_id", "")),
            "schema_version": V2_6_SCHEMA_VERSION,
            "prompt_hash": hashlib.sha256(self.prompt.encode("utf-8")).hexdigest(),
            "schema_hash": stable_hash(self.schema),
            "model_inference_config_hash": self.model_inference_config_hash(),
            "transport_execution_config_hash": self.transport_execution_config_hash(),
            "attempt_count": attempts,
            "retry_request_count": attempts - 1,
            "raw_response": raw,
            "parsed_output": parsed,
            "normalization": normalization,
            "parsed_json_before_normalization": parsed_before_type_normalization,
            "normalization_actions": normalization.get("type_normalization_actions", []),
            "normalized_json": parsed,
            "schema_errors_before_repair": schema_errors_before_repair,
            "schema_repair_attempted": schema_repair_attempted,
            "schema_repair_count": schema_repair_count,
            "repaired_json": repaired_json,
            "final_schema_status": "pass",
            "latency_seconds": round(time.perf_counter() - started_all, 6),
            "fallback_used": False,
        }
        self.last_trace = trace
        if self.trace_sink:
            self.trace_sink(trace)
        return AgentRunResult(output=parsed, trace=trace)

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        return self.run_with_trace(context).output


ALLOWED_ECG_KEYS = {
    "ecg_summary", "machine_interpretation_raw", "machine_statement_list",
    "machine_statement_list_json", "ecg_machine_interpretation_summary",
    "ecg_statement_summary", "ecg_structured_parameter_summary",
    "ecg_qc_summary", "ecg_qc_flags", "basic_qc_pass_flag",
    "physical_abs_max_estimated_mV", "large_amplitude_warning_flag",
    "uncommon_sig_len_flag", "machine_interpretation_missing_flag",
    "conversion_status", "fs_hz", "n_leads", "sig_len", "duration_sec",
    "heart_rate_bpm", "p_duration_ms", "pr_interval_ms",
    "qrs_duration_ms", "qt_interval_ms", "qtc_interval_ms",
    "p_axis_deg", "qrs_axis_deg", "t_axis_deg", "rv5_mV", "sv1_mV",
    "machine_statement_count", "st_segment_summary",
    "t_wave_summary", "agent1_ecg_evidence_summary",
    "waveform_uncertainty", "important_leads", "important_time_windows",
    "attribution_method", "attribution_stability", "embedding_reference",
    "waveform_explanation_summary", "waveform_explanation_unavailable",
}

FORMAL_ECG_FEATURES = (
    "heart_rate_bpm", "p_duration_ms", "pr_interval_ms", "qrs_duration_ms",
    "qt_interval_ms", "qtc_interval_ms", "p_axis_deg", "qrs_axis_deg",
    "t_axis_deg", "rv5_mV", "sv1_mV", "basic_qc_pass_flag",
    "physical_abs_max_estimated_mV", "large_amplitude_warning_flag",
    "uncommon_sig_len_flag", "machine_statement_count",
    "machine_interpretation_missing_flag",
)

ECG_QC_FIELDS = {
    "basic_qc_pass_flag": "basic_qc",
    "large_amplitude_warning_flag": "large_amplitude_warning",
    "uncommon_sig_len_flag": "uncommon_signal_length",
}

LEGACY_ECG_CONCEPT_RULES_DISABLED = {
    "st_depression": ("st depression", "st-depression", "st段压低", "st 段压低", "缺血"),
    "st_elevation": ("st elevation", "st-elevation", "st段抬高", "st 段抬高"),
    "t_wave_abnormality": ("t wave", "t-wave", "t波", "t 波", "倒置"),
    "pathologic_q_wave": ("q wave", "q-wave", "病理性q", "病理性 q"),
    "left_bundle_branch_block": ("lbbb", "left bundle", "左束支"),
    "right_bundle_branch_block": ("rbbb", "right bundle", "右束支"),
    "paced_rhythm": ("paced rhythm", "pacemaker", "ventricular paced", "起搏"),
    "left_ventricular_hypertrophy": ("lvh", "hypertrophy", "左室肥厚"),
    "atrial_fibrillation": ("atrial fibrillation", "afib", "房颤"),
    "poor_r_wave_progression": ("poor r", "r wave progression", "r波递增不良"),
    "nonspecific_st_t_change": ("nonspecific st", "st-t", "非特异"),
    "early_repolarization": ("early repolarization", "早期复极"),
    "j_point_elevation_pattern": ("j point elevation", "j-point elevation", "j点抬高"),
    "persistent_st_elevation": ("persistent st elevation", "持续st抬高"),
    "prior_myocardial_infarction": ("old myocardial infarction", "prior myocardial infarction", "old mi", "既往心肌梗死"),
    "left_ventricular_aneurysm": ("left ventricular aneurysm", "ventricular aneurysm", "室壁瘤"),
    "q_wave_abnormality": ("q wave abnormality", "abnormal q wave", "异常q波"),
    "wellens_specific_pattern": ("wellens", "wellens-like"),
    "dynamic_t_wave_change": ("dynamic t wave", "dynamic t-wave", "new t wave"),
    "serial_ecg_evidence": ("serial ecg", "compared with prior ecg", "dynamic change"),
    "avr_diffuse_st_pattern": ("avr elevation with diffuse st depression", "diffuse st depression with avr"),
    "posterior_occlusion_pattern": ("posterior occlusion", "posterior infarction"),
    "reciprocal_st_depression": ("reciprocal st depression", "reciprocal change"),
    "brugada_pattern": ("brugada", "brugada-like"),
    "hyperacute_t_wave": ("hyperacute t wave", "hyperacute t-wave"),
}

LEGACY_ECG_MIMIC_RULES_DISABLED = {
    "left_ventricular_hypertrophy_mimic": ("lvh", "左室肥厚"),
    "bundle_branch_block_mimic": ("lbbb", "rbbb", "束支传导阻滞"),
    "drug_or_electrolyte_st_t_mimic": ("digoxin", "electrolyte", "洋地黄", "电解质"),
    "poor_quality_or_baseline_wander": ("artifact", "noise", "baseline", "伪差", "噪声"),
}


# v2.7 final-design repair: keep the active deterministic concept mapper in
# readable UTF-8 and avoid mapping generic ischemia wording directly to a
# specific ST-depression concept.  The legacy block above is kept only as
# historical source context; these definitions are the active dictionaries used
# by build_ecg_retrieval_concepts below.
ECG_CONCEPT_RULES = {
    "st_depression": (
        "st depression", "st-depression", "st segment depression",
        "st-segment depression", "st段压低", "st 段压低",
    ),
    "st_elevation": (
        "st elevation", "st-elevation", "st segment elevation",
        "st-segment elevation", "st段抬高", "st 段抬高",
    ),
    "generic_ischemic_statement": (
        "ischemia", "ischemic change", "myocardial ischemia", "缺血", "心肌缺血",
    ),
    "t_wave_abnormality": ("t wave", "t-wave", "t波", "t 波", "倒置", "inversion"),
    "pathologic_q_wave": ("q wave", "q-wave", "pathologic q", "pathological q", "病理性q", "病理性 q"),
    "left_bundle_branch_block": ("lbbb", "left bundle", "left bundle branch block", "左束支"),
    "right_bundle_branch_block": ("rbbb", "right bundle", "right bundle branch block", "右束支"),
    "paced_rhythm": ("paced rhythm", "pacemaker", "ventricular paced", "起搏"),
    "left_ventricular_hypertrophy": ("lvh", "hypertrophy", "left ventricular hypertrophy", "左室肥厚"),
    "atrial_fibrillation": ("atrial fibrillation", "afib", "房颤"),
    "poor_r_wave_progression": ("poor r", "r wave progression", "r波递增不良"),
    "nonspecific_st_t_change": ("nonspecific st", "non-specific st", "st-t", "非特异"),
    "early_repolarization": ("early repolarization", "早期复极"),
    "j_point_elevation_pattern": ("j point elevation", "j-point elevation", "j点抬高"),
    "persistent_st_elevation": ("persistent st elevation", "persistent st-segment elevation", "持续st抬高"),
    "prior_myocardial_infarction": ("old myocardial infarction", "prior myocardial infarction", "old mi", "既往心肌梗死"),
    "left_ventricular_aneurysm": ("left ventricular aneurysm", "ventricular aneurysm", "室壁瘤"),
    "q_wave_abnormality": ("q wave abnormality", "abnormal q wave", "异常q波", "异常 q 波"),
    "wellens_specific_pattern": ("wellens", "wellens-like"),
    "dynamic_t_wave_change": ("dynamic t wave", "dynamic t-wave", "new t wave"),
    "serial_ecg_evidence": ("serial ecg", "compared with prior ecg", "dynamic change"),
    "avr_diffuse_st_pattern": ("avr elevation with diffuse st depression", "diffuse st depression with avr"),
    "posterior_occlusion_pattern": ("posterior occlusion", "posterior infarction"),
    "reciprocal_st_depression": ("reciprocal st depression", "reciprocal change"),
    "brugada_pattern": ("brugada", "brugada-like"),
    "hyperacute_t_wave": ("hyperacute t wave", "hyperacute t-wave"),
}

ECG_MIMIC_RULES = {
    "left_ventricular_hypertrophy_mimic": ("lvh", "left ventricular hypertrophy", "左室肥厚"),
    "bundle_branch_block_mimic": ("lbbb", "rbbb", "bundle branch block", "束支传导阻滞"),
    "drug_or_electrolyte_st_t_mimic": ("digoxin", "electrolyte", "洋地黄", "电解质"),
    "poor_quality_or_baseline_wander": ("artifact", "noise", "baseline", "伪差", "噪声"),
}

ECG_CONCEPT_RULES.update({
    "st_depression": ECG_CONCEPT_RULES["st_depression"] + ("st段压低", "st 段压低", "ST段压低"),
    "st_elevation": ECG_CONCEPT_RULES["st_elevation"] + ("st段抬高", "st 段抬高", "ST段抬高"),
    "t_wave_abnormality": ECG_CONCEPT_RULES["t_wave_abnormality"] + ("t波", "T波", "t波倒置", "T波倒置"),
    "left_ventricular_hypertrophy": ECG_CONCEPT_RULES["left_ventricular_hypertrophy"] + ("左室肥厚",),
})
ECG_MIMIC_RULES.update({
    "left_ventricular_hypertrophy_mimic": ECG_MIMIC_RULES["left_ventricular_hypertrophy_mimic"] + ("左室肥厚",),
})

ECG_NEGATION_TERMS = (
    "no", "not", "without", "absent", "negative for", "free of",
    "未见", "无", "没有", "未发现", "未提示", "否认", "不支持",
)
ECG_UNCERTAINTY_TERMS = (
    "possible", "borderline", "probable", "suspected", "cannot exclude",
    "questionable", "?", "可能", "疑似", "可疑", "不排除",
)
ECG_ENGLISH_NEGATION_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\babsent\b",
        r"\bnegative\s+for\b", r"\bfree\s+of\b",
    )
)
ECG_CLAUSE_BOUNDARY_RE = re.compile(r"[，,；;。.!?:：]")


def _flatten_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(_flatten_text(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(_flatten_text(v) for v in value)
    return "" if value is None else str(value)


def _iter_machine_statement_texts(packet: dict[str, Any]) -> list[str]:
    statements: list[str] = []

    def add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, dict):
            text = " ".join(str(v) for v in value.values() if str(v).strip())
            if text.strip():
                statements.append(text.strip())
            return
        if isinstance(value, list):
            for item in value:
                add(item)
            return
        text = str(value).strip()
        if not text:
            return
        for part in re.split(r"(?<=[。；;.!?])\s+", text):
            if part.strip():
                statements.append(part.strip())

    add(packet.get("machine_statement_list"))
    raw_json = packet.get("machine_statement_list_json")
    if raw_json:
        try:
            add(json.loads(raw_json) if isinstance(raw_json, str) else raw_json)
        except Exception:
            add(raw_json)
    if statements:
        return statements
    for key in (
        "machine_interpretation_raw", "ecg_machine_interpretation_summary",
        "ecg_statement_summary", "st_segment_summary", "t_wave_summary",
        "ecg_summary",
    ):
        add(packet.get(key))
    return statements


def _local_status_for_term(statement: str, term: str) -> str | None:
    text = statement.lower()
    needle = term.lower()
    matches = list(re.finditer(re.escape(needle), text))
    if not matches:
        return None
    statuses: list[str] = []
    for match in matches:
        start = match.start()
        end = match.end()
        before = text[max(0, start - 32):start]
        after = text[end:min(len(text), end + 16)]
        local = f"{before} {needle} {after}"
        if any(token.lower() in local for token in ECG_UNCERTAINTY_TERMS):
            statuses.append("unknown")
            continue
        negated = False
        for pattern in ECG_ENGLISH_NEGATION_PATTERNS:
            for neg in pattern.finditer(before):
                between = before[neg.end():]
                if not ECG_CLAUSE_BOUNDARY_RE.search(between):
                    negated = True
                    break
            if negated:
                break
        if not negated:
            for token in ("未见", "无", "没有", "未发现", "未提示", "否认", "不支持"):
                token_l = token.lower()
                idx = before.rfind(token_l)
                if idx < 0:
                    continue
                between = before[idx + len(token_l):]
                if not ECG_CLAUSE_BOUNDARY_RE.search(between) and len(between) <= 8:
                    negated = True
                    break
        statuses.append("absent" if negated else "present")
    return _aggregate_concept_status(statuses)


def _aggregate_concept_status(statuses: list[str]) -> str:
    unique = set(statuses)
    if "unknown" in unique:
        return "unknown"
    if "present" in unique and "absent" in unique:
        return "unknown"
    if "present" in unique:
        return "present"
    if "absent" in unique:
        return "absent"
    return "unknown"


def _patient_visible_ecg_concept_allowlist(packet: dict[str, Any]) -> set[str]:
    visible = set(map(str, packet.get("ecg_positive_concept_ids", [])))
    visible |= set(map(str, packet.get("ecg_negative_concept_ids", [])))
    visible |= set(map(str, packet.get("ecg_uncertain_concept_ids", [])))
    visible |= set(map(str, packet.get("possible_ecg_mimic_terms", [])))
    visible |= {
        str(item.get("canonical_concept_id"))
        for item in packet.get("ecg_evidence_items", [])
        if item.get("evidence_domain") == "QC" and item.get("canonical_concept_id")
    }
    visible |= {
        "non_diagnostic_ecg_limitation",
        "machine_interpretation_available",
        "machine_interpretation_missing",
    }
    return visible


def validate_agent1_patient_visible_evidence_ids(evidence: dict[str, Any], packet: dict[str, Any]) -> None:
    allowed = _patient_visible_ecg_concept_allowlist(packet)
    illegal: list[str] = []
    for field in ("ischemic_evidence_ids", "non_specific_or_confounding_evidence_ids", "retrieval_concepts"):
        for item in evidence.get(field, []) or []:
            if str(item) not in allowed:
                illegal.append(f"{field}:{item}")
    if illegal:
        raise ValueError(f"illegal_patient_evidence_id:{sorted(set(illegal))}")


def _canonical_ecg_concept_from_auxiliary_item(item: Any) -> str:
    text = str(item).strip()
    if not text:
        return ""
    parts = [part.strip() for part in text.split(":") if part.strip()]
    if len(parts) >= 2 and parts[-1].lower() in {"present", "absent", "unknown", "uncertain"}:
        return parts[-2]
    return text


def sanitize_agent1_auxiliary_evidence_fields(
    evidence: dict[str, Any],
    packet: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize/contain Agent1 auxiliary concept fields without aborting the case.

    Agent1-A's core medical judgment comes from the complete patient ECG packet.
    The list fields remain useful for indexing/RAG, but they must not be a hard
    case-success gate.  This function keeps canonical visible concepts, maps
    deterministic evidence-id strings to canonical concepts, records grounded
    but unmapped text, and removes unsupported concepts from structured fields.
    """

    allowed = _patient_visible_ecg_concept_allowlist(packet)
    raw_text = _flatten_text({
        key: packet.get(key)
        for key in (
            "ecg_summary",
            "machine_interpretation_raw",
            "machine_statement_list",
            "machine_statement_list_json",
            "ecg_machine_interpretation_summary",
            "ecg_statement_summary",
            "st_segment_summary",
            "t_wave_summary",
            "ecg_structured_parameter_summary",
            "ecg_qc_summary",
            "ecg_qc_flags",
        )
    }).lower()
    sanitized = deepcopy(evidence)
    report: dict[str, Any] = {
        "auxiliary_id_error_aborts_case": False,
        "normalized_auxiliary_items": [],
        "unmapped_patient_ecg_evidence": [],
        "unsupported_patient_ecg_evidence_removed": [],
        "grounding_warning": False,
    }
    structured_fields = ("ischemic_evidence_ids", "non_specific_or_confounding_evidence_ids")
    for field in structured_fields:
        kept: list[str] = []
        for item in evidence.get(field, []) or []:
            raw_item = str(item)
            canonical = _canonical_ecg_concept_from_auxiliary_item(raw_item)
            if canonical in allowed:
                kept.append(canonical)
                if canonical != raw_item:
                    report["normalized_auxiliary_items"].append({
                        "field": field,
                        "raw_value": raw_item,
                        "canonical_value": canonical,
                        "rule": "evidence_id_to_visible_canonical_concept",
                    })
                continue
            grounded = bool(raw_item and raw_item.lower().replace("_", " ") in raw_text)
            if grounded:
                report["unmapped_patient_ecg_evidence"].append({
                    "field": field,
                    "raw_value": raw_item,
                    "reason": "grounded_in_patient_ecg_text_but_not_structured_concept",
                })
            else:
                report["unsupported_patient_ecg_evidence_removed"].append({
                    "field": field,
                    "raw_value": raw_item,
                    "canonical_candidate": canonical,
                    "reason": "not_supported_by_visible_ecg_concept_projection_or_raw_text",
                })
                report["grounding_warning"] = True
        sanitized[field] = sorted(dict.fromkeys(kept))
    retrieval_kept: list[str] = []
    for item in evidence.get("retrieval_concepts", []) or []:
        canonical = _canonical_ecg_concept_from_auxiliary_item(item)
        if canonical in allowed:
            retrieval_kept.append(canonical)
            if canonical != str(item):
                report["normalized_auxiliary_items"].append({
                    "field": "retrieval_concepts",
                    "raw_value": str(item),
                    "canonical_value": canonical,
                    "rule": "retrieval_evidence_id_to_visible_canonical_concept",
                })
        elif str(item).lower().replace("_", " ") in raw_text:
            report["unmapped_patient_ecg_evidence"].append({
                "field": "retrieval_concepts",
                "raw_value": str(item),
                "reason": "grounded_text_not_used_as_structured_rag_keyword",
            })
        else:
            report["unsupported_patient_ecg_evidence_removed"].append({
                "field": "retrieval_concepts",
                "raw_value": str(item),
                "canonical_candidate": canonical,
                "reason": "not_supported_by_visible_ecg_concept_projection_or_raw_text",
            })
            report["grounding_warning"] = True
    sanitized["retrieval_concepts"] = sorted(dict.fromkeys(retrieval_kept))
    if report["grounding_warning"] and sanitized.get("ecg_uncertainty") == "LOW":
        sanitized["ecg_uncertainty"] = "MEDIUM"
        report["uncertainty_adjustment"] = "LOW_TO_MEDIUM_DUE_TO_UNSUPPORTED_AUXILIARY_ECG_EVIDENCE"
    else:
        report["uncertainty_adjustment"] = "NONE"
    report["unsupported_patient_fact_reaches_agent3_count"] = 0
    report["unsupported_patient_fact_reaches_rag_count"] = 0
    return sanitized, report


def build_ecg_retrieval_concepts(packet: dict[str, Any]) -> dict[str, list[str]]:
    """Project ECG concepts with statement-local negation/uncertainty scope."""
    statements = _iter_machine_statement_texts(packet)
    if not statements:
        statements = [_flatten_text({
            key: packet.get(key)
            for key in (
                "ecg_summary", "machine_interpretation_raw", "machine_statement_list",
                "machine_statement_list_json", "ecg_machine_interpretation_summary",
                "ecg_statement_summary", "st_segment_summary", "t_wave_summary",
                "ecg_structured_parameter_summary", "ecg_qc_summary", "ecg_qc_flags",
            )
        })]
    positive: list[str] = []
    uncertain: list[str] = []
    negative: list[str] = []
    mimic: list[str] = []
    for concept, terms in ECG_CONCEPT_RULES.items():
        statuses: list[str] = []
        for statement in statements:
            for term in terms:
                status = _local_status_for_term(statement, term)
                if status:
                    statuses.append(status)
        if not statuses:
            continue
        final_status = _aggregate_concept_status(statuses)
        if final_status == "present":
            positive.append(concept)
        elif final_status == "absent":
            negative.append(concept)
        else:
            uncertain.append(concept)
    for concept, terms in ECG_MIMIC_RULES.items():
        statuses: list[str] = []
        for statement in statements:
            for term in terms:
                status = _local_status_for_term(statement, term)
                if status:
                    statuses.append(status)
        if statuses and _aggregate_concept_status(statuses) == "present":
            mimic.append(concept)
    return {
        "ecg_retrieval_concept_ids": sorted(set(positive + uncertain + mimic)),
        "ecg_positive_evidence_terms": sorted(set(positive)),
        "ecg_negative_evidence_terms": sorted(set(negative)),
        "ecg_uncertain_evidence_terms": sorted(set(uncertain)),
        "possible_ecg_mimic_terms": sorted(set(mimic)),
    }


def _value_available(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return text not in {"", "nan", "none", "null", "missing", "unknown"}


def _bool_state(value: Any) -> str:
    if not _value_available(value):
        return "missing"
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return "present"
    if text in {"0", "false", "no", "n"}:
        return "absent"
    return "unknown"


def build_ecg_evidence_items(packet: dict[str, Any]) -> list[dict[str, Any]]:
    """Project deterministic ECG evidence without inventing morphology or thresholds."""
    items: list[dict[str, Any]] = []
    source_table = str(packet.get("_source_table", "data/processed/v2_9/ECG_agent1_input_table_v5.csv"))
    for field in FORMAL_ECG_FEATURES:
        if field not in packet:
            continue
        available = _value_available(packet.get(field))
        if field == "machine_interpretation_missing_flag":
            raw_status = _bool_state(packet.get(field))
            if raw_status == "present":
                status = "present"
                concept_id = "machine_interpretation_missing"
            elif raw_status == "absent":
                status = "present"
                concept_id = "machine_interpretation_available"
            else:
                status = raw_status
                concept_id = "machine_interpretation_availability_unknown"
        elif field in ECG_QC_FIELDS:
            status = _bool_state(packet.get(field))
            concept_id = ECG_QC_FIELDS[field]
        else:
            status = "present" if available else "missing"
            concept_id = f"{field}_measurement"
        items.append({
            "evidence_id": f"ECG:{field}",
            "evidence_domain": "QC" if field in ECG_QC_FIELDS or field == "machine_interpretation_missing_flag" else "ECG",
            "canonical_concept_id": concept_id,
            "concept_status": status,
            "source_field_ids": [field],
            "source_table": source_table,
            "source_value_available": available,
            "projection_method": "deterministic_field_state_projection",
            "projection_rule_id": f"v2_6_ecg_{field}_state_v1",
            "confidence_source": "formal_source_field",
            "patient_specific": True,
            "score_derived": False,
            "label_derived": False,
        })
    concepts = build_ecg_retrieval_concepts(packet)
    for status_key, status in (
        ("ecg_positive_evidence_terms", "present"),
        ("ecg_negative_evidence_terms", "absent"),
        ("ecg_uncertain_evidence_terms", "unknown"),
        ("possible_ecg_mimic_terms", "present"),
    ):
        for concept_id in concepts[status_key]:
            items.append({
                "evidence_id": f"ECG:machine_interpretation:{concept_id}:{status}",
                "evidence_domain": "ECG",
                "canonical_concept_id": concept_id,
                "concept_status": status,
                "source_field_ids": [
                    field for field in ("machine_interpretation_raw", "machine_statement_list_json")
                    if _value_available(packet.get(field))
                ],
                "source_table": source_table,
                "source_value_available": True,
                "projection_method": "deterministic_lexical_concept_projection",
                "projection_rule_id": f"v2_6_ecg_machine_{concept_id}_v1",
                "confidence_source": "machine_interpretation_text",
                "patient_specific": True,
                "score_derived": False,
                "label_derived": False,
            })
    unique: dict[str, dict[str, Any]] = {}
    for item in items:
        unique[item["evidence_id"]] = item
    return [unique[key] for key in sorted(unique)]


def build_safe_ecg_packet(context: dict[str, Any]) -> dict[str, Any]:
    source = context.get("agent1_ecg_evidence") or context.get("ecg_evidence_packet") or context
    source_forbidden = find_forbidden_v2_6_fields(source)
    if source_forbidden:
        raise ValueError(f"forbidden ECG source fields: {source_forbidden}")
    packet = {key: source.get(key) for key in ALLOWED_ECG_KEYS if key in source}
    packet["case_id"] = str(context.get("case_id", ""))
    has_attribution = bool(
        packet.get("waveform_explanation_summary")
        or packet.get("important_leads")
        or packet.get("important_time_windows")
    )
    packet["waveform_explanation_unavailable"] = not has_attribution
    if not has_attribution:
        packet.pop("important_leads", None)
        packet.pop("important_time_windows", None)
    packet.update(build_ecg_retrieval_concepts(packet))
    evidence_items = build_ecg_evidence_items(packet)
    packet["ecg_evidence_items"] = evidence_items
    packet["ecg_positive_concept_ids"] = sorted({
        item["canonical_concept_id"] for item in evidence_items
        if item["concept_status"] == "present" and item["projection_method"] == "deterministic_lexical_concept_projection"
    })
    packet["ecg_negative_concept_ids"] = sorted({
        item["canonical_concept_id"] for item in evidence_items if item["concept_status"] == "absent"
    })
    packet["ecg_uncertain_concept_ids"] = sorted({
        item["canonical_concept_id"] for item in evidence_items if item["concept_status"] == "unknown"
    })
    packet["ecg_missing_concept_ids"] = sorted({
        item["canonical_concept_id"] for item in evidence_items if item["concept_status"] == "missing"
    })
    packet["ecg_qc_status"] = (
        "caution" if any(
            item["evidence_domain"] == "QC"
            and item["canonical_concept_id"] != "basic_qc"
            and item["concept_status"] == "present"
            for item in evidence_items
        )
        else "pass" if any(
            item["canonical_concept_id"] == "basic_qc" and item["concept_status"] == "present"
            for item in evidence_items
        )
        else "unknown"
    )
    available_count = sum(
        1 for item in evidence_items
        if item["projection_method"] == "deterministic_field_state_projection"
        and item["source_value_available"]
    )
    packet["ecg_payload_projection_coverage"] = round(available_count / len(FORMAL_ECG_FEATURES), 6)
    packet["ecg_payload_projection_limitations"] = (
        [] if available_count == len(FORMAL_ECG_FEATURES)
        else ["one_or_more_formal_ecg_fields_missing"]
    )
    forbidden = find_forbidden_v2_6_fields(packet)
    if forbidden:
        raise ValueError(f"forbidden ECG packet fields: {forbidden}")
    return packet


AGENT1_STAGE_A_MODEL_ATTRIBUTION_KEYS = {
    "waveform_uncertainty",
    "important_leads",
    "important_time_windows",
    "attribution_method",
    "attribution_stability",
    "embedding_reference",
    "waveform_explanation_summary",
    "waveform_explanation_unavailable",
}


def build_agent1_stage_a_blind_packet(context: dict[str, Any]) -> dict[str, Any]:
    """Patient ECG evidence packet for Agent1-A.

    Agent1-A is the blind ECG evidence pass. It may read ECG measurements,
    machine statements, QC fields, and deterministic patient-level ECG concept
    projections, but not downstream waveform-model attribution or any score.
    """
    packet = build_safe_ecg_packet(context)
    statements = packet.get("machine_statement_list")
    if not isinstance(statements, list):
        try:
            statements = json.loads(str(packet.get("machine_statement_list_json", "")))
        except Exception:
            statements = []
    machine_text = [str(item) for item in statements if str(item).strip()]
    if not machine_text and str(packet.get("machine_interpretation_raw", "")).strip():
        machine_text = [str(packet["machine_interpretation_raw"])]
    return {
        "machine_statements": machine_text,
        "ecg_parameters": {
            key: packet[key] for key in (
                "heart_rate_bpm", "pr_interval_ms", "qrs_duration_ms",
                "qtc_interval_ms", "qrs_axis_deg", "t_axis_deg",
            ) if packet.get(key) is not None
        },
        "quality": {
            key: packet.get(key) for key in (
                "basic_qc_pass_flag", "large_amplitude_warning_flag",
                "uncommon_sig_len_flag", "machine_interpretation_missing_flag",
            ) if key in packet
        },
        "missingness": {
            "machine_interpretation_missing": bool(packet.get("machine_interpretation_missing_flag", False)),
        },
    }


def _required_locked_float(mapping: dict[str, Any], key: str, error_code: str) -> float:
    value = mapping.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(error_code)
    try:
        number = float(value)
    except Exception as exc:
        raise ValueError(error_code) from exc
    if not math.isfinite(number):
        raise ValueError(error_code)
    return number


def build_ecg_score_consistency_packet(
    context: dict[str, Any],
    evidence_pass: dict[str, Any],
) -> dict[str, Any]:
    ecg_anchor = context.get("ecg_anchor_context", {})
    threshold = ecg_anchor.get("locked_ecg_threshold")
    explanation = {
        key: context.get("agent1_ecg_evidence", {}).get(key)
        for key in (
            "waveform_uncertainty", "waveform_explanation_summary",
            "important_leads", "important_time_windows",
            "attribution_method", "attribution_stability", "embedding_reference",
        )
        if context.get("agent1_ecg_evidence", {}).get(key) is not None
    }
    return {
        "case_id": str(context.get("case_id", "")),
        "locked_ecg_score": ecg_anchor.get("locked_ecg_score"),
        "locked_ecg_threshold": threshold,
        "locked_ecg_risk_level": ecg_anchor.get("locked_ecg_risk_level") or ecg_anchor.get("ecg_baseline_state"),
        "waveform_model_explanation": explanation,
        "evidence_pass": evidence_pass,
        "evidence_pass_hash": stable_hash(evidence_pass),
        "score_may_not_overwrite_evidence": True,
    }


class ECGEvidenceAgent(LLMAgentBase):
    """Agent1: ECG baseline audit agent."""

    agent_id = "agent1"
    agent_name = "Agent1_ECG_Baseline_Audit_Agent"
    prompt_key = "agent1"
    schema_key = "agent1"

    def build_payload(self, context: dict[str, Any]) -> dict[str, Any]:
        return {"ecg_evidence_packet": build_agent1_stage_a_blind_packet(context)}

    def run_with_trace(self, context: dict[str, Any]) -> AgentRunResult:
        if self.client is None:
            raise RuntimeError(f"{self.__class__.__name__} requires an injected LLM client")
        packet = build_safe_ecg_packet(context)
        stage_a_packet = build_agent1_stage_a_blind_packet(context)
        self.validate_payload(stage_a_packet)
        stage_schema_key = "agent1.evidence_pass"
        stage_schema = V2_6_AGENT_STAGE_SCHEMAS[stage_schema_key]
        evidence_fields = set(stage_schema["required"])
        user_payload = {
            "agent_id": self.agent_id,
            "case_id": str(context.get("case_id", "")),
            "runtime_version": "v2.8",
            "schema_version": V2_6_SCHEMA_VERSION,
            "schema_hash": stable_hash(stage_schema),
            "input": {"ecg_evidence_packet": stage_a_packet},
        }
        started = time.perf_counter()
        raw, attempt_count, network_retry_errors = self._complete_with_network_retry(
            V2_6_AGENT_EVIDENCE_PROMPTS["agent1"], user_payload
        )
        evidence, parse_error = parse_json_object(raw)
        parsed_before_normalization = dict(evidence) if isinstance(evidence, dict) else {}
        normalization_actions: list[dict[str, Any]] = []
        schema_errors_before_repair: list[str] = []
        schema_repair_attempted = False
        schema_repair_count = 0
        repaired_json: dict[str, Any] | None = None
        if not parse_error:
            evidence, runtime_metadata_actions = remove_matching_runtime_stage_metadata(
                evidence,
                expected_stage_schema_key=stage_schema_key,
            )
            normalization_actions.extend(runtime_metadata_actions)
            evidence, agent_identity_actions = normalize_runtime_owned_agent_identity(
                self.agent_id, evidence
            )
            normalization_actions.extend(agent_identity_actions)
            evidence, type_actions = normalize_llm_structured_output_types(
                evidence,
                ("hidden_ischemia_warning", "false_positive_mimic_warning"),
            )
            normalization_actions.extend(type_actions)
            evidence, enum_actions = normalize_schema_enum_values("agent1", evidence)
            normalization_actions.extend(enum_actions)
            evidence, pattern_actions = normalize_agent1_ecg_evidence_pattern(evidence)
            normalization_actions.extend(pattern_actions)
            evidence, boundary_actions = normalize_v2_6_safety_boundary_check(evidence)
            if boundary_actions.get("safety_boundary_normalized"):
                normalization_actions.append({
                    "field_path": "safety_boundary_check",
                    "normalization_rule": "SAFETY_BOUNDARY_NORMALIZATION",
                })

        def validate_evidence_pass(candidate: dict[str, Any]) -> None:
            validate_agent_stage_output(stage_schema_key, candidate)

        try:
            if parse_error:
                raise ValueError("Agent1 evidence pass schema mismatch")
            validate_evidence_pass(evidence)
        except Exception as exc:
            schema_errors_before_repair.append(str(exc))
            schema_repair_attempted = True
            schema_repair_count = 1
            if "ecg_evidence_pattern" in str(exc):
                allowed = list(self.schema.get("enums", {}).get("ecg_evidence_pattern", []))
                field_repair_raw = self.client.complete(
                    FIELD_LEVEL_SCHEMA_REPAIR_PROMPT,
                    build_field_level_repair_payload(
                        field_path="ecg_evidence_pattern",
                        invalid_value=evidence.get("ecg_evidence_pattern"),
                        allowed_values=allowed,
                        field_definition=(
                            "Coarse case-level ECG evidence category. Specific ECG findings such as "
                            "nonspecific_st_t_change, st_depression, t_wave_inversion, q_wave, or normal "
                            "must not be returned in this field."
                        ),
                    ),
                )
                field_repair_json, field_repair_parse_error = parse_json_object(field_repair_raw)
                if field_repair_parse_error:
                    raise ValueError("Agent1 evidence pass field-level repair returned invalid JSON") from exc
                corrected = field_repair_json.get("corrected_value")
                repaired_json = dict(evidence)
                repaired_json["ecg_evidence_pattern"] = corrected
                repaired_json, repair_actions = normalize_agent1_ecg_evidence_pattern(repaired_json)
                normalization_actions.extend(repair_actions)
            else:
                repaired_raw = self.client.complete(
                    TECHNICAL_SCHEMA_REPAIR_PROMPT,
                    build_schema_repair_payload(
                        parsed_json=parsed_before_normalization,
                        schema_error=str(exc),
                        target_schema={**stage_schema, "required": sorted(evidence_fields)},
                        required_output_fields=sorted(evidence_fields),
                        boolean_fields=["hidden_ischemia_warning", "false_positive_mimic_warning"],
                        stage_schema_key=stage_schema_key,
                    ),
                )
                repaired_json, repaired_parse_error = parse_json_object(repaired_raw)
                if repaired_parse_error:
                    raise ValueError("Agent1 evidence pass schema repair returned invalid JSON") from exc
                repaired_json, runtime_metadata_actions = remove_matching_runtime_stage_metadata(
                    repaired_json,
                    expected_stage_schema_key=stage_schema_key,
                )
                repaired_json, repair_identity_actions = normalize_runtime_owned_agent_identity(
                    self.agent_id, repaired_json
                )
                repaired_json, repair_actions = normalize_llm_structured_output_types(
                    repaired_json,
                    ("hidden_ischemia_warning", "false_positive_mimic_warning"),
                )
                repaired_json, repair_enum_actions = normalize_schema_enum_values("agent1", repaired_json)
                repaired_json, pattern_actions = normalize_agent1_ecg_evidence_pattern(repaired_json)
                repaired_json, boundary_actions = normalize_v2_6_safety_boundary_check(repaired_json)
                normalization_actions.extend(repair_actions)
                normalization_actions.extend(runtime_metadata_actions)
                normalization_actions.extend(repair_identity_actions)
                normalization_actions.extend(repair_enum_actions)
                normalization_actions.extend(pattern_actions)
                if boundary_actions.get("safety_boundary_normalized"):
                    normalization_actions.append({
                        "field_path": "safety_boundary_check",
                        "normalization_rule": "SAFETY_BOUNDARY_NORMALIZATION",
                    })
            validate_evidence_pass(repaired_json)
            evidence = repaired_json
        relation_repaired, relation_repair_actions = repair_agent1_pattern_strength_relation(evidence)
        if relation_repair_actions:
            schema_repair_attempted = True
            schema_repair_count = max(schema_repair_count, 1)
            schema_errors_before_repair.append(
                "cross_field_relation:ecg_evidence_pattern/ecg_directional_strength:"
                f"Agent1 incompatible pattern/strength: {evidence.get('ecg_evidence_pattern')}+{evidence.get('ecg_directional_strength')}"
            )
            normalization_actions.extend(relation_repair_actions)
            build_agent1_pattern_strength_relation_repair_payload(
                parsed_json=evidence,
                relation_error=schema_errors_before_repair[-1],
            )
            repaired_json = relation_repaired
            validate_evidence_pass(repaired_json)
            evidence = repaired_json
        raw_auxiliary_evidence_before_sanitization = deepcopy(evidence)
        evidence, auxiliary_evidence_sanitization_report = sanitize_agent1_auxiliary_evidence_fields(
            evidence,
            packet,
        )
        validate_evidence_pass(evidence)
        score_packet = build_ecg_score_consistency_packet(context, evidence)
        ecg_anchor = context.get("ecg_anchor_context", {})
        score = _required_locked_float(ecg_anchor, "locked_ecg_score", "missing_locked_ecg_score_fail_closed")
        threshold_value = _required_locked_float(ecg_anchor, "locked_ecg_threshold", "missing_locked_ecg_threshold_fail_closed")
        threshold_available = True
        predicted_positive = bool(score >= threshold_value)
        pattern = str(evidence["ecg_evidence_pattern"])
        if pattern == "ISCHEMIA_SUPPORTIVE":
            evidence_side = "up"
        elif pattern == "NO_CLEAR_ISCHEMIC_EVIDENCE":
            evidence_side = "neutral"
        elif pattern in {"NON_SPECIFIC_OR_CONFOUNDED", "MIXED"}:
            evidence_side = "down" if has_explicit_patient_level_ecg_mimic_support(evidence) else "mixed"
        else:
            evidence_side = "not_evaluable"
        if evidence["ecg_signal_quality"] == "LOW" or pattern == "NOT_EVALUABLE":
            consistency = "LOCAL_NOT_EVALUABLE"
            concern = "UNRESOLVED"
        elif not threshold_available or evidence_side in {"mixed", "neutral"}:
            consistency = "LOCAL_MIXED"
            concern = "UNRESOLVED"
        elif evidence_side == "up" and predicted_positive:
            consistency = "LOCAL_CONCORDANT"
            concern = "NONE"
        elif evidence_side == "down" and not predicted_positive:
            consistency = "LOCAL_CONCORDANT"
            concern = "NONE"
        elif evidence_side == "up" and not predicted_positive:
            consistency = "LOCAL_SCORE_POSSIBLY_UNDERSTATED"
            concern = "POSSIBLE_UNDERESTIMATION"
        elif evidence_side == "down" and predicted_positive:
            consistency = "LOCAL_SCORE_POSSIBLY_OVERSTATED"
            concern = "POSSIBLE_OVERESTIMATION"
        else:
            consistency = "LOCAL_MIXED"
            concern = "UNRESOLVED"
        explanation_available = not bool(packet.get("waveform_explanation_unavailable"))
        # Agent1-B is a local ECG baseline audit only. It must not compare
        # against any non-ECG model score or output an executable action.
        model_explanation_consistency = "UNAVAILABLE" if not explanation_available else (
            "CONSISTENT" if consistency == "LOCAL_CONCORDANT"
            else "PARTIALLY_CONSISTENT" if consistency == "LOCAL_MIXED"
            else "INCONSISTENT" if consistency in {"LOCAL_SCORE_POSSIBLY_OVERSTATED", "LOCAL_SCORE_POSSIBLY_UNDERSTATED"}
            else "UNAVAILABLE"
        )
        result = {
            "schema_version": V2_6_SCHEMA_VERSION,
            "agent_name": self.agent_name,
            "case_id": str(context.get("case_id", "")),
            "safety_boundary_check": V2_6_SAFE_BOUNDARY_CHECK,
            **evidence,
            "ecg_self_score_consistency": consistency,
            "ecg_local_concern": concern,
            "ecg_audit_confidence": "LOW" if consistency == "LOCAL_NOT_EVALUABLE" else "MEDIUM" if consistency == "LOCAL_MIXED" else "HIGH",
            "ecg_model_explanation_consistency": model_explanation_consistency,
            "ecg_audit_reason_code": "threshold_unavailable" if not threshold_available else consistency.lower(),
            "evidence_pass_hash": score_packet["evidence_pass_hash"],
            "score_consistency_pass_hash": stable_hash({
                "locked_ecg_score": score_packet["locked_ecg_score"],
                "locked_ecg_threshold": score_packet["locked_ecg_threshold"],
                "locked_ecg_risk_level": score_packet["locked_ecg_risk_level"],
                "ecg_self_score_consistency": consistency,
            }),
        }
        validate_v2_6_agent_output("agent1", result)
        trace = {
            "agent_id": "agent1", "agent_name": self.agent_name,
            "stage_schema_key": stage_schema_key,
            "case_id": result["case_id"], "schema_version": V2_6_SCHEMA_VERSION,
            "active_runtime_version": "v2.8",
            "active_contract_version": "v2_8_original_two_stage_semantic_boundary",
            "active_policy_version": "v2_8_action_only_diagnostic",
            "backward_compatibility_source_version": "v2.6_schema_contract",
            "prompt_hash": hashlib.sha256(V2_6_AGENT_EVIDENCE_PROMPTS["agent1"].encode("utf-8")).hexdigest(),
            "schema_hash": stable_hash(self.schema), "model_inference_config_hash": self.model_inference_config_hash(), "transport_execution_config_hash": self.transport_execution_config_hash(),
            "attempt_count": attempt_count, "retry_request_count": attempt_count - 1,
            "network_retry_errors": network_retry_errors, "raw_response": raw,
            "stage_a_user_payload": user_payload,
            "parsed_output": result, "evidence_pass": evidence,
            "parsed_json_before_normalization": parsed_before_normalization,
            "normalization_actions": normalization_actions,
            "normalized_json": evidence,
            "schema_errors_before_repair": schema_errors_before_repair,
            "schema_repair_attempted": schema_repair_attempted,
            "schema_repair_count": schema_repair_count,
            "repaired_json": repaired_json,
            "raw_auxiliary_evidence_before_sanitization": raw_auxiliary_evidence_before_sanitization,
            "auxiliary_evidence_sanitization_report": auxiliary_evidence_sanitization_report,
            "agent1_concept_projection_auxiliary_only": True,
            "auxiliary_id_error_aborts_case": False,
            "final_schema_status": "pass",
            "score_consistency_packet": score_packet,
            "latency_seconds": round(time.perf_counter() - started, 6),
            "fallback_used": False,
        }
        self.last_trace = trace
        if self.trace_sink:
            self.trace_sink(trace)
        return AgentRunResult(output=result, trace=trace)
