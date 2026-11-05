"""Method-only extraction; study-specific commands and fixtures removed."""
from __future__ import annotations

import argparse

import hashlib

import json

import os

import tempfile

import time

from collections import Counter, defaultdict

from copy import deepcopy

from dataclasses import dataclass

from pathlib import Path

from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]

from src.agents.ecg_audit_agent import (
    ALLOWED_ECG_KEYS,
    FORMAL_ECG_FEATURES,
    ECGEvidenceAgent,
    build_safe_ecg_packet,
    stable_hash,
)

from src.agents.clinical_assessment_agent import (
    CLINICAL_KEYS,
    FORMAL_CLINICAL_FEATURES,
    ClinicalLabEvidenceAgent,
    build_safe_clinical_packet,
)

import src.agents.evidence_synthesis_agent as agent3_rag_module

from src.agents.evidence_synthesis_agent import (
    RAGDiscordanceMechanismAgent,
    build_initial_rag_query_v2_6,
    build_multi_query_bundle_v2_6,
    build_retrieval_plan_v2_6,
    build_disagreement_triage_v2_6,
    evaluate_card_applicability_gate_v2_6,
    expand_query_with_cohort_priors_v2_6,
    patient_present_concepts_v2_6,
    rerank_cards_with_metadata_v2_6,
)

build_agent3_stage_a_only_output_v2_8 = getattr(
    agent3_rag_module,
    "build_agent3_stage_a_only_output_" + "v2" + "_7",
)

from src.agents.decision_recommendation_agent import (
    SafetyReflectionAgent,
)

from src.runtime.final_single_controller import FinalSingleController

from src.agents.schemas import (
    AGENT3_RETRIEVAL_TRACE_ONLY_FIELDS,
    V2_6_AGENT_SCHEMAS,
    V2_6_SAFE_BOUNDARY_CHECK,
    V2_6_SCHEMA_VERSION,
    find_forbidden_v2_6_fields,
    validate_agent3_committed_output,
    validate_agent3_raw_output,
    validate_v2_6_agent_output,
)

from src.runtime.ecg_anchor_contract import (
    assert_ecg_anchor_runtime_boundary,
    build_ecg_anchor_context,
    find_forbidden_runtime_fields,
)

from src.rag.knowledge_card import KnowledgeCard

from src.rag.retriever import KeywordRetriever

STAGES = [
    "PENDING", "AGENT1_COMMITTED", "AGENT2_COMMITTED",
    "RETRIEVAL_ROUND_0_COMMITTED", "AGENT3_ROUND_0_COMMITTED",
    "AGENT4_REFLECTION_ROUND_0_COMMITTED",
    "ELIGIBILITY_COMMITTED",
    "PUBLIC_ARTIFACT_COMMITTED", "CASE_COMPLETE", "QUARANTINED_PARTIAL",
]

def runtime_version_label(policy: dict[str, Any] | None) -> str:
    return str((policy or {}).get("dataset_version", "v2_8")).replace("_", ".")

def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(name, path)
        except PermissionError:
            try:
                Path(name).replace(path)
            except PermissionError:
                with path.open("w", encoding="utf-8", newline="\n") as fallback:
                    json.dump(value, fallback, ensure_ascii=False, indent=2, default=str)
                    fallback.flush()
                    os.fsync(fallback.fileno())
    finally:
        if os.path.exists(name):
            try:
                os.unlink(name)
            except PermissionError:
                pass

AGENT_NAME_TO_ID = {
    schema["agent_name"]: agent_id
    for agent_id, schema in V2_6_AGENT_SCHEMAS.items()
}

def _safe_filename_token(value: Any) -> str:
    text = str(value or "unknown")
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)[:120] or "unknown"

class Agent4EmptyFinalContentError(RuntimeError):
    """HTTP 200 streaming response contained reasoning but no final JSON content."""

    error_code = "AGENT4_EMPTY_FINAL_CONTENT"

class IncompleteStreamingResponseError(RuntimeError):
    """SSE response ended without the provider completion markers."""

class Agent4IncompleteStreamingResponseError(IncompleteStreamingResponseError):
    """HTTP 200 SSE response ended without the provider's completion markers."""

    error_code = "AGENT4_INCOMPLETE_STREAMING_RESPONSE"

class Agent13IncompleteStreamingResponseError(IncompleteStreamingResponseError):
    """Agent1-3 SSE response ended without a complete final JSON body."""

    error_code = "AGENT1_3_INCOMPLETE_STREAMING_RESPONSE"

class Agent13EmptyFinalContentError(RuntimeError):
    """Agent1-3 SSE response completed without final JSON content."""

    error_code = "AGENT1_3_EMPTY_FINAL_CONTENT"

class RawResponsePersistingClient:
    """Persist raw LLM responses before callers can parse or validate them."""

    # Agent implementations must not wrap this client in another full network
    # retry loop.  This wrapper owns technical retry persistence and limits.
    owns_transport_retries = True

    def __init__(
        self,
        client: Any,
        output_dir: Path,
        *,
        runtime_version: str = "v2.8",
        max_technical_retries_per_call: int = 0,
        retry_backoff_seconds: list[float] | None = None,
        agent1_3_streaming_enabled: bool = False,
        agent1_3_max_stream_duration_seconds: float | None = None,
        agent1_3_thinking_mode: str | None = None,
        agent1_3_reasoning_effort: str | None = None,
        agent4_streaming_enabled: bool = False,
        agent4_max_output_tokens: int | None = None,
        agent4_max_total_attempts: int | None = None,
        agent4_thinking_mode: str | None = None,
    ):
        self.client = client
        self.output_dir = output_dir
        self.runtime_version = runtime_version
        self.max_technical_retries_per_call = max(0, int(max_technical_retries_per_call))
        self.retry_backoff_seconds = retry_backoff_seconds or []
        self.agent1_3_streaming_enabled = bool(agent1_3_streaming_enabled)
        self.agent1_3_max_stream_duration_seconds = (
            float(agent1_3_max_stream_duration_seconds)
            if agent1_3_max_stream_duration_seconds is not None
            else None
        )
        self.agent1_3_thinking_mode = str(agent1_3_thinking_mode or "").strip().lower()
        if self.agent1_3_thinking_mode not in {"", "enabled", "disabled"}:
            raise ValueError("agent1_3_thinking_mode must be enabled or disabled")
        self.agent1_3_reasoning_effort = str(agent1_3_reasoning_effort or "").strip().lower()
        if self.agent1_3_reasoning_effort == "none":
            self.agent1_3_reasoning_effort = ""
        if self.agent1_3_reasoning_effort not in {"", "high", "max"}:
            raise ValueError("agent1_3_reasoning_effort must be none, high, or max")
        self.agent4_streaming_enabled = bool(agent4_streaming_enabled)
        self.agent4_max_output_tokens = int(agent4_max_output_tokens) if agent4_max_output_tokens else None
        self.agent4_max_total_attempts = int(agent4_max_total_attempts) if agent4_max_total_attempts else None
        self.agent4_thinking_mode = str(agent4_thinking_mode or "").strip().lower()
        self.call_index = 0
        self.records: list[dict[str, Any]] = []

    @property
    def config(self) -> Any:
        return getattr(self.client, "config", None)

    def _infer_case_id(self, payload: dict[str, Any] | None) -> str:
        if not isinstance(payload, dict):
            return "unknown_case"
        for key in ("case_id", "case_runtime_id"):
            if payload.get(key):
                return str(payload[key])
        parsed = payload.get("parsed_json_before_repair")
        if isinstance(parsed, dict) and parsed.get("case_id"):
            return str(parsed["case_id"])
        return "unknown_case"

    def _infer_agent_id(self, prompt: str, payload: dict[str, Any] | None) -> str:
        if isinstance(payload, dict):
            explicit_agent_id = str(payload.get("agent_id", ""))
            if explicit_agent_id in V2_6_AGENT_SCHEMAS:
                return explicit_agent_id
            schema = payload.get("required_output_schema")
            if isinstance(schema, dict) and schema.get("agent_name") in AGENT_NAME_TO_ID:
                return AGENT_NAME_TO_ID[str(schema["agent_name"])]
            target_schema = payload.get("target_schema")
            if isinstance(target_schema, dict) and target_schema.get("agent_name") in AGENT_NAME_TO_ID:
                return AGENT_NAME_TO_ID[str(target_schema["agent_name"])]
        for agent_id, schema in V2_6_AGENT_SCHEMAS.items():
            if schema["agent_name"] in prompt:
                return agent_id
        if isinstance(payload, dict) and payload.get("field_path") == "ecg_evidence_pattern":
            return "agent1"
        return "unknown_agent"

    def _infer_stage_name(self, payload: dict[str, Any] | None) -> str:
        if isinstance(payload, dict) and payload.get("repair_scope"):
            return str(payload["repair_scope"])
        return "primary_structured_output"

    def _write_record(
        self,
        *,
        prompt: str,
        payload: dict[str, Any] | None,
        raw_response_text: str | None,
        api_status: str,
        latency_seconds: float,
        attempt_index: int,
        retry_planned: bool,
        error: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.call_index += 1
        config = getattr(self.client, "config", None)
        case_runtime_id = self._infer_case_id(payload)
        agent_id = self._infer_agent_id(prompt, payload)
        stage_name = self._infer_stage_name(payload)
        raw_dir = self.output_dir / "restricted_internal" / "raw_llm_responses"
        path = raw_dir / (
            f"{_safe_filename_token(case_runtime_id)}_"
            f"{_safe_filename_token(agent_id)}_"
            f"{_safe_filename_token(stage_name)}_"
            f"{self.call_index:04d}_raw_response.json"
        )
        response_metadata = {
            "provider": getattr(config, "provider", ""),
            "base_url": getattr(config, "api_base", ""),
            "temperature": getattr(config, "temperature", None),
            "timeout_seconds": getattr(config, "timeout_seconds", None),
        }
        if extra_metadata:
            response_metadata.update(extra_metadata)
        if isinstance(payload, dict):
            response_metadata["repair_scope"] = payload.get("repair_scope")
            response_metadata["field_path"] = payload.get("field_path")
        request_record = {
            "prompt_text": prompt,
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "user_payload": payload,
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
                },
            ],
        }
        effective_thinking_mode = (
            self.agent1_3_thinking_mode
            if agent_id in {"agent1", "agent2", "agent3"}
            else self.agent4_thinking_mode if agent_id == "agent4" else ""
        )
        request_record["inference_overrides"] = {
            "thinking_mode": effective_thinking_mode or None,
            "reasoning_effort": (
                self.agent1_3_reasoning_effort
                if agent_id in {"agent1", "agent2", "agent3"}
                and self.agent1_3_thinking_mode == "enabled"
                else None
            ),
        }
        request_record["request_hash"] = stable_hash({
            "prompt_hash": request_record["prompt_hash"],
            "user_payload": payload,
            "model": getattr(config, "model", ""),
            "temperature": getattr(config, "temperature", None),
            "thinking_mode": effective_thinking_mode or None,
            "reasoning_effort": request_record["inference_overrides"]["reasoning_effort"],
        })
        record = {
            "case_runtime_id": case_runtime_id,
            "agent_id": agent_id,
            "stage_name": stage_name,
            "call_index": self.call_index,
            "attempt_index": attempt_index,
            "prompt_hash": request_record["prompt_hash"],
            "request_hash": request_record["request_hash"],
            "raw_request": request_record,
            "schema_version": V2_6_AGENT_SCHEMAS.get(agent_id, {}).get("enums", {}).get("schema_version", [""])[0],
            "runtime_version": self.runtime_version,
            "llm_model": getattr(config, "model", ""),
            "raw_response_text": raw_response_text,
            "raw_response_metadata": response_metadata,
            "api_status": api_status,
            "latency_seconds": round(latency_seconds, 6),
            "timeout_seconds": getattr(config, "timeout_seconds", None),
            "input_token_count": None,
            "output_token_count": None,
            "streaming": bool((extra_metadata or {}).get("streaming", False)),
            "stream_telemetry": extra_metadata or {},
            "created_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            "error": error,
            "error_type": None if error is None else error.split("(", 1)[0],
            "error_message": error,
            "retry_planned": retry_planned,
            "raw_response_saved_before_parse": True,
            "raw_request_saved_before_api": True,
            "raw_request_trace_persisted": True,
            "raw_response_saved_before_normalization": True,
            "raw_response_saved_before_repair": True,
            "raw_response_saved_before_schema_validation": True,
        }
        atomic_write_json(path, record)
        record["raw_response_artifact_path"] = str(path)
        self.records.append(record)
        return record

    @staticmethod
    def _is_retryable_api_error(exc: Exception) -> bool:
        if bool(getattr(exc, "transport_protocol_failure", False)):
            return True
        name = type(exc).__name__.lower()
        text = repr(exc).lower()
        if isinstance(exc, Agent4EmptyFinalContentError):
            return False
        if "readtimeout" in name or "timeout" in name or "timeout" in text:
            return True
        if "connectionerror" in name or "connection" in text:
            return True
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code in {429, 500, 502, 503, 504}:
            return True
        return any(token in text for token in ("temporary network failure", "http 429", "http 500", "http 502", "http 503", "http 504"))

    @staticmethod
    def _is_deterministic_non_transport_error(exc: Exception) -> bool:
        if bool(getattr(exc, "transport_protocol_failure", False)):
            return False
        text = f"{type(exc).__name__}:{exc}".lower()
        return any(token in text for token in (
            "json", "schema", "enum", "forbidden", "safe-view", "safe view",
            "precommit", "controller", "persistence", "empty_final_content",
        ))

    def complete(self, prompt: str, payload: dict[str, Any] | None = None) -> str:
        time_module = __import__("time")
        inferred_agent_id = self._infer_agent_id(prompt, payload)
        agent1_3_streaming = (
            self.agent1_3_streaming_enabled
            and inferred_agent_id in {"agent1", "agent2", "agent3"}
            and hasattr(self.client, "complete_streaming_json")
        )
        agent4_streaming = (
            self.agent4_streaming_enabled
            and inferred_agent_id == "agent4"
            and hasattr(self.client, "complete_streaming_json")
        )
        streaming_enabled = agent1_3_streaming or agent4_streaming
        configured_max_output_tokens = self.agent4_max_output_tokens if agent4_streaming else None
        configured_max_stream_duration_seconds = (
            self.agent1_3_max_stream_duration_seconds if agent1_3_streaming else None
        )
        if agent1_3_streaming and self.agent1_3_thinking_mode not in {"enabled", "disabled"}:
            raise RuntimeError("AGENT1_3_THINKING_MODE_NOT_EXPLICIT")
        if agent1_3_streaming and self.agent1_3_thinking_mode == "enabled":
            if self.agent1_3_reasoning_effort not in {"high", "max"}:
                raise RuntimeError("AGENT1_3_REASONING_EFFORT_NOT_EXPLICIT")
            active_model = str(getattr(self.config, "model", "") or "").strip().lower()
            if "deepseek-v4-flash" not in active_model:
                raise RuntimeError("AGENT1_3_REASONING_EFFORT_MODEL_UNSUPPORTED")
        max_attempts = (
            self.agent4_max_total_attempts
            if agent4_streaming and self.agent4_max_total_attempts
            else self.max_technical_retries_per_call + 1
        )
        for attempt_index in range(1, max_attempts + 1):
            started = time_module.perf_counter()
            try:
                if streaming_enabled:
                    if agent1_3_streaming:
                        extra_body = {"thinking": {"type": self.agent1_3_thinking_mode}}
                        if self.agent1_3_thinking_mode == "enabled":
                            extra_body["reasoning_effort"] = self.agent1_3_reasoning_effort
                    elif (
                        agent4_streaming
                        and self.agent4_thinking_mode in {"disabled", "disable", "off", "false", "0"}
                    ):
                        extra_body = {"thinking": {"type": "disabled"}}
                    else:
                        extra_body = None
                    raw, stream_telemetry = self.client.complete_streaming_json(
                        prompt,
                        payload,
                        max_tokens=configured_max_output_tokens,
                        max_stream_duration_seconds=configured_max_stream_duration_seconds,
                        extra_body=extra_body,
                    )
                    stream_done_seen = bool((stream_telemetry or {}).get("stream_done_seen"))
                    finish_reason = str((stream_telemetry or {}).get("finish_reason") or "").strip().lower()
                    deadline_complete_json_accepted = bool(
                        (stream_telemetry or {}).get("deadline_complete_json_accepted")
                    )
                    if deadline_complete_json_accepted:
                        try:
                            deadline_raw = json.loads(str(raw or ""))
                        except (json.JSONDecodeError, TypeError, ValueError):
                            deadline_raw = None
                        deadline_complete_json_accepted = isinstance(deadline_raw, dict)
                    stream_completion_accepted = (
                        (stream_done_seen and finish_reason == "stop")
                        or (deadline_complete_json_accepted and finish_reason == "stop")
                    )
                    if not stream_completion_accepted:
                        latency = time_module.perf_counter() - started
                        retry_planned = attempt_index < max_attempts
                        error_class = (
                            Agent4IncompleteStreamingResponseError
                            if agent4_streaming
                            else Agent13IncompleteStreamingResponseError
                        )
                        error_code = error_class.error_code
                        self._write_record(
                            prompt=prompt,
                            payload=payload,
                            raw_response_text=raw,
                            api_status="incomplete_stream",
                            latency_seconds=latency,
                            attempt_index=attempt_index,
                            retry_planned=retry_planned,
                            error=error_code,
                            extra_metadata={
                                **stream_telemetry,
                                "streaming": streaming_enabled,
                                "configured_max_output_tokens": configured_max_output_tokens,
                                "max_total_attempts": max_attempts,
                                "incomplete_stream_error_code": error_code,
                                "incomplete_stream_is_transport_protocol_failure": True,
                                "incomplete_stream_not_schema_repair": True,
                            },
                        )
                        if retry_planned:
                            delay = self.retry_backoff_seconds[min(
                                attempt_index - 1,
                                max(0, len(self.retry_backoff_seconds) - 1),
                            )] if self.retry_backoff_seconds else 0
                            if delay:
                                time_module.sleep(delay)
                            continue
                        raise error_class(error_code)
                    content_empty = not str(raw or "").strip()
                    reasoning_bytes = int((stream_telemetry or {}).get("reasoning_content_bytes") or 0)
                    if content_empty:
                        latency = time_module.perf_counter() - started
                        error_code = (
                            "HTTP_SUCCESS_REASONING_ONLY_NO_FINAL_CONTENT"
                            if reasoning_bytes > 0 else
                            "AGENT4_EMPTY_FINAL_CONTENT"
                        )
                        retry_planned = (not agent4_streaming) and attempt_index < max_attempts
                        error_class = Agent4EmptyFinalContentError if agent4_streaming else Agent13EmptyFinalContentError
                        if not agent4_streaming:
                            error_code = Agent13EmptyFinalContentError.error_code
                        self._write_record(
                            prompt=prompt,
                            payload=payload,
                            raw_response_text=raw,
                            api_status="empty_content",
                            latency_seconds=latency,
                            attempt_index=attempt_index,
                            retry_planned=retry_planned,
                            error=error_code,
                            extra_metadata={
                                **stream_telemetry,
                                "streaming": streaming_enabled,
                                "configured_max_output_tokens": configured_max_output_tokens,
                                "max_total_attempts": max_attempts,
                                "empty_content_error_code": error_code,
                                "empty_content_not_transport_retry": bool(agent4_streaming),
                                "empty_content_protocol_retry": bool(not agent4_streaming),
                                "empty_content_not_schema_repair": True,
                            },
                        )
                        if retry_planned:
                            delay = self.retry_backoff_seconds[min(
                                attempt_index - 1,
                                max(0, len(self.retry_backoff_seconds) - 1),
                            )] if self.retry_backoff_seconds else 0
                            if delay:
                                time_module.sleep(delay)
                            continue
                        raise error_class(error_code)
                else:
                    raw = self.client.complete(prompt, payload)
                    stream_telemetry = {}
            except Exception as exc:
                # The incomplete-stream branch above already persisted the
                # partial body and its completion telemetry.  Do not create a
                # second, body-less error record for the same HTTP attempt.
                if isinstance(exc, (IncompleteStreamingResponseError, Agent4EmptyFinalContentError, Agent13EmptyFinalContentError)):
                    raise
                retry_planned = (
                    attempt_index < max_attempts
                    and self._is_retryable_api_error(exc)
                    and not self._is_deterministic_non_transport_error(exc)
                )
                partial_content = getattr(exc, "partial_content", None)
                partial_telemetry = getattr(exc, "stream_telemetry", {})
                self._write_record(
                    prompt=prompt,
                    payload=payload,
                    raw_response_text=partial_content,
                    api_status="error",
                    latency_seconds=time_module.perf_counter() - started,
                    attempt_index=attempt_index,
                    retry_planned=retry_planned,
                    error=repr(exc),
                    extra_metadata={
                        **(partial_telemetry if isinstance(partial_telemetry, dict) else {}),
                        "streaming": streaming_enabled,
                        "configured_max_output_tokens": configured_max_output_tokens,
                        "max_total_attempts": max_attempts,
                    },
                )
                if retry_planned:
                    delay = self.retry_backoff_seconds[min(
                        attempt_index - 1,
                        max(0, len(self.retry_backoff_seconds) - 1),
                    )] if self.retry_backoff_seconds else 0
                    if delay:
                        time_module.sleep(delay)
                    continue
                raise
            self._write_record(
                prompt=prompt,
                payload=payload,
                raw_response_text=raw,
                api_status="success",
                latency_seconds=time_module.perf_counter() - started,
                attempt_index=attempt_index,
                retry_planned=False,
                extra_metadata={
                    **stream_telemetry,
                    "streaming": streaming_enabled,
                    "configured_max_output_tokens": configured_max_output_tokens,
                    "max_total_attempts": max_attempts,
                },
            )
            return raw
        raise RuntimeError("unreachable technical retry state")

def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))

def attach_runtime_policy_metadata(payload: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    """Attach locked runtime policy fragments without selecting artifacts dynamically."""

    enriched = dict(payload)
    agent2_config = policy.get("agent2")
    if isinstance(agent2_config, dict):
        enriched["agent2_discrepancy_model_config"] = dict(agent2_config)
    enriched["active_runtime_policy_path"] = policy.get("_policy_path", "")
    enriched["active_runtime_policy_hash"] = policy.get("_policy_hash", "")
    enriched["runtime_policy_variant"] = policy.get("policy_variant", "")
    enriched["runtime_graded_directional_mode_enabled"] = bool(policy.get("graded_directional_mode_enabled", False))
    enriched["runtime_controller_direction_guard_enabled"] = bool(policy.get("controller_direction_guard_enabled", False))
    enriched["runtime_correction_policy_locked"] = bool(
        policy.get("projection_mode") == "validation_locked_logit_shift"
        and policy.get("numeric_correction_applied") is True
    )
    v31 = policy.get("core_v3_1_minimal_semantic_hardening", {})
    enriched["canonical_relation_v3_1_enabled"] = bool(isinstance(v31, dict) and v31.get("enabled", False))
    return enriched

AI_REVIEWED_CARD_IDS_V1 = {
    "DANGER_DIGOXIN_001", "DANGER_DIGOXIN_004", "DANGER_DIGOXIN_005",
    "DANGER_ECG_001", "DANGER_ECG_002", "DANGER_ECG_003",
    "DANGER_ECG_005", "DATA_001", "DATA_CLINICAL_001",
    "DISC_003", "ECG_FP_007", "fp-drug-001",
    "ECG_FP_001", "ECG_FP_003", "ECG_FN_003", "ECG_FN_004", "DANGER_ECG_ST_001",
    "DANGER_ECG_STE_001", "DANGER_ELECTROLYTE_001",
}

GOVERNANCE_CARD_GROUPS = {"E_data_quality", "E_safety", "D_discordance_rule"}

CARD_RUNTIME_LIST_METADATA_FIELDS = {
    "required_evidence_concepts",
    "required_any_present_concepts",
    "required_all_present_concepts",
    "forbidden_evidence_concepts",
    "forbidden_status",
    "applicable_branches",
    "allowed_query_purposes",
}

CARD_RUNTIME_TEXT_METADATA_FIELDS = {
    "required_status",
    "runtime_pool",
    "reviewer_status",
    "applicability_state",
    "runtime_disposition",
}

def _card_runtime_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    card_id = str(raw.get("card_id", ""))
    group = str(raw.get("card_group") or raw.get("category") or "")
    title = str(raw.get("card_title") or raw.get("title") or "").lower()
    runtime_pool = "governance" if group in GOVERNANCE_CARD_GROUPS or card_id == "DANGER_ECG_SAFETY" else "patient_medical"
    reviewer_status = (
        "ai_reviewed_pending_clinician"
        if card_id in AI_REVIEWED_CARD_IDS_V1
        else "pending_review"
    )
    required_any: list[str] = []
    applicable_branches: list[str] = []
    allowed_purposes = ["core_mechanism", "branch_or_confounder", "counterevidence"]
    applicability_state = "standard"
    runtime_disposition = "KEEP_IN_RUNTIME"

    if card_id == "ECG_FP_001":
        required_any = ["nonspecific_st_t_change"]
        applicable_branches = ["ecg_false_positive_like", "global_cag"]
        allowed_purposes = ["core_mechanism", "branch_or_confounder", "counterevidence"]
    elif card_id == "ECG_FP_003":
        required_any = [
            "left_bundle_branch_block", "right_bundle_branch_block",
            "paced_rhythm", "explicit_bundle_branch_block_mimic",
            "bundle_branch_block_mimic",
        ]
        applicable_branches = ["ecg_false_positive_like", "global_cag", "hidden_high_risk_like"]
        allowed_purposes = ["core_mechanism", "branch_or_confounder", "counterevidence"]
    elif card_id == "ECG_FP_006":
        required_any = ["early_repolarization", "j_point_elevation_pattern"]
        applicable_branches = ["ecg_false_positive_like", "global_cag"]
        runtime_disposition = "EXCLUDE_FROM_RUNTIME"
        reviewer_status = "excluded_from_runtime"
    elif card_id == "ECG_FP_008":
        required_any = [
            "prior_myocardial_infarction", "left_ventricular_aneurysm",
            "persistent_st_elevation", "pathologic_q_wave",
        ]
        applicable_branches = ["ecg_false_positive_like", "global_cag"]
        runtime_disposition = "EXCLUDE_FROM_RUNTIME"
        reviewer_status = "excluded_from_runtime"
    elif card_id == "DANGER_ECG_STE_001":
        required_any = ["pathologic_q_wave", "q_wave_abnormality"]
        applicable_branches = ["ecg_false_positive_like", "global_cag"]
        allowed_purposes = ["core_mechanism", "branch_or_confounder", "counterevidence"]
    elif card_id == "ECG_FN_005":
        required_any = ["ckd", "renal_dysfunction", "egfr_low", "creatinine_high"]
        applicable_branches = ["hidden_high_risk_like", "global_cag"]
        runtime_disposition = "EXCLUDE_FROM_RUNTIME"
        reviewer_status = "excluded_from_runtime"
    elif card_id == "ECG_FN_003":
        required_any = ["diabetes"]
        applicable_branches = ["hidden_high_risk_like", "global_cag"]
        allowed_purposes = ["core_mechanism", "branch_or_confounder", "cohort_context"]
    elif card_id == "ECG_FN_004":
        required_any = []
        applicable_branches = ["hidden_high_risk_like"]
        allowed_purposes = ["cohort_context"]
    elif card_id == "DANGER_ECG_ST_001":
        required_any = ["st_elevation", "st_depression", "st_segment_abnormality"]
        applicable_branches = ["ecg_false_positive_like", "global_cag"]
        allowed_purposes = ["core_mechanism", "branch_or_confounder", "counterevidence"]
    elif card_id == "DANGER_ECG_NSTE_001":
        required_any = ["st_depression", "st_segment_abnormality"]
        applicable_branches = ["ecg_false_positive_like", "global_cag"]
        runtime_disposition = "EXCLUDE_AS_STANDALONE_CARD"
        reviewer_status = "pending_clinician_review"
    elif card_id == "DANGER_ECG_SAFETY":
        runtime_pool = "governance"
        runtime_disposition = "MOVE_TO_DETERMINISTIC_CONTROLLER"
        reviewer_status = "excluded_from_runtime"
    elif card_id.startswith("DANGER_DIGOXIN") or card_id in {"ECG_FP_005", "fp-drug-001"} or "digoxin" in title:
        required_any = ["digoxin_use", "digoxin_toxicity_signal"]
        applicable_branches = ["ecg_false_positive_like", "global_cag"]
    elif card_id in {"DANGER_ECG_001", "DANGER_ECG_WELLENS_001"} or "wellens" in title:
        required_any = ["wellens_specific_pattern"]
        applicability_state = "pending_specific_evidence"
    elif card_id == "DANGER_ECG_003" or "动态t波" in title or "dynamic t" in title:
        required_any = ["dynamic_t_wave_change", "serial_ecg_evidence"]
        applicability_state = "pending_specific_evidence"
    elif card_id == "DANGER_ECG_005" or "持续缺血症状" in title:
        required_any = ["symptom_attack_status"]
    elif card_id in {"DANGER_ECG_002", "DANGER_ECG_GLOBAL_001"}:
        required_any = ["avr_diffuse_st_pattern"]
        applicability_state = "pending_specific_evidence"
    elif card_id in {"DANGER_ECG_004", "DANGER_ECG_POSTERIOR_001"}:
        required_any = ["posterior_occlusion_pattern"]
        applicability_state = "pending_specific_evidence"
    elif card_id in {"DANGER_ECG_006", "DANGER_ECG_ST_002"}:
        required_any = ["reciprocal_st_depression"]
        applicability_state = "pending_specific_evidence"
    elif card_id == "DANGER_ECG_007":
        required_any = ["brugada_pattern"]
        applicability_state = "pending_specific_evidence"
    elif card_id == "DANGER_ECG_T_002":
        required_any = ["hyperacute_t_wave"]
        applicability_state = "pending_specific_evidence"
    elif card_id in {"DANGER_ELECTROLYTE_001", "ECG_FP_004", "DATA_CLINICAL_001"} or "低钾" in title:
        required_any = ["hypokalemia", "potassium_abnormality"]
        allowed_purposes = ["branch_or_confounder", "counterevidence"]
    elif card_id == "ECG_FP_007" or "心肌炎" in title or "心包炎" in title:
        required_any = ["myocarditis_context", "pericarditis_context"]
    elif card_id == "ECG_FP_002":
        required_any = ["left_ventricular_hypertrophy", "left_ventricular_hypertrophy_mimic"]
    elif card_id == "ECG_FP_003":
        required_any = ["left_bundle_branch_block", "right_bundle_branch_block", "bundle_branch_block_mimic"]

    metadata = {
        "required_evidence_concepts": required_any,
        "required_any_present_concepts": required_any,
        "required_all_present_concepts": [],
        "required_status": "present",
        "forbidden_evidence_concepts": required_any,
        "forbidden_status": ["absent", "missing", "unknown"],
        "applicable_branches": applicable_branches,
        "allowed_query_purposes": allowed_purposes,
        "runtime_pool": runtime_pool,
        "reviewer_status": reviewer_status,
        "applicability_state": applicability_state,
        "runtime_disposition": runtime_disposition,
    }
    for field in CARD_RUNTIME_LIST_METADATA_FIELDS:
        if field not in raw:
            continue
        if not isinstance(raw[field], list):
            raise TypeError(f"card runtime metadata must be a list: {card_id}:{field}")
        metadata[field] = [str(value) for value in raw[field]]
    for field in CARD_RUNTIME_TEXT_METADATA_FIELDS:
        if field in raw:
            metadata[field] = str(raw[field])
    return metadata

def _normalize_card(raw: dict[str, Any], source_file: str) -> dict[str, Any]:
    title = str(raw.get("title") or raw.get("card_title") or raw.get("card_id") or "")
    category = str(raw.get("category") or raw.get("card_group") or "")
    content = " ".join(str(raw.get(key, "")) for key in (
        "content", "clinical_problem", "applicable_when", "key_evidence_summary",
        "agent_action_hint", "blocked_action", "usage_boundary",
    ) if raw.get(key) is not None)
    keywords = raw.get("keywords")
    if not isinstance(keywords, list):
        keywords = []
    keywords = [str(item) for item in keywords]
    for text in (title, category, str(raw.get("agent_action_hint", ""))):
        keywords.extend(token for token in text.replace(";", " ").replace(",", " ").split() if token)
    return {
        "card_id": str(raw.get("card_id", "")),
        "title": title,
        "category": category,
        "keywords": sorted(set(keywords)),
        "content": content,
        "risk_direction": str(raw.get("risk_direction") or raw.get("agent_action_hint") or "preserve"),
        "source_id": str(raw.get("evidence_sources") or raw.get("source_excerpt_ids") or source_file),
        "source_type": str(raw.get("status") or raw.get("confidence_level") or "local_card"),
        "card_type": category,
        "target_card_id": str(raw.get("card_id", "")),
        "usage_boundary": str(raw.get("usage_boundary", "")),
        "source_excerpt_ids": raw.get("source_excerpt_ids", ""),
        "confidence_level": raw.get("confidence_level", ""),
        "safety_level": raw.get("safety_level", ""),
        **_card_runtime_metadata(raw),
    }

def _resolve_knowledge_cards_file(cards_dir: Path, cards_file: Path | None = None) -> Path:
    if cards_file is not None:
        selected = Path(cards_file)
        if not selected.is_absolute():
            selected = ROOT / selected
        if not selected.exists():
            raise FileNotFoundError(f"knowledge cards file not found: {selected}")
        return selected.resolve()
    directory = Path(cards_dir)
    if not directory.is_absolute():
        directory = ROOT / directory
    return (directory / "knowledge_cards_v1.jsonl").resolve()

def _load_local_knowledge_cards(cards_dir: Path, cards_file: Path | None = None) -> list[dict[str, Any]]:
    cards: dict[str, dict[str, Any]] = {}
    formal_path = _resolve_knowledge_cards_file(cards_dir, cards_file)
    paths = [formal_path]
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            card = _normalize_card(raw, path.name)
            if card["card_id"]:
                cards.setdefault(card["card_id"], card)
    return list(cards.values())

def _token_counts(text: str) -> Counter[str]:
    return Counter(t.strip(" ,.;:()[]{}").lower() for t in text.split() if t.strip())

def _retrieval_score(query: str, card: dict[str, Any]) -> int:
    q = _token_counts(query)
    text = " ".join([
        str(card.get("title", "")),
        str(card.get("content", "")),
        " ".join(map(str, card.get("keywords", []))),
    ]).lower()
    return sum(count for token, count in q.items() if token and token in text)

def _retrieve_cards(query: str, card_rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    retriever = KeywordRetriever([KnowledgeCard.from_dict(card) for card in card_rows])
    by_id = {card["card_id"]: card for card in card_rows}
    out: list[dict[str, Any]] = []
    for rank, card in enumerate(retriever.retrieve(query, top_k=top_k), start=1):
        row = dict(by_id.get(card.card_id, card.to_dict()))
        row["rank"] = rank
        row["retrieval_score"] = _retrieval_score(query, row)
        out.append(row)
    return out

def _tag_recall_cards(
    card_rows: list[dict[str, Any]],
    plan: dict[str, Any],
    query: dict[str, Any],
) -> list[dict[str, Any]]:
    present = patient_present_concepts_v2_6(plan)
    branch = str(plan.get("selected_branch", "global_cag"))
    purpose = str(query.get("query_purpose", "core_mechanism"))
    linked_cohort_cards = set(map(str, plan.get("linked_cohort_external_card_ids", [])))
    recalled: list[dict[str, Any]] = []
    for card in card_rows:
        if purpose == "cohort_context" and str(card.get("card_id", "")) in linked_cohort_cards:
            row = dict(card)
            row["rank"] = 999
            row["retrieval_score"] = 8
            row["tag_recall_match"] = "source_cohort_factor_card_link"
            recalled.append(row)
            continue
        required = set(map(str, card.get("required_any_present_concepts", card.get("required_evidence_concepts", []))))
        if not required or not (required & present):
            continue
        branches = set(map(str, card.get("applicable_branches", [])))
        if branches and branch not in branches:
            continue
        purposes = set(map(str, card.get("allowed_query_purposes", [])))
        if purposes and purpose not in purposes:
            continue
        row = dict(card)
        row["rank"] = 999
        row["retrieval_score"] = max(1, len(required & present)) + 3
        row["tag_recall_match"] = ";".join(sorted(required & present))
        recalled.append(row)
    return recalled

def _retrieve_query_bundle(
    query_bundle: list[dict[str, Any]],
    card_rows: list[dict[str, Any]],
    plan: dict[str, Any],
    top_k: int,
    rejected_cards: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for query in query_bundle:
        retrieved = _retrieve_cards(str(query.get("query_text", "")), card_rows, max(top_k * 3, top_k))
        by_id = {str(card["card_id"]): card for card in retrieved}
        for card in _tag_recall_cards(card_rows, plan, query):
            current = by_id.get(str(card["card_id"]))
            if current is None or int(card.get("retrieval_score", 0)) > int(current.get("retrieval_score", 0)):
                by_id[str(card["card_id"])] = card
        for card in by_id.values():
            gate = evaluate_card_applicability_gate_v2_6(card, plan, query)
            if not gate["applicability_gate_pass"]:
                if rejected_cards is not None:
                    rejected_cards.append({
                        **gate,
                        "query_id": query.get("query_id", ""),
                        "retrieval_score": card.get("retrieval_score", 0),
                    })
                continue
            current = candidates.get(str(card["card_id"]))
            if current is None or int(card.get("retrieval_score", 0)) > int(current.get("retrieval_score", 0)):
                candidates[str(card["card_id"])] = {
                    **card,
                    "recall_query_ids": [str(query.get("query_id", ""))],
                    "applicability_gate": gate,
                }
            elif str(query.get("query_id", "")) not in current["recall_query_ids"]:
                current["recall_query_ids"].append(str(query.get("query_id", "")))
    reranked = rerank_cards_with_metadata_v2_6(list(candidates.values()), plan)
    for rank, card in enumerate(reranked[:top_k], start=1):
        card["rank"] = rank
    return reranked[:top_k]

def _active_kb_file_hash(cards_dir: Path, cards_file: Path | None = None) -> str:
    formal_path = _resolve_knowledge_cards_file(cards_dir, cards_file)
    if not formal_path.exists():
        return ""
    h = hashlib.sha256()
    with formal_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

class RuntimeFingerprintMismatchError(RuntimeError):
    """Raised before any API call when a resume would mix runtime versions."""

@dataclass
class CaseTransaction:
    output_dir: Path
    case_runtime_id: str
    input_fingerprint: str
    prompt_bundle_hash: str
    schema_hash: str
    model_inference_config_hash: str
    transport_execution_config_hash: str
    fingerprint_schema_version: str
    fingerprint_hash_algorithm: str
    runtime_fingerprint: dict[str, Any]
    resume_from_last_successful_stage: bool = False

    @property
    def case_dir(self) -> Path:
        return self.output_dir / "restricted_internal" / "transactions" / self.case_runtime_id

    @property
    def manifest_path(self) -> Path:
        return self.case_dir / "transaction_manifest_v1.json"

    def initialize(self) -> dict[str, Any]:
        existing = read_json(self.manifest_path)
        identity = {
            "transaction_id": stable_hash({
                "case_runtime_id": self.case_runtime_id,
                "input_fingerprint": self.input_fingerprint,
                "prompt_bundle_hash": self.prompt_bundle_hash,
                "schema_hash": self.schema_hash,
                "model_inference_config_hash": self.model_inference_config_hash,
                "transport_execution_config_hash": self.transport_execution_config_hash,
            }),
            "case_runtime_id": self.case_runtime_id,
            "input_fingerprint": self.input_fingerprint,
            "prompt_bundle_hash": self.prompt_bundle_hash,
            "schema_hash": self.schema_hash,
            "model_inference_config_hash": self.model_inference_config_hash,
            "transport_execution_config_hash": self.transport_execution_config_hash,
            "fingerprint_schema_version": self.fingerprint_schema_version,
            "fingerprint_hash_algorithm": self.fingerprint_hash_algorithm,
            "runtime_fingerprint": self.runtime_fingerprint,
            "runtime_fingerprint_hash": stable_hash(self.runtime_fingerprint),
        }
        if existing:
            for key, value in identity.items():
                if existing.get(key) != value:
                    raise RuntimeFingerprintMismatchError(f"RUNTIME_FINGERPRINT_MISMATCH:{key}")
            return existing
        manifest = {
            **identity,
            "current_stage": "PENDING",
            "last_committed_stage": "PENDING",
            "artifact_sha256": {},
            "status": "active",
        }
        atomic_write_json(self.manifest_path, manifest)
        return manifest

    def stage_path(self, stage: str) -> Path:
        return self.case_dir / f"{stage.lower()}_v1.json"

    def load_committed(self, stage: str) -> Any | None:
        manifest = self.initialize()
        if STAGES.index(manifest["last_committed_stage"]) < STAGES.index(stage):
            return None
        path = self.stage_path(stage)
        if not path.exists():
            raise RuntimeError(f"committed stage artifact missing: {stage}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if manifest["artifact_sha256"].get(stage) != digest:
            raise RuntimeError(f"committed stage hash mismatch: {stage}")
        return read_json(path)

    def commit(self, stage: str, value: Any) -> None:
        manifest = self.initialize()
        if stage not in STAGES:
            raise ValueError(f"unknown transaction stage: {stage}")
        if STAGES.index(stage) <= STAGES.index(manifest["last_committed_stage"]):
            existing = self.load_committed(stage)
            if stable_hash(existing) != stable_hash(value):
                raise RuntimeError(f"idempotency conflict at {stage}")
            return
        path = self.stage_path(stage)
        atomic_write_json(path, value)
        manifest["current_stage"] = stage
        manifest["last_committed_stage"] = stage
        manifest["artifact_sha256"][stage] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["status"] = "complete" if stage == "CASE_COMPLETE" else "active"
        atomic_write_json(self.manifest_path, manifest)

    def quarantine_partial(self, reason: str) -> None:
        manifest = self.initialize()
        manifest["current_stage"] = "QUARANTINED_PARTIAL"
        manifest["status"] = "quarantined"
        manifest["quarantine_reason"] = reason
        atomic_write_json(self.manifest_path, manifest)

def _agent_kwargs(args: argparse.Namespace, sink: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    return {
        "max_retries": args.llm_max_retries,
        "retry_backoff_seconds": args.llm_retry_backoff_seconds,
        "retry_backoff_multiplier": args.llm_retry_backoff_multiplier,
        "trace_sink": sink,
    }

def build_agents(client: Any, args: argparse.Namespace, sink: Callable[[dict[str, Any]], None]) -> tuple[Any, ...]:
    kwargs = _agent_kwargs(args, sink)
    return (
        ECGEvidenceAgent(client, **kwargs),
        ClinicalLabEvidenceAgent(client, **kwargs),
        RAGDiscordanceMechanismAgent(client, **kwargs),
        SafetyReflectionAgent(
            client,
            maximum_agent4_generation_attempts=getattr(args, "agent4_max_generation_attempts", 0) or None,
            maximum_empty_content_regeneration_attempts=getattr(args, "agent4_max_empty_content_regeneration_attempts", None),
            **kwargs,
        ),
        FinalSingleController(),
    )

def build_retrieval_round_v2_6(
    payload: dict[str, Any],
    *,
    agent1_output: dict[str, Any],
    agent2_output: dict[str, Any],
    triage: dict[str, Any],
    knowledge_cards_file: Path | None = None,
) -> dict[str, Any]:
    round_index = 0
    base_cards = payload.get("rag_context", {}).get("retrieved_cards", payload.get("retrieved_cards", []))
    retrieval_plan = build_retrieval_plan_v2_6(payload, agent1_output, agent2_output, triage)
    query = expand_query_with_cohort_priors_v2_6(
        build_initial_rag_query_v2_6(
            {**payload, "retrieval_round_index": round_index},
            agent1_output,
            agent2_output,
            triage,
        ),
        list(agent2_output.get("patient_factor_matches", [])),
        branch_scope=retrieval_plan["selected_branch"],
    )
    query["retrieval_plan"] = retrieval_plan
    query["multi_query_bundle"] = build_multi_query_bundle_v2_6(retrieval_plan, round_index)
    query["query_hash"] = stable_hash(query["multi_query_bundle"])
    normalized_base_cards = [
        _normalize_card(card, "existing_payload")
        for card in base_cards if isinstance(card, dict)
    ]
    gate_rejections: list[dict[str, Any]] = []
    applicable_cards: list[dict[str, Any]] = []
    bundle = list(query.get("multi_query_bundle", []))
    runtime_active_kb_retrieval_used = False
    active_kb_cards: list[dict[str, Any]] = []
    active_kb_cards_dir = ROOT / "knowledge_base" / "cards"
    selected_kb_file = _resolve_knowledge_cards_file(active_kb_cards_dir, knowledge_cards_file)
    active_kb_hash = ""
    if normalized_base_cards:
        for card in normalized_base_cards:
            passes = []
            for bundle_query in bundle:
                gate = evaluate_card_applicability_gate_v2_6(card, query.get("retrieval_plan", {}), bundle_query)
                if gate["applicability_gate_pass"]:
                    passes.append(gate)
                else:
                    gate_rejections.append({**gate, "query_id": bundle_query.get("query_id", "")})
            if passes:
                applicable_cards.append({**card, "applicability_gate": passes[0]})
        reranked_cards = rerank_cards_with_metadata_v2_6(applicable_cards, query.get("retrieval_plan", {}))
        candidate_card_count = len(base_cards)
        retrieval_source = "existing_payload_or_offline_retriever"
    else:
        active_kb_cards = _load_local_knowledge_cards(active_kb_cards_dir, selected_kb_file)
        active_kb_hash = _active_kb_file_hash(active_kb_cards_dir, selected_kb_file)
        if active_kb_cards and bundle:
            runtime_active_kb_retrieval_used = True
            reranked_cards = _retrieve_query_bundle(
                bundle,
                active_kb_cards,
                query.get("retrieval_plan", {}),
                top_k=5,
                rejected_cards=gate_rejections,
            )
        else:
            reranked_cards = []
        candidate_card_count = len(active_kb_cards)
        retrieval_source = "runtime_active_kb_fallback" if runtime_active_kb_retrieval_used else "empty_payload_no_active_kb_candidates"
    return {
        "retrieval_round_index": round_index,
        "technical_retry_count": 0,
        "retrieved_cards": reranked_cards[:5],
        "candidate_card_count": candidate_card_count,
        "applicability_gate_rejection_count": len(gate_rejections),
        "applicability_gate_rejections": gate_rejections,
        "metadata_rerank_consumed": True,
        "multi_query_bundle": query.get("multi_query_bundle", []),
        "query_hash": query.get("query_hash"),
        "retrieval_query_contract": query,
        "retrieval_query": query.get("retrieval_query", ""),
        "retrieval_source": retrieval_source,
        "runtime_active_kb_retrieval_used": runtime_active_kb_retrieval_used,
        "knowledge_base_path": str(selected_kb_file) if runtime_active_kb_retrieval_used else "",
        "knowledge_base_hash": active_kb_hash if runtime_active_kb_retrieval_used else "",
        "retrieved_card_ids": [str(card.get("card_id", "")) for card in reranked_cards[:5]],
        "payload_retrieved_card_count": len(base_cards),
        "api_called": False,
    }

def build_agent3_retrieval_trace(retrieval_round: dict[str, Any]) -> dict[str, Any]:
    """Freeze local query/retrieval diagnostics outside Agent3 semantic output."""
    trace = deepcopy(retrieval_round)
    trace["trace_kind"] = "agent3_retrieval_trace"
    trace["trace_only_fields"] = sorted(
        field for field in trace
        if field in AGENT3_RETRIEVAL_TRACE_ONLY_FIELDS
        or field == "retrieval_query_contract"
    )
    return trace

def build_agent3_committed_output(
    raw_output: dict[str, Any],
    *,
    retrieval_trace: dict[str, Any],
) -> dict[str, Any]:
    """Build the downstream Agent3 object without mutating the validated Raw JSON."""
    validate_agent3_raw_output(raw_output)
    raw_frozen = deepcopy(raw_output)
    query_contract = retrieval_trace.get("retrieval_query_contract", {})
    retrieval_plan = query_contract.get("retrieval_plan", {}) if isinstance(query_contract, dict) else {}
    committed = {
        **raw_frozen,
        "agent3_stageA_executed": True,
        "agent3_stageB_executed": True,
        "rag_trigger": "NEUTRAL_STAGE_A_GATE",
        "evidence_sufficiency": str(retrieval_plan.get("evidence_sufficiency", "INSUFFICIENT")),
        "modality_alignment": str(retrieval_plan.get("modality_alignment", "UNCERTAIN")),
        "overall_confidence": str(retrieval_plan.get("overall_confidence", "LOW")),
        "retrieval_decision": str(retrieval_plan.get("retrieval_decision", "SKIP")),
        "retrieval_reason": str(retrieval_plan.get("retrieval_reason", "NONE")),
        "deterministic_rag_gate_override": bool(retrieval_plan.get("deterministic_rag_gate_override", False)),
        "deterministic_rag_gate_override_reason": list(retrieval_plan.get("deterministic_rag_gate_override_reason", [])),
        "rag_used": True,
        "retrieval_executed": True,
        "retrieval_query_count": len(retrieval_trace.get("multi_query_bundle", [])),
        "retrieved_card_count": len(retrieval_trace.get("retrieved_cards", [])),
        "agent3b_skipped_reason": "",
        "cohort_prior_used": bool(retrieval_plan.get("source_cohort_factor_ids")),
        "cohort_prior_rerank_used": bool(retrieval_plan.get("cohort_prior_rerank_factor_ids")),
        "cohort_prior_context_used": bool(retrieval_plan.get("cohort_factor_context")),
        "cohort_prior_query_expansion_used": False,
        "cohort_prior_enters_supporting_cards": False,
        "cohort_prior_enters_applicable_card_ids": False,
        "agent3_uncertainty_reasons": list(raw_frozen.get("remaining_uncertainty", [])),
    }
    validate_agent3_committed_output(committed)
    return committed

def process_case(
    payload: dict[str, Any],
    *,
    agents: tuple[Any, ...],
    transaction: CaseTransaction,
    correction_policy: dict[str, Any] | None = None,
    progress: RuntimeProgress | None = None,
    knowledge_cards_file: Path | None = None,
) -> dict[str, Any]:
    agent1, agent2, agent3, agent4, controller = agents
    if progress is not None:
        progress.set_stage(str(payload.get("case_id", "")), "building_payload")
    assert_ecg_anchor_runtime_boundary(payload)
    payload["ecg_anchor_context"] = build_ecg_anchor_context(payload)
    transaction.initialize()
    a1 = transaction.load_committed("AGENT1_COMMITTED")
    if a1 is None:
        if progress is not None:
            progress.set_stage(str(payload.get("case_id", "")), "agent1_call")
        a1 = agent1.run(payload)
        transaction.commit("AGENT1_COMMITTED", a1)
    else:
        validate_v2_6_agent_output("agent1", a1)
    a2 = transaction.load_committed("AGENT2_COMMITTED")
    if a2 is None:
        if progress is not None:
            progress.set_stage(str(payload.get("case_id", "")), "agent2_call")
        a2 = agent2.run(payload)
        transaction.commit("AGENT2_COMMITTED", a2)
    else:
        validate_v2_6_agent_output("agent2", a2)
    triage = build_disagreement_triage_v2_6(payload, a1, a2)
    stage_a_plan = build_retrieval_plan_v2_6(payload, a1, a2, triage)
    if not bool(stage_a_plan.get("agent3b_rag_allowed")):
        a3_stage = "AGENT3_ROUND_0_COMMITTED"
        a4_stage = "AGENT4_REFLECTION_ROUND_0_COMMITTED"
        a3 = transaction.load_committed(a3_stage)
        if a3 is None:
            if progress is not None:
                progress.set_stage(str(payload.get("case_id", "")), "agent3_stage_a_gate_skip_rag")
            a3 = build_agent3_stage_a_only_output_v2_8(payload, a1, a2, triage, stage_a_plan)
            validate_agent3_committed_output(a3)
            transaction.commit(a3_stage, a3)
        else:
            validate_agent3_committed_output(a3)
        context = {
            **payload,
            "agent1_output": a1,
            "agent2_output": a2,
            "retrieved_cards": [],
            "retrieval_query_contract": {},
            "retrieval_round_index": 0,
            "triage": triage,
            "agent3_output": a3,
        }
        a4 = transaction.load_committed(a4_stage)
        if a4 is None:
            if progress is not None:
                progress.set_stage(str(payload.get("case_id", "")), "agent4_guardrail_round_0")
            a4 = agent4.run(context)
            transaction.commit(a4_stage, a4)
        else:
            validate_v2_6_agent_output("agent4", a4)
        rag_execution_summary = {
            "rag_used": False,
            "retrieval_executed": False,
            "agent3_stageB_executed": False,
            "agent3b_skipped_reason": "neutral_rag_gate_skip",
        }
    else:
        rag_stage = "RETRIEVAL_ROUND_0_COMMITTED"
        a3_stage = "AGENT3_ROUND_0_COMMITTED"
        a4_stage = "AGENT4_REFLECTION_ROUND_0_COMMITTED"
        rag = transaction.load_committed(rag_stage)
        if rag is None:
            rag = build_retrieval_round_v2_6(
                payload,
                agent1_output=a1,
                agent2_output=a2,
                triage=triage,
                knowledge_cards_file=knowledge_cards_file,
            )
            transaction.commit(rag_stage, rag)
        retrieval_trace = build_agent3_retrieval_trace(rag)
        context = {
            **payload,
            "agent1_output": a1,
            "agent2_output": a2,
            "retrieved_cards": rag["retrieved_cards"],
            "retrieval_query_contract": rag["retrieval_query_contract"],
            "retrieval_round_index": 0,
            "triage": triage,
        }
        a3 = transaction.load_committed(a3_stage)
        if a3 is None:
            if progress is not None:
                progress.set_stage(str(payload.get("case_id", "")), "agent3_call")
            a3_raw = agent3.run(context)
            a3 = build_agent3_committed_output(a3_raw, retrieval_trace=retrieval_trace)
            transaction.commit(a3_stage, a3)
        else:
            validate_agent3_committed_output(a3)
        context["agent3_output"] = a3
        a4 = transaction.load_committed(a4_stage)
        if a4 is None:
            if progress is not None:
                progress.set_stage(str(payload.get("case_id", "")), "agent4_guardrail")
            a4 = agent4.run(context)
            transaction.commit(a4_stage, a4)
        else:
            validate_v2_6_agent_output("agent4", a4)
        rag_execution_summary = {
            "rag_used": True,
            "retrieval_executed": True,
            "agent3_stageB_executed": True,
            "agent3b_skipped_reason": "",
        }
    eligibility = transaction.load_committed("ELIGIBILITY_COMMITTED")
    if eligibility is None:
        if progress is not None:
            progress.set_stage(str(payload.get("case_id", "")), "eligibility_gate")
        eligibility = controller.evaluate(
            agent1_output=a1, agent2_output=a2, agent3_output=a3,
            agent4_output=a4, ecg_anchor_context=payload.get("ecg_anchor_context", {}),
            correction_policy=correction_policy,
        )
        transaction.commit("ELIGIBILITY_COMMITTED", eligibility)
    ecg_anchor_context = payload.get("ecg_anchor_context", {})
    ecg_score = float(ecg_anchor_context.get("locked_ecg_score"))
    ordinal_correction = {
        "locked_ecg_score": ecg_score,
        "locked_ecg_threshold": float(ecg_anchor_context.get("locked_ecg_threshold")),
        "ecg_baseline_state": ecg_anchor_context.get("ecg_baseline_state"),
        **eligibility,
    }
    public = {
        "case_runtime_id": payload["case_id"],
        "runtime_version": runtime_version_label(correction_policy),
        "mas_mode": (
            f"{str((correction_policy or {}).get('dataset_version', 'v2_8'))}_final_single_controller_validation_locked_logit"
            if eligibility["correction_policy_locked"]
            else f"{str((correction_policy or {}).get('dataset_version', 'v2_8'))}_final_single_controller_unlocked_policy"
        ),
        "locked_ecg_score": ordinal_correction["locked_ecg_score"],
        "locked_ecg_threshold": ordinal_correction["locked_ecg_threshold"],
        "ecg_baseline_state": ordinal_correction["ecg_baseline_state"],
        "agent4_action_raw": eligibility["agent4_action_raw"],
        "controller_action": eligibility["controller_action"],
        "correction_eligible": eligibility["correction_eligible"],
        "correction_tier": eligibility["correction_tier"],
        "correction_policy_locked": eligibility["correction_policy_locked"],
        "applied_delta": eligibility["applied_delta"],
        "corrected_score": eligibility["corrected_score"],
        "controller_veto_reason": eligibility["controller_veto_reason"],
        "single_pass_agent3_agent4_contract": True,
        **rag_execution_summary,
        "agent3_stage_a_evidence_sufficiency": a3.get("evidence_sufficiency"),
        "agent3_modality_alignment": a3.get("modality_alignment"),
        "agent3_overall_confidence": a3.get("overall_confidence"),
        "agent3_retrieval_decision": a3.get("retrieval_decision"),
        "agent3_retrieval_reason": a3.get("retrieval_reason"),
        "deterministic_controller_authoritative": True,
    }
    if progress is not None:
        progress.set_stage(str(payload.get("case_id", "")), "writing_artifact")
    transaction.commit("PUBLIC_ARTIFACT_COMMITTED", public)
    transaction.commit("CASE_COMPLETE", {"case_runtime_id": payload["case_id"], "status": "complete"})
    return public

class RuntimeProgress:
    """Console-only progress display; it never changes runtime artifacts or API order."""

    def __init__(self, *, total: int, initial_completed: int = 0, enabled: bool = True, refresh_seconds: float = 1.0) -> None:
        self.total = int(total)
        self.completed = int(initial_completed)
        self.failed = 0
        self.enabled = bool(enabled)
        self.refresh_seconds = max(0.1, float(refresh_seconds))
        self.started = time.perf_counter()
        self.last_refresh = 0.0
        self.current_case = ""
        self.current_stage = "initializing"
        self._pbar = None
        if self.enabled:
            try:
                from tqdm import tqdm  # type: ignore

                self._pbar = tqdm(
                    total=self.total,
                    initial=min(self.completed, self.total),
                    desc="v2.8 runtime",
                    unit="case",
                    dynamic_ncols=True,
                    leave=True,
                )
            except Exception:
                self._pbar = None
                self._write_plain(force=True)

    def _stats(self) -> dict[str, Any]:
        elapsed = max(0.0, time.perf_counter() - self.started)
        processed = max(0, self.completed + self.failed)
        remaining = max(0, self.total - processed)
        avg_s = elapsed / processed if processed else 0.0
        eta_s = int(remaining * avg_s) if processed else 0
        return {
            "completed": self.completed,
            "failed": self.failed,
            "remaining": remaining,
            "elapsed_s": int(elapsed),
            "avg_s": round(avg_s, 1),
            "eta": time.strftime("%H:%M:%S", time.gmtime(eta_s)),
            "current": self.current_case,
            "stage": self.current_stage,
        }

    def _write_plain(self, *, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self.last_refresh < self.refresh_seconds:
            return
        self.last_refresh = now
        stats = self._stats()
        print(
            "v2.8 runtime progress "
            f"completed={stats['completed']}/{self.total} "
            f"failed={stats['failed']} remaining={stats['remaining']} "
            f"avg_s={stats['avg_s']} eta={stats['eta']} "
            f"current={stats['current']} stage={stats['stage']}",
            flush=True,
        )

    def _refresh(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        stats = self._stats()
        if self._pbar is not None:
            now = time.perf_counter()
            if force or now - self.last_refresh >= self.refresh_seconds:
                self.last_refresh = now
                self._pbar.set_postfix_str(
                    f"completed={stats['completed']}, failed={stats['failed']}, "
                    f"remaining={stats['remaining']}, avg_s={stats['avg_s']}, "
                    f"eta={stats['eta']}, current={stats['current']}, stage={stats['stage']}",
                    refresh=True,
                )
            return
        self._write_plain(force=force)

    def set_stage(self, case_id: str, stage: str) -> None:
        self.current_case = str(case_id)
        self.current_stage = str(stage)
        self._refresh(force=True)

    def case_done(self, case_id: str, *, already_counted: bool = False) -> None:
        self.current_case = str(case_id)
        self.current_stage = "case_complete"
        if not already_counted:
            self.completed += 1
            if self._pbar is not None:
                self._pbar.update(1)
        self._refresh(force=True)

    def case_failed(self, case_id: str) -> None:
        self.current_case = str(case_id)
        self.current_stage = "case_failed"
        self.failed += 1
        if self._pbar is not None:
            self._pbar.update(1)
        self._refresh(force=True)

    def close(self) -> None:
        if self._pbar is not None:
            self._pbar.close()
