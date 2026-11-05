"""Canonical, non-secret fingerprints for LLM inference and transport settings.

These fingerprints deliberately answer two different governance questions:
``model_inference`` records settings that can change generated content, while
``transport_execution`` records how the request is delivered.  Do not combine
them into one opaque value.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit


FINGERPRINT_SCHEMA_VERSION = "llm_config_fingerprint_v4"
FINGERPRINT_HASH_ALGORITHM = "sha256"
_SECRET_KEYS = {"api_key", "authorization", "token", "secret", "password", "access_token", "bearer_token"}


def _normalise_url(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parsed = urlsplit(text)
    # Scheme and host are case-insensitive; a terminal slash is not a distinct
    # OpenAI-compatible API endpoint.  Path case/query/fragment are preserved.
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.query, parsed.fragment))


def _normalise_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("fingerprint values must be finite")
        return int(value) if value.is_integer() else value
    if isinstance(value, Mapping):
        return {str(key): _normalise_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise_value(item) for item in value]
    return str(value)


def _assert_no_secret_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in _SECRET_KEYS:
                raise ValueError(f"secret-like field is forbidden in fingerprint: {key}")
            _assert_no_secret_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_secret_keys(item)


def _stable_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonicalize_model_inference_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact non-secret LLM generation settings used by runtime."""
    _assert_no_secret_keys(config)
    candidate = {
        "provider": config.get("provider", ""),
        "normalized_base_url": _normalise_url(config.get("api_base", config.get("base_url", ""))),
        "model": config.get("model", ""),
        "temperature": config.get("temperature"),
        "response_format": config.get("response_format", "json_object"),
        "agent1_3_thinking_mode": config.get("agent1_3_thinking_mode") or None,
        "agent1_3_reasoning_effort": config.get("agent1_3_reasoning_effort") or None,
        "agent4_thinking_mode": config.get("agent4_thinking_mode") or None,
        "agent4_max_output_tokens": config.get("agent4_max_output_tokens") or None,
    }
    # Only add knobs that are genuinely supplied by this runtime implementation.
    for key in ("top_p", "seed"):
        if config.get(key) is not None:
            candidate[key] = config[key]
    _assert_no_secret_keys(candidate)
    return _normalise_value(candidate)


def hash_model_inference_config(config: Mapping[str, Any]) -> str:
    return _stable_hash(canonicalize_model_inference_config(config))


def canonicalize_transport_execution_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return request-delivery settings, excluding model semantics and secrets."""
    _assert_no_secret_keys(config)
    candidate = {
        "client_class": config.get("client_class", ""),
        "transport_adapter": config.get("transport_adapter", ""),
        "connect_timeout_seconds": config.get("connect_timeout_seconds"),
        "read_timeout_seconds": config.get("read_timeout_seconds"),
        "max_technical_retries_per_call": config.get("max_technical_retries_per_call"),
        "technical_retry_backoff_seconds": config.get("technical_retry_backoff_seconds", []),
        "agent1_3_streaming_enabled": bool(config.get("agent1_3_streaming_enabled", False)),
        "agent1_3_max_stream_duration_seconds": config.get("agent1_3_max_stream_duration_seconds"),
        "agent4_streaming_enabled": bool(config.get("agent4_streaming_enabled", False)),
        "agent4_max_total_attempts": config.get("agent4_max_total_attempts") or None,
    }
    _assert_no_secret_keys(candidate)
    return _normalise_value(candidate)


def hash_transport_execution_config(config: Mapping[str, Any]) -> str:
    return _stable_hash(canonicalize_transport_execution_config(config))


def build_runtime_fingerprint_components(
    llm_config: Any,
    execution: Mapping[str, Any],
    *,
    client_class: str = "OpenAICompatibleClient",
    transport_adapter: str = "RawResponsePersistingClient",
) -> dict[str, Any]:
    """Build both authoritative canonical objects from resolved runtime values."""
    inference_input = {
        "provider": getattr(llm_config, "provider", ""),
        "api_base": getattr(llm_config, "api_base", ""),
        "model": getattr(llm_config, "model", ""),
        "temperature": getattr(llm_config, "temperature", None),
        "response_format": "json_object",
        "agent1_3_thinking_mode": execution.get("agent1_3_thinking_mode"),
        "agent1_3_reasoning_effort": execution.get("agent1_3_reasoning_effort"),
        "agent4_thinking_mode": execution.get("agent4_thinking_mode"),
        "agent4_max_output_tokens": execution.get("agent4_max_output_tokens"),
    }
    transport_input = {
        "client_class": client_class,
        "transport_adapter": transport_adapter,
        "connect_timeout_seconds": execution.get("connect_timeout_seconds"),
        "read_timeout_seconds": execution.get("read_timeout_seconds"),
        "max_technical_retries_per_call": execution.get("max_technical_retries_per_call"),
        "technical_retry_backoff_seconds": execution.get("technical_retry_backoff_seconds", []),
        "agent1_3_streaming_enabled": execution.get("agent1_3_streaming_enabled", False),
        "agent1_3_max_stream_duration_seconds": execution.get("agent1_3_max_stream_duration_seconds"),
        "agent4_streaming_enabled": execution.get("agent4_streaming_enabled", False),
        "agent4_max_total_attempts": execution.get("agent4_max_total_attempts"),
    }
    model_canonical = canonicalize_model_inference_config(inference_input)
    transport_canonical = canonicalize_transport_execution_config(transport_input)
    return {
        "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        "fingerprint_hash_algorithm": FINGERPRINT_HASH_ALGORITHM,
        "model_inference_config_canonical": model_canonical,
        "model_inference_config_hash": _stable_hash(model_canonical),
        "transport_execution_config_canonical": transport_canonical,
        "transport_execution_config_hash": _stable_hash(transport_canonical),
    }
