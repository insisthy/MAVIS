from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from dotenv import dotenv_values, load_dotenv

from .json_parser import parse_json_object


@dataclass
class LLMConfig:
    provider: str
    api_base: str
    api_key: str
    model: str
    timeout_seconds: int | tuple[int, int] = 60
    connect_timeout_seconds: int = 30
    read_timeout_seconds: int = 60
    max_retries: int = 3
    temperature: float = 0.0
    env_path: str = ""
    env_file_exists: bool = False
    env_file_loaded: bool = False
    api_key_source: str = ""
    api_base_source: str = ""
    model_source: str = ""


class StreamingTransportError(RuntimeError):
    """Network failure during SSE with any already-received content attached."""

    def __init__(self, message: str, *, partial_content: str, stream_telemetry: dict[str, Any], response: Any = None):
        super().__init__(message)
        self.partial_content = partial_content
        self.stream_telemetry = stream_telemetry
        self.response = response


class StreamingProtocolError(RuntimeError):
    """Provider SSE framing failed before the Agent JSON contract was reached."""

    transport_protocol_failure = True

    def __init__(self, message: str, *, partial_content: str, stream_telemetry: dict[str, Any], response: Any = None):
        super().__init__(message)
        self.partial_content = partial_content
        self.stream_telemetry = stream_telemetry
        self.response = response


class StreamingTotalDurationExceeded(RuntimeError):
    """A live SSE response exceeded its explicit end-to-end duration budget."""

    transport_protocol_failure = True

    def __init__(self, message: str, *, partial_content: str, stream_telemetry: dict[str, Any], response: Any = None):
        super().__init__(message)
        self.partial_content = partial_content
        self.stream_telemetry = stream_telemetry
        self.response = response


def _first_env(names: list[str]) -> str:
    for name in names:
        value = os.getenv(name, "")
        if value:
            return value
    return ""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_env_path(env_path: str | Path | None = None) -> Path:
    if env_path is None:
        return _project_root() / ".env"
    path = Path(env_path)
    if path.is_absolute():
        return path
    return _project_root() / path


def _first_config_value(names: list[str], env_values: dict[str, Any], existing_env: dict[str, str], default: str = "") -> tuple[str, str]:
    for name in names:
        value = existing_env.get(name, "")
        if value:
            return value, f"environment:{name}"
    for name in names:
        value = str(env_values.get(name, "") or "")
        if value:
            return value, f".env:{name}"
    return default, "default" if default else ""


def load_llm_config(env_path: str | Path | None = None) -> LLMConfig:
    resolved_env_path = _resolve_env_path(env_path)
    env_file_exists = resolved_env_path.exists()
    env_values = dotenv_values(resolved_env_path) if env_file_exists else {}
    existing_env = dict(os.environ)
    env_file_loaded = bool(load_dotenv(resolved_env_path, override=False)) if env_file_exists else False
    provider, provider_source = _first_config_value(["LLM_PROVIDER", "DEEPSEEK_PROVIDER"], env_values, existing_env, "deepseek")
    api_base, api_base_source = _first_config_value(
        ["DEEPSEEK_BASE_URL", "DEEPSEEK_API_BASE", "LLM_API_BASE", "LLM_BASE_URL"],
        env_values,
        existing_env,
        "https://api.deepseek.com",
    )
    api_key, api_key_source = _first_config_value(
        ["DEEPSEEK_API_KEY", "LLM_API_KEY", "SILICONFLOW_API_KEY", "OPENAI_API_KEY", "DASHSCOPE_API_KEY"],
        env_values,
        existing_env,
    )
    model, model_source = _first_config_value(["DEEPSEEK_MODEL", "LLM_MODEL"], env_values, existing_env)
    timeout_seconds, _ = _first_config_value(["LLM_TIMEOUT_SECONDS"], env_values, existing_env, "60")
    max_retries, _ = _first_config_value(["LLM_MAX_RETRIES"], env_values, existing_env, "3")
    temperature, _ = _first_config_value(["LLM_TEMPERATURE"], env_values, existing_env, "0.0")
    return LLMConfig(
        provider=provider,
        api_base=api_base,
        api_key=api_key,
        model=model,
        timeout_seconds=int(timeout_seconds),
        max_retries=int(max_retries),
        temperature=float(temperature),
        env_path=str(resolved_env_path),
        env_file_exists=env_file_exists,
        env_file_loaded=env_file_loaded,
        api_key_source=api_key_source,
        api_base_source=api_base_source,
        model_source=model_source,
    )


def redact_api_key(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "***REDACTED***"
    return f"{value[:3]}***REDACTED***{value[-4:]}"


def inspect_llm_config(env_path: str | Path | None = None) -> dict[str, Any]:
    cfg = load_llm_config(env_path)
    env_values = dotenv_values(cfg.env_path) if cfg.env_file_exists else {}
    variable_names = [
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
        "LLM_API_KEY",
        "LLM_API_BASE",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "SILICONFLOW_API_KEY",
        "DASHSCOPE_API_KEY",
    ]
    presence = {
        name: bool(os.environ.get(name, "") or str(env_values.get(name, "") or ""))
        for name in variable_names
    }
    return {
        "mode": "inspect_llm_config",
        "llm_api_called": False,
        "env_file_path": cfg.env_path,
        "env_file_exists": cfg.env_file_exists,
        "env_file_loaded": cfg.env_file_loaded,
        "env_variable_presence": presence,
        "api_key_variable_checked": [
            "DEEPSEEK_API_KEY",
            "LLM_API_KEY",
            "SILICONFLOW_API_KEY",
            "OPENAI_API_KEY",
            "DASHSCOPE_API_KEY",
        ],
        "api_key_present": bool(cfg.api_key),
        "api_key_source": cfg.api_key_source,
        "api_key_preview": redact_api_key(cfg.api_key),
        "base_url": cfg.api_base,
        "base_url_source": cfg.api_base_source,
        "model": cfg.model,
        "model_source": cfg.model_source,
        "client_type": "OpenAICompatibleClient",
        "provider": cfg.provider,
        "ready_for_runtime": bool(cfg.api_key and cfg.api_base and cfg.model),
    }


class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat completion client."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def complete(self, prompt: str, payload: dict[str, Any] | None = None) -> str:
        if not self.config.api_base or not self.config.api_key or not self.config.model:
            raise RuntimeError(
                "No complete LLM config found. Checked DEEPSEEK_API_KEY in environment and project .env "
                f"plus compatible aliases. Please configure DEEPSEEK_API_KEY in {self.config.env_path} "
                "or set it in the current shell, and configure DEEPSEEK_BASE_URL/DEEPSEEK_MODEL or compatible aliases."
            )
        url = self.config.api_base.rstrip("/") + "/chat/completions"
        body = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload or {}, ensure_ascii=False, default=str)},
            ],
        }
        headers = {"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"}
        session = requests.Session()
        session.trust_env = False
        response = session.post(url, json=body, headers=headers, timeout=self.config.timeout_seconds)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def complete_streaming_json(
        self,
        prompt: str,
        payload: dict[str, Any] | None = None,
        *,
        max_tokens: int | None = None,
        max_stream_duration_seconds: float | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Call an OpenAI-compatible endpoint with SSE streaming.

        The returned text is assembled only from ``choices[].delta.content``.
        Provider-specific ``reasoning_content`` is counted for telemetry but is
        not appended to the JSON text consumed by runtime validators.
        """
        if not self.config.api_base or not self.config.api_key or not self.config.model:
            raise RuntimeError(
                "No complete LLM config found. Checked DEEPSEEK_API_KEY in environment and project .env "
                f"plus compatible aliases. Please configure DEEPSEEK_API_KEY in {self.config.env_path} "
                "or set it in the current shell, and configure DEEPSEEK_BASE_URL/DEEPSEEK_MODEL or compatible aliases."
            )
        url = self.config.api_base.rstrip("/") + "/chat/completions"
        body: dict[str, Any] = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "response_format": {"type": "json_object"},
            "stream": True,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload or {}, ensure_ascii=False, default=str)},
            ],
        }
        if max_tokens is not None:
            body["max_tokens"] = int(max_tokens)
        if extra_body:
            body.update(extra_body)
        headers = {"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"}
        session = requests.Session()
        session.trust_env = False
        started = time.perf_counter()
        first_byte_at: float | None = None
        first_token_at: float | None = None
        raw_stream_bytes = 0
        content_parts: list[str] = []
        reasoning_content_bytes = 0
        delta_chunk_count = 0
        finish_reason = ""
        status_code = ""
        provider_request_id = ""
        stream_done_seen = False
        deadline_complete_json_accepted = False
        sse_event_count = 0
        sse_multiline_event_count = 0
        sse_multiline_compact_join_count = 0
        headers_at: float | None = None
        response: Any = None

        def build_telemetry(
            ended: float,
            *,
            partial_response_persisted: bool,
            malformed_sse_event_fragment: str = "",
        ) -> dict[str, Any]:
            content = "".join(content_parts)
            return {
                "streaming": True,
                "response_headers_received": headers_at is not None,
                "response_headers_latency_seconds": round(headers_at - started, 6) if headers_at is not None else None,
                "first_byte_latency_seconds": round(first_byte_at - started, 6) if first_byte_at is not None else None,
                "first_token_latency_seconds": round(first_token_at - started, 6) if first_token_at is not None else None,
                "total_latency_seconds": round(ended - started, 6),
                "raw_stream_bytes": raw_stream_bytes,
                "content_bytes": len(content.encode("utf-8")),
                "reasoning_content_bytes": reasoning_content_bytes,
                "delta_chunk_count": delta_chunk_count,
                "configured_max_output_tokens": max_tokens,
                "configured_max_stream_duration_seconds": max_stream_duration_seconds,
                "stream_duration_limit_exceeded": bool(
                    max_stream_duration_seconds is not None
                    and ended - started >= float(max_stream_duration_seconds)
                    and not stream_done_seen
                ),
                "request_body_contract": {
                    "response_format": body.get("response_format"),
                    "stream": body.get("stream"),
                    "max_tokens": body.get("max_tokens"),
                    "thinking": body.get("thinking"),
                },
                "finish_reason": finish_reason,
                "stream_done_seen": stream_done_seen,
                "deadline_complete_json_accepted": deadline_complete_json_accepted,
                "sse_event_count": sse_event_count,
                "sse_multiline_event_count": sse_multiline_event_count,
                "sse_multiline_compact_join_count": sse_multiline_compact_join_count,
                "sse_wire_line_delimiter": "LF_BYTES",
                "sse_wire_text_encoding": "utf-8",
                "malformed_sse_event_fragment": malformed_sse_event_fragment[:4096],
                "http_status": status_code,
                "provider_request_id": provider_request_id,
                "partial_response_persisted": partial_response_persisted,
            }

        def consume_sse_event(data_lines: list[str]) -> bool:
            """Consume one complete SSE event, delimited by a blank line."""
            nonlocal delta_chunk_count
            nonlocal finish_reason
            nonlocal first_token_at
            nonlocal reasoning_content_bytes
            nonlocal sse_event_count
            nonlocal sse_multiline_compact_join_count
            nonlocal sse_multiline_event_count
            nonlocal stream_done_seen

            if not data_lines:
                return False
            sse_event_count += 1
            if len(data_lines) > 1:
                sse_multiline_event_count += 1
            data_text = "\n".join(data_lines).strip()
            if data_text == "[DONE]":
                stream_done_seen = True
                return True
            try:
                event = json.loads(data_text)
            except json.JSONDecodeError as exc:
                compact_text = "".join(data_lines).strip()
                if len(data_lines) > 1 and compact_text != data_text:
                    try:
                        event = json.loads(compact_text)
                    except json.JSONDecodeError:
                        event = None
                    else:
                        sse_multiline_compact_join_count += 1
                else:
                    event = None
                if event is None:
                    ended = time.perf_counter()
                    raise StreamingProtocolError(
                        f"SSE_EVENT_JSON_DECODE_ERROR: {exc}",
                        partial_content="".join(content_parts),
                        stream_telemetry=build_telemetry(
                            ended,
                            partial_response_persisted=bool(content_parts),
                            malformed_sse_event_fragment=data_text,
                        ),
                        response=response,
                    ) from exc
            choices = event.get("choices") or []
            if not choices:
                return False
            choice0 = choices[0]
            finish_reason = str(choice0.get("finish_reason") or finish_reason or "")
            delta = choice0.get("delta") or {}
            reasoning = delta.get("reasoning_content")
            if reasoning:
                reasoning_content_bytes += len(str(reasoning).encode("utf-8"))
            content = delta.get("content")
            if content:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                delta_chunk_count += 1
                content_parts.append(str(content))
            return False

        def enforce_stream_deadline(checked_at: float) -> bool:
            """Fail at an SSE event boundary, or accept a complete final JSON.

            The current wire line is consumed before this check.  This avoids
            discarding the final content or ``[DONE]`` line merely because it
            arrived on the duration boundary.  A late response is accepted
            without repair only when the provider already emitted
            ``finish_reason=stop`` and the accumulated body is a complete JSON
            object; downstream Agent schema validation remains authoritative.
            """

            nonlocal deadline_complete_json_accepted
            if (
                max_stream_duration_seconds is None
                or checked_at - started < float(max_stream_duration_seconds)
                or stream_done_seen
            ):
                return False
            content = "".join(content_parts)
            if str(finish_reason or "").strip().lower() == "stop" and content.strip():
                try:
                    parsed = json.loads(content)
                except (json.JSONDecodeError, TypeError, ValueError):
                    parsed = None
                if isinstance(parsed, dict):
                    deadline_complete_json_accepted = True
                    return True
            raise StreamingTotalDurationExceeded(
                "STREAM_TOTAL_DURATION_EXCEEDED:"
                f"limit={float(max_stream_duration_seconds):g}s,elapsed={checked_at - started:.6f}s",
                partial_content=content,
                stream_telemetry=build_telemetry(
                    checked_at,
                    partial_response_persisted=bool(content),
                ),
                response=response,
            )

        event_data_lines: list[str] = []
        try:
            with session.post(
                url,
                json=body,
                headers=headers,
                timeout=self.config.timeout_seconds,
                stream=True,
            ) as response:
                status_code = str(response.status_code)
                provider_request_id = response.headers.get("x-request-id") or response.headers.get("x-sf-request-id") or ""
                response.raise_for_status()
                headers_at = time.perf_counter()
                # Parse the wire protocol as bytes and split only on the SSE
                # LF delimiter.  requests.iter_lines(decode_unicode=True)
                # delegates to str.splitlines(); when a provider omits an
                # explicit UTF-8 charset, UTF-8 continuation byte 0x85 can be
                # decoded as U+0085 and incorrectly treated as a line break
                # inside a JSON string.  That produced the recurrent
                # "unterminated string at column 202" failures.
                for line in response.iter_lines(decode_unicode=False, delimiter=b"\n"):
                    now = time.perf_counter()
                    if first_byte_at is None:
                        first_byte_at = now
                    if isinstance(line, bytes):
                        raw_stream_bytes += len(line)
                        wire_line = line[:-1] if line.endswith(b"\r") else line
                        try:
                            text_line = wire_line.decode("utf-8", errors="strict")
                        except UnicodeDecodeError as exc:
                            ended = time.perf_counter()
                            raise StreamingProtocolError(
                                f"SSE_WIRE_UTF8_DECODE_ERROR: {exc}",
                                partial_content="".join(content_parts),
                                stream_telemetry=build_telemetry(
                                    ended,
                                    partial_response_persisted=bool(content_parts),
                                    malformed_sse_event_fragment="raw_hex:" + wire_line[:2048].hex(),
                                ),
                                response=response,
                            ) from exc
                    else:
                        text_line = str(line).removesuffix("\r")
                        raw_stream_bytes += len(text_line.encode("utf-8"))
                    deadline_checkpoint = False
                    if text_line == "":
                        deadline_checkpoint = True
                        if event_data_lines:
                            done = consume_sse_event(event_data_lines)
                            event_data_lines = []
                            if done:
                                break
                    elif text_line.startswith(":"):
                        deadline_checkpoint = True
                    elif text_line.startswith("data:"):
                        event_data_lines.append(text_line[5:].lstrip())
                    else:
                        deadline_checkpoint = True
                    if deadline_checkpoint and enforce_stream_deadline(now):
                        break
                if not stream_done_seen and event_data_lines:
                    consume_sse_event(event_data_lines)
                if enforce_stream_deadline(time.perf_counter()):
                    pass
        except requests.RequestException as exc:
            if event_data_lines:
                try:
                    consume_sse_event(event_data_lines)
                except StreamingProtocolError:
                    # The network exception remains the primary failure.  An
                    # incomplete trailing SSE event is retained in telemetry
                    # by the transport failure record and is never treated as
                    # an Agent JSON/Schema error.
                    pass
            ended = time.perf_counter()
            partial_content = "".join(content_parts)
            raise StreamingTransportError(
                f"{type(exc).__name__}: {exc}",
                partial_content=partial_content,
                stream_telemetry=build_telemetry(ended, partial_response_persisted=bool(partial_content)),
                response=getattr(exc, "response", response),
            ) from exc
        ended = time.perf_counter()
        content = "".join(content_parts)
        telemetry = build_telemetry(ended, partial_response_persisted=False)
        return content, telemetry

    def complete_json(self, prompt: str, payload: dict[str, Any] | None = None) -> tuple[dict[str, Any], bool]:
        return parse_json_object(self.complete(prompt, payload))
