"""Structural wire revision only: no clinical defaults or semantic regeneration."""
from copy import deepcopy
import json
import math

from src.llm.llm_client import OpenAICompatibleClient

def require(condition, message):
    if not condition:
        raise WireError(message)
from src.agents.schemas import (
    V2_6_AGENT_STAGE_SCHEMAS, V2_6_SAFE_BOUNDARY_CHECK,
    build_agent_raw_prompt_schema, declared_schema_fields,
)

PROTOCOL = "legacy_v1_structured_wire_v2"


class WireError(RuntimeError):
    pass


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise WireError("DUPLICATE_JSON_KEY:" + key)
            result[key] = value
        return result
    def reject(value):
        raise WireError("NONFINITE_JSON:" + value)
    try:
        packet = json.loads(text, object_pairs_hook=pairs, parse_constant=reject)
    except (TypeError, ValueError) as exc:
        raise WireError("INVALID_JSON:" + str(exc)) from exc
    if not isinstance(packet, dict):
        raise WireError("JSON_ROOT_NOT_OBJECT")
    def finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise WireError("NONFINITE_JSON")
        if isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)
    finite(packet)
    return packet


def wire_schema(agent_id):
    if agent_id in {"agent1", "agent2"}:
        metadata = deepcopy(V2_6_AGENT_STAGE_SCHEMAS[agent_id + ".evidence_pass"])
    else:
        metadata = build_agent_raw_prompt_schema(agent_id)
    properties = {}
    for field in sorted(declared_schema_fields(metadata)):
        if field in metadata.get("enums", {}):
            prop = {"type": "string", "enum": metadata["enums"][field]}
        elif field in metadata.get("booleans", []) + metadata.get("optional_booleans", []):
            prop = {"type": "boolean"}
        elif field in metadata.get("lists", []) + metadata.get("optional_lists", []):
            # Legacy validators constrain array containers, not item types.
            prop = {"type": "array", "items": {}}
        elif field in metadata.get("numbers", []) + metadata.get("optional_numbers", []):
            prop = {"type": "number"}
        elif field == "safety_boundary_check":
            boundary = {key: {"type": "boolean" if isinstance(value, bool) else "string"}
                        for key, value in V2_6_SAFE_BOUNDARY_CHECK.items()}
            prop = {"type": "object", "properties": boundary, "required": sorted(boundary), "additionalProperties": False}
        elif field == "agent_name":
            prop = {"type": "string", "enum": [metadata["agent_name"]]}
        else:
            # Do not invent string-only requirements for generic legacy prose.
            prop = {}
        properties[field] = prop
    return {"type": "object", "properties": properties,
            "required": sorted(metadata["required"]), "additionalProperties": False}


def validate_structure(value, schema, path="$"):
    """Validate only the small JSON Schema vocabulary emitted above."""
    kind = schema.get("type")
    valid = {"object": lambda: isinstance(value, dict), "array": lambda: isinstance(value, list),
             "boolean": lambda: type(value) is bool, "string": lambda: isinstance(value, str),
             "number": lambda: type(value) in (int, float) and math.isfinite(value)}
    if kind and (kind not in valid or not valid[kind]()):
        raise WireError("WIRE_TYPE:" + path + ":" + str(kind))
    if "enum" in schema and value not in schema["enum"]:
        raise WireError("WIRE_ENUM:" + path)
    if isinstance(value, dict) and kind == "object":
        missing = set(schema.get("required", [])) - set(value)
        extra = set(value) - set(schema.get("properties", {}))
        if missing:
            raise WireError("WIRE_MISSING:" + path + ":" + ",".join(sorted(missing)))
        if extra and schema.get("additionalProperties") is False:
            raise WireError("WIRE_EXTRA:" + path + ":" + ",".join(sorted(extra)))
        for key, item in value.items():
            if key in schema.get("properties", {}):
                validate_structure(item, schema["properties"][key], path + "." + key)
    if isinstance(value, list) and kind == "array":
        for index, item in enumerate(value):
            validate_structure(item, schema.get("items", {}), path + f"[{index}]")
    if isinstance(value, float) and not math.isfinite(value):
        raise WireError("NONFINITE_JSON:" + path)


def response_format(agent_id):
    return {"type": "json_schema", "json_schema": {
        "name": PROTOCOL + "_" + agent_id, "strict": True, "schema": wire_schema(agent_id)}}


class StructuredClient(OpenAICompatibleClient):
    def complete_streaming_json(self, prompt, payload=None, *, max_tokens=None,
                                max_stream_duration_seconds=None, extra_body=None):
        agent = (payload or {}).get("agent_id")
        if agent not in {"agent1", "agent2", "agent3", "agent4"}:
            raise WireError("NON_PRIMARY_REQUEST_REQUIRES_REVIEW; no new repair or sampling")
        body = deepcopy(extra_body or {})
        require("response_format" not in body, "WIRE_FORMAT_ALREADY_OVERRIDDEN")
        body["response_format"] = response_format(agent)
        return super().complete_streaming_json(prompt, payload, max_tokens=max_tokens,
            max_stream_duration_seconds=max_stream_duration_seconds, extra_body=body)


