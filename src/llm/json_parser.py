from __future__ import annotations

import json
import re
from typing import Any


def parse_json_object(text: str) -> tuple[dict[str, Any], bool]:
    """Parse a JSON object and return (data, parse_error)."""
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data, False
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data, False
        except json.JSONDecodeError:
            pass
    return {}, True
