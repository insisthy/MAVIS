from __future__ import annotations

from typing import Any


def classify_dataset_v4_missingness(agent2_output: dict[str, Any] | None) -> dict[str, Any]:
    """Separate reported Dataset-v4 missingness from action-blocking missingness.

    The active Dataset-v4 payload has no validated rule that can infer a
    patient-level safety contraindication from an unavailable field alone.
    Existing missing IDs are therefore structural caution fields by default.
    Only a future validated local rule may explicitly mark a reported ID as
    action-blocking.
    """

    source = agent2_output or {}
    reported = list(dict.fromkeys(map(str, source.get("critical_missing_factor_ids", []) or [])))
    explicitly_blocking = set(map(str, source.get("action_blocking_missing_factor_ids", []) or []))
    blocking = [factor_id for factor_id in reported if factor_id in explicitly_blocking]
    structural = [factor_id for factor_id in reported if factor_id not in explicitly_blocking]
    return {
        "reported_missing_factor_ids": reported,
        "structural_missing_factor_ids": structural,
        "action_blocking_missing_factor_ids": blocking,
        "missingness_action_policy": "STRUCTURAL_CAUTION_NOT_ACTION_VETO",
    }
