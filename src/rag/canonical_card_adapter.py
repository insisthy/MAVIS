from __future__ import annotations

from typing import Any


ACTION_HINT_FIELDS = {
    "agent_action_hint",
    "action_hint",
    "recommended_action",
    "policy_action_hint",
    "delta_hint",
}

CARD_GROUP_TO_CATEGORY = {
    "A_ecg_false_positive": "ecg_false_positive_mechanism",
    "B_ecg_false_negative": "ecg_false_negative_mechanism",
    "C_clinical_context": "clinical_context",
    "D_discordance_rule": "governance_reference",
    "E_safety": "governance_reference",
    "E_data_quality": "governance_reference",
}


def adapt_card_to_canonical_schema(raw: dict[str, Any]) -> dict[str, Any]:
    """Project a legacy/raw card row into the active retrieval card schema.

    The active ECG-anchor runtime allows RAG to provide mechanism/background
    evidence only. Historical card-level action hints are intentionally blocked
    so Agent4 remains the sole semantic action source.
    """

    data = dict(raw)
    for field in ACTION_HINT_FIELDS:
        data.pop(field, None)
    category = str(data.get("category") or "").strip()
    card_group = str(data.get("card_group") or "").strip()
    if not category and card_group:
        category = CARD_GROUP_TO_CATEGORY.get(card_group, "medical_mechanism")
    title = data.get("title") or data.get("card_title") or data.get("source_title") or data.get("card_id")
    content = (
        data.get("content")
        or data.get("knowledge_card")
        or data.get("evidence_summary")
        or data.get("key_evidence_summary")
        or data.get("excerpt")
        or ""
    )
    keywords = data.get("keywords") or data.get("retrieval_keywords") or []
    if isinstance(keywords, str):
        keywords = [item.strip() for item in keywords.replace(";", ",").split(",") if item.strip()]
    return {
        **data,
        "card_id": str(data.get("card_id") or data.get("target_card_id") or ""),
        "title": str(title or ""),
        "category": category,
        "keywords": list(keywords),
        "content": str(content or ""),
        "risk_direction": "preserve",
        "direct_action_hints_blocked": True,
    }
