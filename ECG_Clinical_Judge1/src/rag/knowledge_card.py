from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .canonical_card_adapter import adapt_card_to_canonical_schema


@dataclass(frozen=True)
class KnowledgeCard:
    card_id: str
    title: str
    category: str
    keywords: list[str]
    content: str
    risk_direction: str = "preserve"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeCard":
        data = adapt_card_to_canonical_schema(data)
        return cls(
            card_id=str(data["card_id"]),
            title=str(data["title"]),
            category=str(data.get("category", "")),
            keywords=list(data.get("keywords", [])),
            content=str(data.get("content", "")),
            risk_direction=str(data.get("risk_direction", "preserve")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "title": self.title,
            "category": self.category,
            "keywords": self.keywords,
            "content": self.content,
            "risk_direction": self.risk_direction,
        }
