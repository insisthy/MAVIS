from __future__ import annotations

from collections import Counter

from .knowledge_card import KnowledgeCard


def _tokens(text: str) -> list[str]:
    return [t.strip(" ,.;:()[]{}").lower() for t in text.split() if t.strip()]


class KeywordRetriever:
    """Simple deterministic keyword retriever for local reproducibility."""

    def __init__(self, cards: list[KnowledgeCard]):
        self.cards = cards

    def retrieve(self, query: str, top_k: int = 5) -> list[KnowledgeCard]:
        q = Counter(_tokens(query))
        scored: list[tuple[int, KnowledgeCard]] = []
        for card in self.cards:
            text = " ".join([card.title, card.content, " ".join(card.keywords)]).lower()
            score = sum(count for token, count in q.items() if token and token in text)
            if score > 0:
                scored.append((score, card))
        scored.sort(key=lambda item: (-item[0], item[1].card_id))
        return [card for _, card in scored[:top_k]]
