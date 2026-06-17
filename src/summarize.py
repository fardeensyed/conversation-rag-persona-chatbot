from __future__ import annotations

import math
from collections import Counter

from .utils import top_keywords, tokenize


def _sentence_score(text: str, global_counts: Counter[str], position: int, total: int) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    freq_score = sum(global_counts[token] for token in tokens) / math.sqrt(len(tokens))
    position_bonus = 1.15 if position in (0, total - 1) else 1.0
    return freq_score * position_bonus


def summarize_messages(messages: list[dict], max_sentences: int = 3, max_chars: int = 650) -> str:
    """Local extractive summary over message texts.

    The method ranks messages by keyword density and keeps the selected messages in
    chronological order so the summary remains grounded in real conversation turns.
    """
    if not messages:
        return ""
    if len(messages) <= max_sentences:
        return " ".join(f"{m['speaker']}: {m['text']}" for m in messages)[:max_chars]

    counts: Counter[str] = Counter()
    for message in messages:
        counts.update(tokenize(message["text"]))

    scored = [
        (_sentence_score(message["text"], counts, idx, len(messages)), idx, message)
        for idx, message in enumerate(messages)
    ]
    chosen = sorted(scored, reverse=True)[:max_sentences]
    chosen_by_time = [item[2] for item in sorted(chosen, key=lambda item: item[1])]
    summary = " ".join(f"{m['speaker']}: {m['text']}" for m in chosen_by_time)
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3].rstrip() + "..."
    return summary


def describe_segment(messages: list[dict]) -> dict:
    texts = [message["text"] for message in messages]
    return {
        "summary": summarize_messages(messages),
        "keywords": top_keywords(texts),
        "speakers": sorted({message["speaker"] for message in messages}),
        "evidence_sample": [message["id"] for message in messages[:2] + messages[-2:]][:4],
    }
