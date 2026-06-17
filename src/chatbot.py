from __future__ import annotations

import argparse

from .rag import format_rag_answer, retrieve
from .utils import DATA_DIR, read_json


PERSONA_TRIGGERS = [
    "person", "habit", "personality", "talk", "communicate", "communication", "style",
    "kind of", "type of", "trait", "facts", "user 1", "user 2",
]

GENERIC_TOPIC_TERMS = {
    "love", "great", "thats", "fun", "sounds", "favorite", "thanks", "ive", "sure",
    "lot", "really", "good", "nice", "awesome", "amazing", "cool", "well", "like",
    "enjoy", "thing", "things", "today", "day", "time", "always", "want", "wanted",
    "glad", "theyre", "whats", "youre", "pretty", "hear", "hope", "tell",
}


def is_persona_query(query: str) -> bool:
    low = query.lower()
    return any(trigger in low for trigger in PERSONA_TRIGGERS)


def is_topic_overview_query(query: str) -> bool:
    low = query.lower()
    return "topic" in low and any(word in low for word in ["most", "common", "discuss", "talk"])


def _target_users(query: str) -> list[str]:
    low = query.lower()
    if "user 1" in low:
        return ["User 1"]
    if "user 2" in low:
        return ["User 2"]
    return ["User 1", "User 2"]


def persona_answer(query: str) -> str:
    persona = read_json(DATA_DIR / "persona.json")
    lines = [f"Persona answer for: {query}", ""]
    low = query.lower()
    for user in _target_users(query):
        data = persona[user]
        lines.append(f"{user}:")
        style = data["communication_style"]
        if "habit" in low:
            for category, signals in data["habits"].items():
                if signals:
                    sample = "; ".join(f"#{item['message_id']}: {item['text']}" for item in signals[:3])
                    lines.append(f"- {category}: {sample}")
        elif "talk" in low or "communicat" in low or "style" in low:
            lines.append(
                f"- Average words/message: {style['avg_words_per_message']}; "
                f"question ratio: {style['question_ratio']}; exclamation ratio: {style['exclamation_ratio']}; "
                f"uses emojis: {style['uses_emojis']}."
            )
            lines.append(f"- Common openings: {style['common_openings'][:5]}")
            lines.append(f"- Tone markers: {style['tone_markers'][:5]}")
        elif "fact" in low or "personal" in low:
            for category, facts in data["personal_facts"].items():
                if facts:
                    sample = "; ".join(f"#{item['message_id']}: {item.get('value', item['text'])}" for item in facts[:4])
                    lines.append(f"- {category}: {sample}")
        else:
            traits = ", ".join(item["trait"] for item in data["personality_traits"]) or "not enough strong trait evidence"
            top_keywords = ", ".join(word for word, _ in data.get("top_keywords", [])[:8])
            lines.append(f"- Message count: {data['message_count']}")
            lines.append(f"- Evidence-backed traits: {traits}")
            lines.append(f"- Frequent content words: {top_keywords}")
            lines.append(
                f"- Communication: {style['avg_words_per_message']} words/message, "
                f"{style['question_ratio']} question ratio, {style['exclamation_ratio']} exclamation ratio."
            )
        lines.append("")
    return "\n".join(lines).strip()


def topic_overview_answer(query: str) -> str:
    from collections import Counter

    topics = read_json(DATA_DIR / "topic_checkpoints.json")
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict]] = {}
    for topic in topics:
        for keyword in topic.get("keywords", [])[:5]:
            if keyword in GENERIC_TOPIC_TERMS or len(keyword) < 4:
                continue
            counts[keyword] += 1
            examples.setdefault(keyword, [])
            if len(examples[keyword]) < 2:
                examples[keyword].append(topic)

    lines = [f"Topic overview for: {query}", ""]
    for keyword, count in counts.most_common(10):
        sample = examples[keyword][0]
        lines.append(
            f"- {keyword}: appears in {count} topic checkpoints; "
            f"example topic {sample['topic_id']} messages {sample['start_id']}-{sample['end_id']}: {sample['summary']}"
        )
    return "\n".join(lines)


def chat(query: str) -> str:
    if not query.strip():
        return "Ask a question about the conversations or either user's persona."
    if is_topic_overview_query(query):
        return f"{topic_overview_answer(query)}\n\nSupporting RAG evidence:\n{format_rag_answer(query, retrieve(query, top_k=2))}"
    if is_persona_query(query):
        persona_context = persona_answer(query)
        rag_context = format_rag_answer(query, retrieve(query, top_k=2))
        return f"{persona_context}\n\nSupporting RAG evidence:\n{rag_context}"
    return format_rag_answer(query)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the local chatbot a question.")
    parser.add_argument("query")
    args = parser.parse_args()
    print(chat(args.query))


if __name__ == "__main__":
    main()
