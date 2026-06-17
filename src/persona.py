from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

from .utils import DATA_DIR, read_jsonl, tokenize, write_json


EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)

CATEGORIES = {
    "sleep": ["sleep", "slept", "tired", "woke", "morning", "midnight", "late night", "nap"],
    "food": ["cook", "food", "eat", "dinner", "lunch", "breakfast", "restaurant", "bake", "meal"],
    "exercise": ["run", "running", "yoga", "gym", "hike", "hiking", "workout", "walk", "exercise"],
    "routine": ["every day", "usually", "always", "weekend", "after work", "before work", "routine"],
}

FACT_PATTERNS = {
    "occupation": [
        re.compile(r"\bI(?:'m| am) (?:a|an) ([A-Za-z][A-Za-z\s-]{2,40})", re.I),
        re.compile(r"\bI work as (?:a|an)?\s*([A-Za-z][A-Za-z\s-]{2,40})", re.I),
    ],
    "location": [
        re.compile(r"\bI live in ([A-Za-z][A-Za-z\s,.-]{2,50})", re.I),
        re.compile(r"\bI(?:'m| am) moving to ([A-Za-z][A-Za-z\s,.-]{2,50})", re.I),
    ],
    "education": [
        re.compile(r"\bstudying ([A-Za-z][A-Za-z\s-]{2,50})", re.I),
        re.compile(r"\bI(?:'m| am) a student\b", re.I),
    ],
}

TRAIT_SIGNALS = {
    "curious": ["?", "what", "why", "how", "tell me"],
    "enthusiastic": ["!", "awesome", "amazing", "love", "excited"],
    "empathetic": ["sorry", "hope", "glad", "that sounds", "thanks for"],
    "social": ["family", "friends", "band", "chatting", "talking"],
}


def evidence_item(message: dict) -> dict:
    return {"message_id": message["id"], "day": message["day"], "text": message["text"]}


def _append_signal(target: list[dict], label: str, message: dict, max_items: int = 12) -> None:
    if len(target) >= max_items:
        return
    target.append({"signal": label, **evidence_item(message)})


def contains_keyword(text: str, keyword: str) -> bool:
    if " " in keyword:
        return keyword in text
    return re.search(rf"\b{re.escape(keyword)}\b", text) is not None


def extract_for_speaker(messages: list[dict], speaker: str) -> dict:
    speaker_messages = [message for message in messages if message["speaker"] == speaker]
    word_counts = [len(message["text"].split()) for message in speaker_messages]
    lower_texts = [message["text"].lower() for message in speaker_messages]
    all_text = " ".join(lower_texts)

    persona = {
        "speaker": speaker,
        "message_count": len(speaker_messages),
        "habits": {category: [] for category in CATEGORIES},
        "personal_facts": {"occupation": [], "location": [], "family": [], "education": [], "life_events": []},
        "personality_traits": [],
        "communication_style": {
            "avg_words_per_message": round(sum(word_counts) / max(len(word_counts), 1), 2),
            "uses_emojis": any(EMOJI_RE.search(message["text"]) for message in speaker_messages),
            "emoji_message_count": sum(1 for message in speaker_messages if EMOJI_RE.search(message["text"])),
            "question_ratio": round(sum("?" in message["text"] for message in speaker_messages) / max(len(speaker_messages), 1), 3),
            "exclamation_ratio": round(sum("!" in message["text"] for message in speaker_messages) / max(len(speaker_messages), 1), 3),
            "common_openings": [],
            "tone_markers": [],
        },
    }

    openings = Counter()
    tone_markers = Counter()
    trait_hits: dict[str, list[dict]] = defaultdict(list)

    for message in speaker_messages:
        low = message["text"].lower()
        words = message["text"].split()
        if words:
            openings[words[0].strip(".,!?").lower()] += 1
        for category, keywords in CATEGORIES.items():
            if any(contains_keyword(low, keyword) for keyword in keywords):
                _append_signal(persona["habits"][category], category, message)
        for fact_name, patterns in FACT_PATTERNS.items():
            for pattern in patterns:
                match = pattern.search(message["text"])
                if match:
                    value = match.group(1).strip(" .,!") if match.groups() else message["text"]
                    persona["personal_facts"][fact_name].append({"value": value, **evidence_item(message)})
                    break
        if any(word in low for word in ["mom", "dad", "parent", "parents", "sister", "brother", "family", "wife", "husband"]):
            _append_signal(persona["personal_facts"]["family"], "family mention", message)
        if any(word in low for word in ["moving", "new job", "graduated", "passed away", "started", "birthday"]):
            _append_signal(persona["personal_facts"]["life_events"], "life event", message)
        for trait, signals in TRAIT_SIGNALS.items():
            if any(signal in low for signal in signals):
                if len(trait_hits[trait]) < 5:
                    trait_hits[trait].append(evidence_item(message))
        for marker in ["thanks", "love", "awesome", "amazing", "sorry", "great", "cool"]:
            if marker in low:
                tone_markers[marker] += 1

    persona["communication_style"]["common_openings"] = openings.most_common(10)
    persona["communication_style"]["tone_markers"] = tone_markers.most_common(10)
    for trait, evidence in sorted(trait_hits.items()):
        if len(evidence) >= 3:
            persona["personality_traits"].append({"trait": trait, "evidence": evidence})

    persona["top_keywords"] = Counter(tokenize(all_text)).most_common(20)
    return persona


def build_personas() -> dict:
    messages = list(read_jsonl(DATA_DIR / "all_messages.jsonl"))
    user1 = extract_for_speaker(messages, "User 1")
    user2 = extract_for_speaker(messages, "User 2")
    combined = {"User 1": user1, "User 2": user2}
    write_json(DATA_DIR / "persona_user1.json", user1)
    write_json(DATA_DIR / "persona_user2.json", user2)
    write_json(DATA_DIR / "persona.json", combined)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract evidence-backed personas for both users.")
    parser.parse_args()
    print(json.dumps(build_personas(), indent=2, ensure_ascii=False)[:4000])
    print("\n... saved full persona JSON files in data/")


if __name__ == "__main__":
    main()
