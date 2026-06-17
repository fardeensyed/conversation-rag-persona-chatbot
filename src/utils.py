from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "indexes"
REPORT_DIR = PROJECT_ROOT / "reports"

STOPWORDS = {
    "a", "about", "after", "all", "also", "am", "an", "and", "any", "are", "as",
    "at", "be", "because", "been", "but", "by", "can", "could", "did", "do",
    "does", "doing", "for", "from", "get", "go", "going", "good", "got", "had",
    "has", "have", "he", "her", "here", "hey", "hi", "him", "his", "how", "i",
    "if", "im", "in", "is", "it", "its", "just", "know", "like", "me", "my",
    "no", "not", "of", "oh", "on", "or", "our", "out", "really", "she", "so",
    "some", "that", "the", "their", "them", "then", "there", "they", "this",
    "to", "too", "up", "very", "was", "we", "well", "were", "what", "when",
    "where", "who", "why", "will", "with", "would", "yeah", "yes", "you",
    "your", "youre",
}

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z']+")


def ensure_dirs() -> None:
    for path in (DATA_DIR, INDEX_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def tokenize(text: str) -> list[str]:
    return [
        token.replace("'", "").lower()
        for token in TOKEN_RE.findall(text)
        if token.lower().replace("'", "") not in STOPWORDS
    ]


def top_keywords(texts: Iterable[str], limit: int = 8) -> list[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    return [word for word, _ in counts.most_common(limit)]


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def message_label(message: dict) -> str:
    return f"{message['speaker']} #{message['id']} (day {message['day']}): {message['text']}"
