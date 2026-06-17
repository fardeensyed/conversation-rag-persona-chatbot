from __future__ import annotations

import argparse
import json
from pathlib import Path

from .summarize import describe_segment
from .utils import DATA_DIR, ensure_dirs, read_jsonl, write_json
from .vectorize import cosine_for_texts


def detect_topic_change(window_texts: list[str], new_text: str, threshold: float) -> bool:
    if len(window_texts) < 3:
        return False
    return cosine_for_texts(window_texts, new_text) < threshold


def tune_threshold(messages: list[dict], thresholds: list[float] | None = None, limit: int = 500) -> dict:
    thresholds = thresholds or [0.1, 0.15, 0.2, 0.25, 0.3]
    sample = messages[:limit]
    results = {}
    for threshold in thresholds:
        changes = 0
        for idx in range(15, len(sample), 5):
            window = [m["text"] for m in sample[max(0, idx - 15) : idx]]
            if detect_topic_change(window, sample[idx]["text"], threshold):
                changes += 1
        results[str(threshold)] = changes
    return {"sample_messages": len(sample), "changes_by_threshold": results, "chosen_threshold": 0.15}


def build_checkpoints(
    messages_path: Path = DATA_DIR / "all_messages.jsonl",
    topic_output: Path = DATA_DIR / "topic_checkpoints.json",
    fixed_output: Path = DATA_DIR / "msg100_checkpoints.json",
    window: int = 15,
    check_every: int = 5,
    threshold: float = 0.15,
    min_topic_messages: int = 20,
) -> dict:
    ensure_dirs()
    messages = list(read_jsonl(messages_path))
    topic_checkpoints: list[dict] = []
    msg100_checkpoints: list[dict] = []

    tuning = tune_threshold(messages)
    write_json(DATA_DIR / "threshold_tuning.json", tuning)

    current_topic: list[dict] = []
    current_start = 0
    topic_id = 0

    for idx, message in enumerate(messages):
        current_topic.append(message)

        if (idx + 1) % 100 == 0:
            chunk = messages[idx - 99 : idx + 1]
            details = describe_segment(chunk)
            msg100_checkpoints.append(
                {
                    "checkpoint_id": len(msg100_checkpoints),
                    "start_id": chunk[0]["id"],
                    "end_id": chunk[-1]["id"],
                    **details,
                }
            )

        should_check = len(current_topic) > window and idx % check_every == 0
        if should_check and len(current_topic) >= min_topic_messages:
            window_texts = [m["text"] for m in current_topic[-window - 1 : -1]]
            if detect_topic_change(window_texts, message["text"], threshold):
                completed = current_topic[:-1]
                details = describe_segment(completed)
                topic_checkpoints.append(
                    {
                        "topic_id": topic_id,
                        "start_id": completed[0]["id"],
                        "end_id": completed[-1]["id"],
                        **details,
                    }
                )
                topic_id += 1
                current_topic = [message]
                current_start = message["id"]

    if current_topic:
        details = describe_segment(current_topic)
        topic_checkpoints.append(
            {
                "topic_id": topic_id,
                "start_id": current_start,
                "end_id": current_topic[-1]["id"],
                **details,
            }
        )

    remainder_start = len(msg100_checkpoints) * 100
    if remainder_start < len(messages):
        chunk = messages[remainder_start:]
        details = describe_segment(chunk)
        msg100_checkpoints.append(
            {
                "checkpoint_id": len(msg100_checkpoints),
                "start_id": chunk[0]["id"],
                "end_id": chunk[-1]["id"],
                "partial": True,
                **details,
            }
        )

    write_json(topic_output, topic_checkpoints)
    write_json(fixed_output, msg100_checkpoints)
    stats = {
        "messages": len(messages),
        "topic_checkpoints": len(topic_checkpoints),
        "msg100_checkpoints": len(msg100_checkpoints),
        "window": window,
        "check_every": check_every,
        "threshold": threshold,
        "min_topic_messages": min_topic_messages,
    }
    write_json(DATA_DIR / "checkpoint_stats.json", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build topic and 100-message checkpoints.")
    parser.add_argument("--messages", default=str(DATA_DIR / "all_messages.jsonl"))
    parser.add_argument("--threshold", type=float, default=0.15)
    args = parser.parse_args()
    print(json.dumps(build_checkpoints(Path(args.messages), threshold=args.threshold), indent=2))


if __name__ == "__main__":
    main()
