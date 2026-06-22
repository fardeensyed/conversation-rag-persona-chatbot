# Conversation RAG + Persona Chatbot

This project processes a CSV of conversations in chronological order, creates topic and 100-message checkpoints, extracts evidence-backed personas for both users, and exposes a simple chatbot.

## Quick Start

```powershell
python -m src.parse --input C:\Users\syedf\Downloads\conversations.csv
python -m src.checkpoints
python -m src.index_store
python -m src.persona
python -m src.chatbot "What kind of person is User 1?"
streamlit run app.py
```

The system is local-first. It uses a built-in TF-IDF implementation, so it can run even before optional ML packages are installed.
## Recent Updates

### New Features

* Added conversation parsing pipeline for chronological message processing.
* Implemented automatic topic change detection using TF-IDF similarity and semantic drift analysis.
* Added multi-level retrieval system using:

  * Topic checkpoint index
  * 100-message checkpoint index
  * Raw conversation chunk index
* Implemented evidence-backed persona extraction for both participants.
* Added communication style analysis including:

  * Average message length
  * Question frequency
  * Emoji usage
  * Tone indicators
* Added Streamlit web interface for interactive querying.
* Added threshold tuning utility for optimizing topic segmentation.
* Improved RAG pipeline by combining summaries with raw message evidence.
* Added separate persona generation for User 1 and User 2 with supporting message references.

### Technical Highlights

* Local-first architecture with no external API dependency.
* Custom TF-IDF retrieval implementation.
* Evidence-grounded persona generation.
* Multi-index retrieval strategy for improved answer quality.
* Lightweight deployment suitable for personal systems.


Install the base demo dependencies with:

```powershell
pip install -r requirements.txt
```

Optional heavier retrieval/model packages are listed separately:

```powershell
pip install -r requirements-optional.txt
```

## Outputs

- `data/all_messages.jsonl`
- `data/topic_checkpoints.json`
- `data/msg100_checkpoints.json`
- `data/persona_user1.json`
- `data/persona_user2.json`
- `data/persona.json`
- `indexes/topic_index.json`
- `indexes/msg100_index.json`
- `indexes/chunk_index.json`
- `reports/RAG_Project_Documentation.docx`

## How Topic Changes Are Detected

The system processes messages in chronological order: CSV row order first, then message order inside each row. Each parsed message receives a global `id`, `day`, `turn_in_day`, `speaker`, and `text`.

Topic detection is local and does not require an external API:

1. The checkpoint builder keeps a rolling topic segment.
2. Every 5 messages, after the current segment has enough history, it compares the current message with the previous 15 messages in that segment.
3. Both the rolling window and current message are converted into TF-IDF-style vectors using local tokenization and stopword removal.
4. If cosine similarity drops below the configured threshold, the system treats that as semantic drift and closes the previous segment as a topic checkpoint.
5. Each topic checkpoint stores its message range, speakers, keywords, summary, and evidence message IDs.

The default settings are:

- `WINDOW = 15`
- `CHECK_EVERY = 5`
- `THRESHOLD = 0.15`
- `MIN_TOPIC_MESSAGES = 20`

The project also writes `data/threshold_tuning.json`, which counts detected topic changes on the first 500 messages for several thresholds.

## How Retrieval Works

Retrieval combines summaries and raw message evidence instead of depending on one source.

The indexing step builds three local TF-IDF indexes:

- `topic_index.json`: searches topic checkpoint summaries and keywords.
- `msg100_index.json`: searches independent 100-message checkpoint summaries.
- `chunk_index.json`: searches overlapping raw message chunks.

When a question is asked:

1. The query is tokenized and converted into the same local TF-IDF representation.
2. The system searches the topic, 100-message, and raw chunk indexes.
3. The answer includes the most relevant topic summaries, fixed checkpoint summaries, and raw message previews.
4. Persona-style questions are routed to persona JSON first, then supplemented with RAG evidence.

This means the chatbot can answer from both high-level summaries and lower-level message evidence.

## How Persona Is Built

Persona extraction is evidence-backed and runs separately for `User 1` and `User 2`.

For each speaker, the system reads only that speaker's messages and builds:

- `habits`: sleep, food, exercise, and routine signals from keyword/phrase matches.
- `personal_facts`: occupation, location, education, family mentions, and life events from regex and keyword patterns.
- `personality_traits`: traits such as curious, enthusiastic, empathetic, and social when repeated message evidence exists.
- `communication_style`: quantitative metrics such as average words per message, emoji usage, question ratio, exclamation ratio, common openings, and tone markers.

Every qualitative persona item keeps supporting evidence, including message ID, day, and original text. This keeps the persona grounded in actual conversation signals instead of guesses.

Generated persona files:

- `data/persona_user1.json`
- `data/persona_user2.json`
- `data/persona.json`

