# Lab 3: Human-in-the-Loop Product Review Moderation

This project implements a LangGraph HITL moderation pipeline for e-commerce reviews.
It follows the lab requirements:

- Uses `StateGraph` with a typed state schema.
- Uses `interrupt()` in `PresentForModerationNode`.
- Uses `MemorySaver` to persist state through pause/resume.
- Enforces moderator commands:
  - `APPROVE`
  - `OVERRIDE <APPROVE|FLAG|REMOVE> <reason>`
  - `ESCALATE <senior_note>`
- Logs final decisions to `moderation_log.json`.
- Logs escalations to `escalations.json` via a dedicated escalation node.

## Project Structure

- `graph.py`: Full LangGraph orchestration with pause/resume and routing.
- `nodes/analyse.py`: Retrieval + AI policy analysis.
- `nodes/present.py`: Human checkpoint with `interrupt()`.
- `nodes/moderate.py`: Human command validation and normalization.
- `nodes/record.py`: Decision and escalation audit logging.
- `ingest.py`: Policy document ingestion to ChromaDB collection `moderation_policies`.
- `.env`: Ollama + Chroma configuration.
- `moderation_log.json`: Audit log of moderation decisions.
- `escalations.json`: Dedicated escalation log.
- `data/ecommerce_policies/`: Local sample policy documents.

## Run Steps

1. Activate your environment (already created by you):

```bash
conda activate agentic
```

2. Go to the project folder:

```bash
cd /home/chaitanya-kohli/AgenticLab_Kohli/lab3_hitl
```

3. Ingest policy documents (one-time setup):

```bash
python ingest.py --data-dir data/ecommerce_policies --chroma-dir chroma_db --reset
```

4. Run a moderation session:

```bash
python graph.py --review_id REV-2001 --review_text "This is the worst product, seller is a scammer"
```

5. When prompted at the checkpoint, enter one of:

- `APPROVE`
- `OVERRIDE FLAG The review is harsh but not policy violating`
- `ESCALATE Needs senior fraud analyst review`

## How Pause/Resume Works

- `present_for_moderation()` calls `interrupt(panel_text)`.
- Graph execution pauses and returns control to caller.
- Caller resumes with:

```python
graph.invoke(Command(resume=human_input), config=thread_config)
```

- State is preserved by `MemorySaver`, so analysis is not recomputed on resume.

## Notes

- The collection name is fixed to `moderation_policies` as required.
- The Chroma persist path defaults to `./chroma_db`.
- No cloud API keys are required; local Ollama is used.
