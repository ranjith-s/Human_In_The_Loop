"""AnalyseReviewNode for Lab 3 HITL moderation pipeline.

This module handles three responsibilities:
1) Retrieve relevant moderation policy snippets from ChromaDB.
2) Ask the local Ollama model to analyse a review against those snippets.
3) Return structured fields that downstream HITL nodes can display and moderate.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from langchain_community.llms import Ollama

# Load environment values like OLLAMA_BASE_URL and OLLAMA_MODEL from .env.
load_dotenv()

COLLECTION_NAME = "moderation_policies"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")


def _build_policy_collection(persist_dir: str = DEFAULT_PERSIST_DIR):
    """Return the moderation_policies ChromaDB collection.

    The collection is read-only during analysis; this function only opens it.
    """
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"purpose": "Lab3 moderation policy retrieval"},
    )


def _format_retrieved_policies(chroma_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert raw ChromaDB query output into a clean list of policy dictionaries."""
    documents = chroma_result.get("documents", [[]])[0]
    metadatas = chroma_result.get("metadatas", [[]])[0]
    distances = chroma_result.get("distances", [[]])[0]
    ids = chroma_result.get("ids", [[]])[0]

    formatted: list[dict[str, Any]] = []
    for idx, text in enumerate(documents):
        # Distance can be missing depending on ChromaDB version/config.
        distance = distances[idx] if idx < len(distances) else None
        metadata = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
        doc_id = ids[idx] if idx < len(ids) else f"policy_{idx}"

        formatted.append(
            {
                "id": doc_id,
                "text": text,
                "category": metadata.get("category", "general"),
                "severity": metadata.get("severity", "medium"),
                "policy_id": metadata.get("policy_id", doc_id),
                "distance": distance,
            }
        )

    return formatted


def _build_analysis_prompt(review_text: str, policies: list[dict[str, Any]]) -> str:
    """Create a strict JSON-output prompt for the local moderation model."""
    policy_lines = []
    for policy in policies:
        policy_lines.append(
            f"- PolicyID={policy['policy_id']} | Category={policy['category']}"
            f" | Severity={policy['severity']} | Text={policy['text']}"
        )

    joined_policies = "\n".join(policy_lines) if policy_lines else "- No policies retrieved."

    # The model is instructed to return JSON only so parsing remains reliable.
    return f"""
You are a moderation analyst for an e-commerce platform.
Given the user review and retrieved policy snippets, produce one JSON object only.

Allowed final actions: APPROVE, FLAG, REMOVE

Review:
{review_text}

Retrieved policies:
{joined_policies}

Return STRICT JSON only with this exact schema:
{{
  "action": "APPROVE|FLAG|REMOVE",
  "confidence": 0.0,
  "violated_rules": ["policy_id_or_rule_name"],
  "reasoning": "short but specific explanation"
}}

Rules:
- confidence must be a float between 0 and 1.
- If uncertain, prefer action=FLAG.
- violated_rules should include policy IDs from retrieved policies when applicable.
- Do not output markdown, prose, or code fences.
""".strip()


def _extract_first_json_object(text: str) -> str:
    """Extract the first JSON object from model text.

    Some local models occasionally add preface/postface text despite instructions.
    This helper salvages the first top-level JSON object from the response.
    """
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output.")
    return match.group(0)


def _parse_model_response(response_text: str) -> dict[str, Any]:
    """Parse model response into canonical moderation fields with safe defaults."""
    raw_json = _extract_first_json_object(response_text)
    parsed = json.loads(raw_json)

    action = str(parsed.get("action", "FLAG")).upper().strip()
    if action not in {"APPROVE", "FLAG", "REMOVE"}:
        action = "FLAG"

    confidence = parsed.get("confidence", 0.5)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    violated_rules = parsed.get("violated_rules", [])
    if not isinstance(violated_rules, list):
        violated_rules = []

    reasoning = str(parsed.get("reasoning", "No reasoning provided.")).strip()

    return {
        "action": action,
        "confidence": confidence,
        "violated_rules": violated_rules,
        "reasoning": reasoning,
    }


def analyse_review(state: dict[str, Any]) -> dict[str, Any]:
    """AnalyseReviewNode implementation.

    Input state (required):
    - review_text: str

    Returns state updates:
    - retrieved_policies: list[dict]
    - ai_recommendation: APPROVE|FLAG|REMOVE
    - ai_reasoning: str
    - violated_rules: list[str]
    - confidence: float
    """
    review_text = str(state.get("review_text", "")).strip()
    if not review_text:
        # Fail safely by flagging empty submissions.
        return {
            "retrieved_policies": [],
            "ai_recommendation": "FLAG",
            "ai_reasoning": "Review text is empty; cannot analyse content safely.",
            "violated_rules": ["missing_review_text"],
            "confidence": 1.0,
        }

    try:
        collection = _build_policy_collection()
        query_result = collection.query(query_texts=[review_text], n_results=5)
        retrieved_policies = _format_retrieved_policies(query_result)

        llm = Ollama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=0.0,
        )
        prompt = _build_analysis_prompt(review_text, retrieved_policies)
        llm_response = llm.invoke(prompt)

        parsed = _parse_model_response(str(llm_response))
        return {
            "retrieved_policies": retrieved_policies,
            "ai_recommendation": parsed["action"],
            "ai_reasoning": parsed["reasoning"],
            "violated_rules": parsed["violated_rules"],
            "confidence": parsed["confidence"],
        }

    except Exception as exc:
        # Never crash the pipeline in production moderation flows.
        return {
            "retrieved_policies": [],
            "ai_recommendation": "FLAG",
            "ai_reasoning": f"Analysis fallback due to runtime error: {exc}",
            "violated_rules": ["analysis_runtime_error"],
            "confidence": 0.2,
        }
