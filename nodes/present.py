"""PresentForModerationNode for Lab 3 HITL moderation pipeline.

This node is the explicit human-in-the-loop checkpoint.
It renders the AI recommendation, then pauses graph execution using interrupt().
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt


def _format_policy_summary(retrieved_policies: list[dict[str, Any]]) -> str:
    """Build a compact policy summary for the moderator console view."""
    if not retrieved_policies:
        return "No policies retrieved for this review."

    lines: list[str] = []
    for policy in retrieved_policies[:5]:
        lines.append(
            f"- {policy.get('policy_id', 'unknown')}"
            f" ({policy.get('category', 'general')}, {policy.get('severity', 'medium')})"
        )
    return "\n".join(lines)


def _render_panel(state: dict[str, Any]) -> str:
    """Render a human-friendly moderation panel that includes input instructions."""
    review_id = state.get("review_id", "unknown_review")
    review_text = state.get("review_text", "")
    ai_recommendation = state.get("ai_recommendation", "FLAG")
    confidence = state.get("confidence", 0.0)
    violated_rules = state.get("violated_rules", [])
    ai_reasoning = state.get("ai_reasoning", "")
    validation_error = state.get("validation_error", "")

    policy_summary = _format_policy_summary(state.get("retrieved_policies", []))
    violated = ", ".join(violated_rules) if violated_rules else "None"

    error_block = ""
    if validation_error:
        # When moderator input is invalid, we show exactly what was wrong and re-prompt.
        error_block = f"\nINPUT ERROR: {validation_error}\n"

    return f"""
================ HUMAN MODERATION CHECKPOINT ================
Review ID: {review_id}

Review Text:
{review_text}

AI Recommendation: {ai_recommendation}
AI Confidence: {confidence}
Violated Rules: {violated}

AI Reasoning:
{ai_reasoning}

Retrieved Policy Context:
{policy_summary}
{error_block}
Enter one of the following commands:
1) APPROVE
2) OVERRIDE <APPROVE|FLAG|REMOVE> <reason>
3) ESCALATE <senior_moderator_note>
==============================================================
""".strip()


def present_for_moderation(state: dict[str, Any]) -> dict[str, Any]:
    """Pause the graph and collect a human moderator decision via interrupt()."""
    panel_text = _render_panel(state)

    # interrupt() freezes execution and returns control to the caller.
    # The graph resumes when the caller invokes Command(resume=<human_input>).
    human_input = interrupt(panel_text)

    return {
        "human_decision": str(human_input).strip(),
        # Clear previous validation error after a new input arrives.
        "validation_error": "",
    }
