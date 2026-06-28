"""HumanModerationNode for Lab 3 HITL moderation pipeline.

This node validates moderator input and normalizes it into fields used by routing
and logging nodes. Invalid input triggers a re-prompt in the graph.
"""

from __future__ import annotations

from typing import Any

ALLOWED_ACTIONS = {"APPROVE", "FLAG", "REMOVE"}


def _parse_human_decision(raw_input: str, ai_recommendation: str) -> dict[str, Any]:
    """Parse and validate moderator commands.

    Supported commands:
    - APPROVE
    - OVERRIDE <APPROVE|FLAG|REMOVE> <reason>
    - ESCALATE <senior_moderator_note>
    """
    text = (raw_input or "").strip()
    if not text:
        return {
            "validation_error": "Empty input. Please enter APPROVE, OVERRIDE, or ESCALATE.",
        }

    tokens = text.split()
    command = tokens[0].upper()

    if command == "APPROVE":
        # APPROVE means accept the AI recommendation as final.
        final_action = ai_recommendation if ai_recommendation in ALLOWED_ACTIONS else "FLAG"
        return {
            "parsed_human_action": "APPROVE",
            "final_action": final_action,
            "moderator_note": "Moderator approved AI recommendation.",
            "escalation_required": False,
            "validation_error": "",
        }

    if command == "OVERRIDE":
        if len(tokens) < 3:
            return {
                "validation_error": (
                    "OVERRIDE requires both a new action and a reason. "
                    "Format: OVERRIDE <APPROVE|FLAG|REMOVE> <reason>"
                )
            }

        requested_action = tokens[1].upper().strip()
        reason = " ".join(tokens[2:]).strip()

        if requested_action not in ALLOWED_ACTIONS:
            return {
                "validation_error": (
                    "Invalid OVERRIDE action. Use APPROVE, FLAG, or REMOVE."
                )
            }

        if not reason:
            return {
                "validation_error": "OVERRIDE reason cannot be empty.",
            }

        return {
            "parsed_human_action": "OVERRIDE",
            "final_action": requested_action,
            "moderator_note": f"Moderator override reason: {reason}",
            "escalation_required": False,
            "validation_error": "",
        }

    if command == "ESCALATE":
        note = " ".join(tokens[1:]).strip()
        if not note:
            return {
                "validation_error": "ESCALATE requires a senior moderator note.",
            }

        return {
            "parsed_human_action": "ESCALATE",
            "final_action": "ESCALATE",
            "moderator_note": note,
            "escalation_required": True,
            "validation_error": "",
        }

    return {
        "validation_error": (
            "Unknown command. Use APPROVE, OVERRIDE <action> <reason>, "
            "or ESCALATE <note>."
        )
    }


def human_moderation(state: dict[str, Any]) -> dict[str, Any]:
    """HumanModerationNode entry point.

    It parses moderator input and returns graph state updates.
    """
    human_decision = str(state.get("human_decision", ""))
    ai_recommendation = str(state.get("ai_recommendation", "FLAG")).upper()
    return _parse_human_decision(human_decision, ai_recommendation)
