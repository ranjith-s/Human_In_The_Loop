"""RecordDecisionNode and EscalationNode for Lab 3 HITL moderation pipeline.

These nodes are the audit layer for moderation governance.
- record_escalation writes escalated reviews to escalations.json
- record_decision writes all finalized reviews to moderation_log.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODERATION_LOG_PATH = Path("moderation_log.json")
ESCALATION_LOG_PATH = Path("escalations.json")


def _now_iso() -> str:
    """Return UTC timestamp in ISO-8601 format for audit consistency."""
    return datetime.now(timezone.utc).isoformat()


def _read_json_list(file_path: Path) -> list[dict[str, Any]]:
    """Read a JSON array from disk; if absent or invalid, return an empty list."""
    if not file_path.exists():
        return []

    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        return []
    except json.JSONDecodeError:
        return []


def _write_json_list(file_path: Path, data: list[dict[str, Any]]) -> None:
    """Write a JSON array with indentation for human-readable audit trails."""
    file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _build_base_audit_entry(state: dict[str, Any]) -> dict[str, Any]:
    """Build common fields used by both escalation and moderation logs."""
    return {
        "timestamp": _now_iso(),
        "review_id": state.get("review_id", "unknown_review"),
        "review_text": state.get("review_text", ""),
        "ai_recommendation": state.get("ai_recommendation", "FLAG"),
        "ai_reasoning": state.get("ai_reasoning", ""),
        "violated_rules": state.get("violated_rules", []),
        "confidence": state.get("confidence", 0.0),
        "human_decision": state.get("human_decision", ""),
        "final_action": state.get("final_action", "FLAG"),
        "moderator_note": state.get("moderator_note", ""),
    }


def record_escalation(state: dict[str, Any]) -> dict[str, Any]:
    """Write an escalation-specific entry when moderator selects ESCALATE."""
    entry = _build_base_audit_entry(state)
    entry["escalated"] = True

    escalations = _read_json_list(ESCALATION_LOG_PATH)
    escalations.append(entry)
    _write_json_list(ESCALATION_LOG_PATH, escalations)

    return {"escalation_logged": True}


def record_decision(state: dict[str, Any]) -> dict[str, Any]:
    """Write the final moderation decision to moderation_log.json.

    This node should run only after validated human input is available.
    """
    entry = _build_base_audit_entry(state)
    entry["escalated"] = bool(state.get("escalation_required", False))

    moderation_log = _read_json_list(MODERATION_LOG_PATH)
    moderation_log.append(entry)
    _write_json_list(MODERATION_LOG_PATH, moderation_log)

    return {"decision_logged": True}
