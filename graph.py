"""Lab 3 HITL moderation graph.

Graph flow:
START -> analyse -> present (interrupt) -> moderate
  -> if invalid input: present (re-prompt)
  -> if ESCALATE: escalate -> record
  -> else: record
record -> END

This design guarantees that no moderation decision is written before human action.
"""

from __future__ import annotations

from typing import Any
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from nodes import (
    analyse_review,
    human_moderation,
    present_for_moderation,
    record_decision,
    record_escalation,
)


class ReviewModerationState(TypedDict, total=False):
    """State schema required by the lab handbook and routing logic."""

    review_text: str
    review_id: str
    retrieved_policies: list[dict[str, Any]]
    ai_recommendation: str
    ai_reasoning: str
    violated_rules: list[str]
    confidence: float

    human_decision: str
    final_action: str
    moderator_note: str

    escalation_required: bool
    escalation_logged: bool
    decision_logged: bool
    validation_error: str


def _route_after_moderation(state: ReviewModerationState) -> str:
    """Conditional router after human input parsing."""
    if state.get("validation_error"):
        # Invalid moderator input loops back to PresentForModerationNode.
        return "present"

    if state.get("escalation_required", False):
        return "escalate"

    return "record"


def build_graph():
    """Compile and return the HITL graph with MemorySaver checkpointing."""
    graph_builder = StateGraph(ReviewModerationState)

    graph_builder.add_node("analyse", analyse_review)
    graph_builder.add_node("present", present_for_moderation)
    graph_builder.add_node("moderate", human_moderation)
    graph_builder.add_node("escalate", record_escalation)
    graph_builder.add_node("record", record_decision)

    graph_builder.add_edge(START, "analyse")
    graph_builder.add_edge("analyse", "present")
    graph_builder.add_edge("present", "moderate")

    graph_builder.add_conditional_edges(
        "moderate",
        _route_after_moderation,
        {
            "present": "present",
            "escalate": "escalate",
            "record": "record",
        },
    )

    graph_builder.add_edge("escalate", "record")
    graph_builder.add_edge("record", END)

    # MemorySaver ensures state survives interrupt() pause/resume cycles.
    return graph_builder.compile(checkpointer=MemorySaver())


def run_hitl_session(review_id: str, review_text: str, thread_id: str = "lab3-session") -> dict[str, Any]:
    """Run a complete interactive HITL moderation session from terminal.

    The function invokes the graph once, then keeps resuming from interrupts until
    a final state is produced.
    """
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    state: ReviewModerationState = {
        "review_id": review_id,
        "review_text": review_text,
    }

    result = graph.invoke(state, config=config)

    while "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"][0].value
        print("\n" + str(interrupt_payload) + "\n")

        moderator_input = input("Moderator decision: ").strip()
        result = graph.invoke(Command(resume=moderator_input), config=config)

    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run Lab 3 HITL moderation flow.")
    parser.add_argument("--review_id", type=str, help="Unique identifier for the review")
    parser.add_argument("--review_text", type=str, help="The review text to moderate")
    parser.add_argument(
        "--thread-id",
        type=str,
        default="lab3-session",
        help="Thread ID for LangGraph checkpoint state",
    )

    args = parser.parse_args()
    final_state = run_hitl_session(
        review_id=args.review_id,
        review_text=args.review_text,
        thread_id=args.thread_id,
    )

    print("Final state:")
    print(json.dumps(final_state, indent=2, default=str))
