# modelpack/orchestration/graph.py
from typing import TypedDict

import structlog
from langgraph.graph import END, StateGraph

from ..agents import (
    agent1_extractor,
    agent3_mathifier,
    agent4_pyomo,
    agent5_datagen,
    agent6_screen,
    agent7_checker,
    agent8_solver,
    agent9_judge,
)
from ..llm import llm_client
from ..schemas import ModelPack

logger = structlog.get_logger(__name__)

# Benchmark traces in temp/ only exercise the main full graph.
MAIN_FULL_GRAPH_VARIANT = "main"
GRAPH_VARIANT_ALIASES = {
    "main": MAIN_FULL_GRAPH_VARIANT,
    "no_a2_merge_a0_a1": MAIN_FULL_GRAPH_VARIANT,
    "no_a2_merge_a0_a1_no_a3b": MAIN_FULL_GRAPH_VARIANT,
}


class GraphState(TypedDict):
    model_pack: ModelPack


def _feedback_summary(feedback: object) -> dict[str, object] | None:
    if feedback is None:
        return None
    model_dump = getattr(feedback, "model_dump", None)
    if callable(model_dump):
        try:
            payload = model_dump(mode="json")
        except TypeError:
            payload = model_dump()
        if isinstance(payload, dict):
            return payload
    if isinstance(feedback, dict):
        return dict(feedback)
    return {"value": str(feedback)}


def _append_trajectory_event(model_pack: ModelPack, **event: object) -> None:
    trajectory = model_pack.tests.setdefault("trajectory", [])
    if not isinstance(trajectory, list):
        trajectory = []
        model_pack.tests["trajectory"] = trajectory
    entry = {"sequence": len(trajectory) + 1}
    entry.update(event)
    trajectory.append(entry)


async def _run_agent(
    state: GraphState,
    *,
    label: str,
    handler,
) -> GraphState:
    before = llm_client.trace_length()
    model_pack = await handler(state["model_pack"])
    after = llm_client.trace_length()
    llm_sequences = list(range(before + 1, after + 1))
    _append_trajectory_event(
        model_pack,
        type="agent",
        agent=label,
        llm_call_sequences=llm_sequences,
        status=model_pack.status,
    )
    state["model_pack"] = model_pack
    return state


async def run_a0_a1(state: GraphState) -> GraphState:
    return await _run_agent(state, label="A0A1_specify_extract", handler=agent1_extractor.a0_a1_specify_extract)


async def run_a3(state: GraphState) -> GraphState:
    return await _run_agent(state, label="A3_mathifier", handler=agent3_mathifier.a3_mathifier)


async def run_a4(state: GraphState) -> GraphState:
    return await _run_agent(state, label="A4_pyomo", handler=agent4_pyomo.a4_pyomo)


async def run_a5(state: GraphState) -> GraphState:
    return await _run_agent(state, label="A5_datagen", handler=agent5_datagen.a5_datagen)


async def run_a6(state: GraphState) -> GraphState:
    return await _run_agent(state, label="A6_screen", handler=agent6_screen.a6_screen)


async def run_a7(state: GraphState) -> GraphState:
    return await _run_agent(state, label="A7_checker", handler=agent7_checker.a7_checker)


async def run_a8(state: GraphState) -> GraphState:
    return await _run_agent(state, label="A8_solver", handler=agent8_solver.a8_solver)


async def run_a9(state: GraphState) -> GraphState:
    return await _run_agent(state, label="A9_judge", handler=agent9_judge.a9_judge)


def route_after_a6(state: GraphState) -> str:
    """Route after feasibility screen."""
    feedback = state["model_pack"].tests.get("last_feedback")

    if feedback and feedback.target_agent == "A4":
        logger.info("routing_to_a4", issue=feedback.issue)
        _append_trajectory_event(
            state["model_pack"],
            type="route",
            from_agent="A6_screen",
            to_agent="A4_pyomo",
            reason=_feedback_summary(feedback),
        )
        return "A4_pyomo"

    _append_trajectory_event(
        state["model_pack"],
        type="route",
        from_agent="A6_screen",
        to_agent="A7_checker",
        reason=_feedback_summary(feedback) or {"reason": "screen_passed"},
    )
    return "A7_checker"


def route_after_a9(state: GraphState) -> str:
    """Route after cross-validation."""
    feedback = state["model_pack"].tests.get("last_feedback")

    if feedback and feedback.issue == "checker_false_negative":
        _append_trajectory_event(
            state["model_pack"],
            type="route",
            from_agent="A9_judge",
            to_agent="A7_checker",
            reason=_feedback_summary(feedback),
        )
        return "A7_checker"

    _append_trajectory_event(
        state["model_pack"],
        type="route",
        from_agent="A9_judge",
        to_agent="END",
        reason=_feedback_summary(feedback) or {"reason": "judge_completed"},
    )
    return "END"


def _normalize_graph_variant(graph_variant: str) -> str:
    normalized = (graph_variant or MAIN_FULL_GRAPH_VARIANT).strip().lower()
    try:
        return GRAPH_VARIANT_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported full graph variant: {graph_variant}") from exc


def create_graph(graph_variant: str = MAIN_FULL_GRAPH_VARIANT) -> StateGraph:
    """Create the benchmark-used full graph with feedback loops."""

    _normalize_graph_variant(graph_variant)

    graph = StateGraph(GraphState)

    graph.add_node("A0A1_specify_extract", run_a0_a1)
    graph.add_node("A3_mathifier", run_a3)
    graph.add_node("A4_pyomo", run_a4)
    graph.add_node("A5_datagen", run_a5)
    graph.add_node("A6_screen", run_a6)
    graph.add_node("A7_checker", run_a7)
    graph.add_node("A8_solver", run_a8)
    graph.add_node("A9_judge", run_a9)

    graph.add_edge("A0A1_specify_extract", "A3_mathifier")
    graph.add_edge("A3_mathifier", "A4_pyomo")
    graph.add_edge("A4_pyomo", "A5_datagen")
    graph.add_edge("A5_datagen", "A6_screen")
    graph.add_edge("A7_checker", "A8_solver")
    graph.add_edge("A8_solver", "A9_judge")

    graph.add_conditional_edges(
        "A6_screen",
        route_after_a6,
        {
            "A4_pyomo": "A4_pyomo",
            "A7_checker": "A7_checker",
        },
    )
    graph.add_conditional_edges(
        "A9_judge",
        route_after_a9,
        {
            "A7_checker": "A7_checker",
            "END": END,
        },
    )

    graph.set_entry_point("A0A1_specify_extract")
    return graph


def create_generation_graph() -> StateGraph:
    """Create the minimal MAS path that stops after Pyomo generation."""

    graph = StateGraph(GraphState)

    graph.add_node("A0A1_specify_extract", run_a0_a1)
    graph.add_node("A3_mathifier", run_a3)
    graph.add_node("A4_pyomo", run_a4)

    graph.add_edge("A0A1_specify_extract", "A3_mathifier")
    graph.add_edge("A3_mathifier", "A4_pyomo")
    graph.add_edge("A4_pyomo", END)

    graph.set_entry_point("A0A1_specify_extract")
    return graph


def create_app(graph_variant: str = MAIN_FULL_GRAPH_VARIANT):
    """Create compiled app without checkpointing."""
    graph = create_graph(graph_variant=graph_variant)
    return graph.compile()


def create_generation_app():
    """Create compiled generation-only app without checkpointing."""
    graph = create_generation_graph()
    return graph.compile()
