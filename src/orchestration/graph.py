# modelpack/orchestration/graph.py
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import END, StateGraph

from ..agents import (
    build_model,
    check_solution,
    derive_math,
    generate_data,
    judge_solution,
    screen_data,
    solve_model,
    specify_problem,
)
from ..llm import llm_client
from ..schemas import ModelPack

MAIN_FULL_GRAPH_VARIANT = "main"
SUPPORTED_GRAPH_VARIANTS = {
    MAIN_FULL_GRAPH_VARIANT,
}

END_NODE = "END"


@dataclass(frozen=True)
class AgentSpec:
    key: str
    node_name: str
    handler: Callable[[ModelPack], Awaitable[ModelPack]]
    feedback_target: str | None = None


AGENT_SPECS = (
    AgentSpec(
        "specify",
        "specify_problem",
        specify_problem.specify_problem,
    ),
    AgentSpec("math", "derive_math", derive_math.derive_math),
    AgentSpec(
        "model",
        "build_model",
        build_model.build_model,
        feedback_target="build_model",
    ),
    AgentSpec("data", "generate_data", generate_data.generate_data),
    AgentSpec("screen", "screen_data", screen_data.screen_data),
    AgentSpec("solve", "solve_model", solve_model.solve_model),
    AgentSpec(
        "check",
        "check_solution",
        check_solution.check_solution,
        feedback_target="check_solution",
    ),
    AgentSpec(
        "judge",
        "judge_solution",
        judge_solution.judge_solution,
    ),
)

AGENTS_BY_KEY = {spec.key: spec for spec in AGENT_SPECS}
AGENTS_BY_NODE = {spec.node_name: spec for spec in AGENT_SPECS}
AGENTS_BY_FEEDBACK_TARGET = {
    spec.feedback_target: spec
    for spec in AGENT_SPECS
    if spec.feedback_target
}
GENERATION_PATH = tuple(spec.node_name for spec in AGENT_SPECS[:3])
FULL_GRAPH_NODES = tuple(spec.node_name for spec in AGENT_SPECS)
PRE_SCREEN_PATH = tuple(spec.node_name for spec in AGENT_SPECS[:5])


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


def _node(key: str) -> str:
    return AGENTS_BY_KEY[key].node_name


def _feedback_target(key: str) -> str | None:
    return AGENTS_BY_KEY[key].feedback_target


def _node_for_feedback_target(target_agent: str | None) -> str | None:
    spec = AGENTS_BY_FEEDBACK_TARGET.get(target_agent)
    return spec.node_name if spec else None


def _make_runner(label: str) -> Callable[[GraphState], Awaitable[GraphState]]:
    handler = AGENTS_BY_NODE[label].handler

    async def run(state: GraphState) -> GraphState:
        return await _run_agent(state, label=label, handler=handler)

    return run


RUNNERS = {label: _make_runner(label) for label in AGENTS_BY_NODE}


def _route(
    state: GraphState,
    *,
    from_node: str,
    to_node: str,
    reason: object,
) -> str:
    _append_trajectory_event(
        state["model_pack"],
        type="route",
        from_agent=from_node,
        to_agent=to_node,
        reason=reason,
    )
    return to_node


def _feedback_route(
    state: GraphState,
    *,
    from_node: str,
    default_to: str,
    default_reason: object,
    allowed_targets: frozenset[str] | None = None,
) -> str:
    feedback = state["model_pack"].tests.get("last_feedback")
    target_agent = getattr(feedback, "target_agent", None) if feedback else None

    if target_agent and (allowed_targets is None or target_agent in allowed_targets):
        target_node = _node_for_feedback_target(target_agent)
        if target_node:
            return _route(
                state,
                from_node=from_node,
                to_node=target_node,
                reason=_feedback_summary(feedback),
            )

    return _route(
        state,
        from_node=from_node,
        to_node=default_to,
        reason=_feedback_summary(feedback) or default_reason,
    )


def route_after_screen(state: GraphState) -> str:
    model_feedback_target = _feedback_target("model")
    return _feedback_route(
        state,
        from_node=_node("screen"),
        default_to=_node("solve"),
        default_reason={"reason": "screen_passed"},
        allowed_targets=frozenset({model_feedback_target}) if model_feedback_target else frozenset(),
    )


def route_after_solve(state: GraphState) -> str:
    solved_instances = [
        instance
        for instance in state["model_pack"].tests.get("instances", [])
        if str(getattr(instance, "id", "")).startswith("solve_")
        and getattr(instance, "feasible", False)
        and getattr(instance, "solution_dict", None)
    ]

    if solved_instances:
        return _route(
            state,
            from_node=_node("solve"),
            to_node=_node("check"),
            reason={"reason": "solved_instances_available", "count": len(solved_instances)},
        )

    return _route(
        state,
        from_node=_node("solve"),
        to_node=END_NODE,
        reason={"reason": "no_solved_instances"},
    )


def route_after_judge(state: GraphState) -> str:
    return _feedback_route(
        state,
        from_node=_node("judge"),
        default_to=END_NODE,
        default_reason={"reason": "judge_completed"},
    )

def _validate_graph_variant(graph_variant: str) -> None:
    normalized = (graph_variant or MAIN_FULL_GRAPH_VARIANT).strip().lower()
    if normalized not in SUPPORTED_GRAPH_VARIANTS:
        raise ValueError(f"Unsupported full graph variant: {graph_variant}")


def _add_nodes(graph: StateGraph, node_names: tuple[str, ...]) -> None:
    for node_name in node_names:
        graph.add_node(node_name, RUNNERS[node_name])


def _add_linear_edges(graph: StateGraph, node_names: tuple[str, ...]) -> None:
    for from_node, to_node in zip(node_names, node_names[1:]):
        graph.add_edge(from_node, to_node)


def create_graph(graph_variant: str = MAIN_FULL_GRAPH_VARIANT) -> StateGraph:
    _validate_graph_variant(graph_variant)

    graph = StateGraph(GraphState)
    _add_nodes(graph, FULL_GRAPH_NODES)
    _add_linear_edges(graph, PRE_SCREEN_PATH)
    graph.add_edge(_node("check"), _node("judge"))

    graph.add_conditional_edges(
        _node("screen"),
        route_after_screen,
        {
            _node("model"): _node("model"),
            _node("solve"): _node("solve"),
        },
    )
    graph.add_conditional_edges(
        _node("solve"),
        route_after_solve,
        {
            _node("check"): _node("check"),
            END_NODE: END,
        },
    )
    graph.add_conditional_edges(
        _node("judge"),
        route_after_judge,
        {
            _node("model"): _node("model"),
            _node("check"): _node("check"),
            END_NODE: END,
        },
    )

    graph.set_entry_point(_node("specify"))
    return graph


def create_generation_graph() -> StateGraph:
    graph = StateGraph(GraphState)
    _add_nodes(graph, GENERATION_PATH)
    _add_linear_edges(graph, GENERATION_PATH)
    graph.add_edge(_node("model"), END)
    graph.set_entry_point(_node("specify"))
    return graph


def create_app(graph_variant: str = MAIN_FULL_GRAPH_VARIANT):
    graph = create_graph(graph_variant=graph_variant)
    return graph.compile()


def create_generation_app():
    graph = create_generation_graph()
    return graph.compile()
