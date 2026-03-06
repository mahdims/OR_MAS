# modelpack/orchestration/graph.py
from typing import TypedDict
from langgraph.graph import StateGraph, END
import structlog

from ..schemas import ModelPack
from ..agents import (
    agent0_specifier,
    agent1_extractor,
    agent2_reviser,
    agent3_mathifier,
    agent3b_data_extractor,
    agent3c_schema,
    agent4_pyomo,
    agent5_datagen,
    agent6_screen,
    agent7_checker,
    agent8_solver,
    agent9_judge,
)

logger = structlog.get_logger(__name__)

# Best current full-graph baseline from local ablations.
MAIN_FULL_GRAPH_VARIANT = "no_a2"
GRAPH_VARIANT_ALIASES = {"main": MAIN_FULL_GRAPH_VARIANT}
FULL_GRAPH_VARIANTS = {
    "full",
    "no_a2",
    "no_a9",
    "merge_a1_a2",
    "no_a2_no_a3b",
    "no_a2_no_a9",
    "no_a2_no_a7_a9",
    "no_a2_frontend_strict",
    "no_a2_merge_a0_a1",
    "no_a2_merge_a0_a1_strict",
    "no_a2_merge_a1_a3",
    "no_a2_merge_a0_a1_a3",
}


class GraphState(TypedDict):
    model_pack: ModelPack


# Wrapper functions
async def run_a0(state: GraphState) -> GraphState:
    state["model_pack"] = await agent0_specifier.a0_specifier(state["model_pack"])
    return state


async def run_a1(state: GraphState) -> GraphState:
    state["model_pack"] = await agent1_extractor.a1_extractor(state["model_pack"])
    return state


async def run_a1_strict(state: GraphState) -> GraphState:
    state["model_pack"].context["frontend_prompt_mode"] = "strict"
    return await run_a1(state)


async def run_a0_a1_merged(state: GraphState) -> GraphState:
    state["model_pack"] = await agent1_extractor.a0_a1_specify_extract(state["model_pack"])
    return state


async def run_a0_a1_merged_strict(state: GraphState) -> GraphState:
    state["model_pack"].context["frontend_prompt_mode"] = "strict"
    return await run_a0_a1_merged(state)


async def run_a1_a2_merged(state: GraphState) -> GraphState:
    state = await run_a1(state)
    state = await run_a2(state)
    return state


async def run_a2(state: GraphState) -> GraphState:
    state["model_pack"] = await agent2_reviser.a2_reviser(state["model_pack"])
    return state


async def run_a3(state: GraphState) -> GraphState:
    state["model_pack"] = await agent3_mathifier.a3_mathifier(state["model_pack"])
    return state


async def run_a3_strict(state: GraphState) -> GraphState:
    state["model_pack"].context["frontend_prompt_mode"] = "strict"
    return await run_a3(state)


async def run_a1_a3_merged(state: GraphState) -> GraphState:
    state["model_pack"] = await agent3_mathifier.a1_a3_extract_mathify(state["model_pack"])
    return state


async def run_a0_a1_a3_merged(state: GraphState) -> GraphState:
    state["model_pack"] = await agent3_mathifier.a0_a1_a3_frontend_bundle(state["model_pack"])
    return state


async def run_a3b(state: GraphState) -> GraphState:
    state["model_pack"] = await agent3b_data_extractor.a3b_data_extractor(state["model_pack"])
    return state


async def run_a3c(state: GraphState) -> GraphState:
    state["model_pack"] = await agent3c_schema.a3c_schema(state["model_pack"])
    return state


async def run_a4(state: GraphState) -> GraphState:
    state["model_pack"] = await agent4_pyomo.a4_pyomo(state["model_pack"])
    return state


async def run_a5(state: GraphState) -> GraphState:
    state["model_pack"] = await agent5_datagen.a5_datagen(state["model_pack"])
    return state


async def run_a6(state: GraphState) -> GraphState:
    state["model_pack"] = await agent6_screen.a6_screen(state["model_pack"])
    return state


async def run_a7(state: GraphState) -> GraphState:
    state["model_pack"] = await agent7_checker.a7_checker(state["model_pack"])
    return state


async def run_a8(state: GraphState) -> GraphState:
    state["model_pack"] = await agent8_solver.a8_solver(state["model_pack"])
    return state


async def run_a9(state: GraphState) -> GraphState:
    state["model_pack"] = await agent9_judge.a9_judge(state["model_pack"])
    return state


# Routing functions with feedback loops
def route_after_a6(state: GraphState) -> str:
    """Route after feasibility screen."""
    feedback = state["model_pack"].tests.get("last_feedback")

    if feedback:
        if feedback.target_agent == "A4":
            logger.info("routing_to_a4", issue=feedback.issue)
            return "A4_pyomo"
        elif feedback.target_agent == "A5":
            logger.info("routing_to_a5", issue=feedback.issue)
            return "A5_datagen"

    return "A7_checker"


def route_after_a9(state: GraphState) -> str:
    """Route after cross-validation."""
    feedback = state["model_pack"].tests.get("last_feedback")

    if feedback:
        if feedback.issue == "checker_false_negative":
            return "A7_checker"
        elif feedback.issue == "data_infeasible":
            return "A5_datagen"
        elif feedback.issue == "code_build_error":
            return "A4_pyomo"

    return "END"


def _normalize_graph_variant(graph_variant: str) -> str:
    normalized = (graph_variant or MAIN_FULL_GRAPH_VARIANT).strip().lower()
    normalized = GRAPH_VARIANT_ALIASES.get(normalized, normalized)
    if normalized not in FULL_GRAPH_VARIANTS:
        raise ValueError(f"Unsupported full graph variant: {graph_variant}")
    return normalized


def create_graph(graph_variant: str = MAIN_FULL_GRAPH_VARIANT) -> StateGraph:
    """Create orchestration graph with feedback loops."""

    graph_variant = _normalize_graph_variant(graph_variant)
    merge_a1_a2 = graph_variant == "merge_a1_a2"
    merge_a0_a1 = graph_variant in {"no_a2_merge_a0_a1", "no_a2_merge_a0_a1_strict"}
    merge_a1_a3 = graph_variant == "no_a2_merge_a1_a3"
    merge_a0_a1_a3 = graph_variant == "no_a2_merge_a0_a1_a3"
    frontend_strict = graph_variant in {"no_a2_frontend_strict", "no_a2_merge_a0_a1_strict"}
    skip_a2 = graph_variant in {
        "no_a2",
        "no_a2_no_a3b",
        "no_a2_no_a9",
        "no_a2_no_a7_a9",
        "no_a2_frontend_strict",
        "no_a2_merge_a0_a1",
        "no_a2_merge_a0_a1_strict",
        "no_a2_merge_a1_a3",
        "no_a2_merge_a0_a1_a3",
    }
    skip_a3b = graph_variant == "no_a2_no_a3b"
    skip_a7 = graph_variant == "no_a2_no_a7_a9"
    skip_a9 = graph_variant in {"no_a9", "no_a2_no_a9", "no_a2_no_a7_a9"}

    graph = StateGraph(GraphState)

    # Add nodes
    if merge_a0_a1_a3:
        graph.add_node("A0A1A3_frontend_bundle", run_a0_a1_a3_merged)
    else:
        if merge_a0_a1:
            graph.add_node(
                "A0A1_specify_extract",
                run_a0_a1_merged_strict if frontend_strict else run_a0_a1_merged,
            )
        else:
            graph.add_node("A0_specifier", run_a0)
            if merge_a1_a3:
                graph.add_node("A1A3_extract_mathify", run_a1_a3_merged)
            elif merge_a1_a2:
                graph.add_node("A1A2_extract_revise", run_a1_a2_merged)
            else:
                graph.add_node("A1_extractor", run_a1_strict if frontend_strict else run_a1)
                if not skip_a2:
                    graph.add_node("A2_reviser", run_a2)
        if not merge_a1_a3:
            graph.add_node("A3_mathifier", run_a3_strict if frontend_strict else run_a3)

    if not skip_a3b:
        graph.add_node("A3B_data_extractor", run_a3b)
    graph.add_node("A3C_schema", run_a3c)
    graph.add_node("A4_pyomo", run_a4)
    graph.add_node("A5_datagen", run_a5)
    graph.add_node("A6_screen", run_a6)
    if not skip_a7:
        graph.add_node("A7_checker", run_a7)
    graph.add_node("A8_solver", run_a8)
    if not skip_a9:
        graph.add_node("A9_judge", run_a9)

    # Add edges (happy path)
    if merge_a0_a1_a3:
        graph.add_edge("A0A1A3_frontend_bundle", "A3C_schema" if skip_a3b else "A3B_data_extractor")
    elif merge_a0_a1:
        graph.add_edge("A0A1_specify_extract", "A3_mathifier")
        graph.add_edge("A3_mathifier", "A3C_schema" if skip_a3b else "A3B_data_extractor")
    elif merge_a1_a3:
        graph.add_edge("A0_specifier", "A1A3_extract_mathify")
        graph.add_edge("A1A3_extract_mathify", "A3C_schema" if skip_a3b else "A3B_data_extractor")
    elif merge_a1_a2:
        graph.add_edge("A0_specifier", "A1A2_extract_revise")
        graph.add_edge("A1A2_extract_revise", "A3_mathifier")
        graph.add_edge("A3_mathifier", "A3C_schema" if skip_a3b else "A3B_data_extractor")
    else:
        graph.add_edge("A0_specifier", "A1_extractor")
        if skip_a2:
            graph.add_edge("A1_extractor", "A3_mathifier")
        else:
            graph.add_edge("A1_extractor", "A2_reviser")
            graph.add_edge("A2_reviser", "A3_mathifier")
        if skip_a3b:
            graph.add_edge("A3_mathifier", "A3C_schema")
        else:
            graph.add_edge("A3_mathifier", "A3B_data_extractor")
            graph.add_edge("A3B_data_extractor", "A3C_schema")
    if not skip_a3b and (merge_a0_a1 or merge_a1_a3 or merge_a0_a1_a3):
        graph.add_edge("A3B_data_extractor", "A3C_schema")
    graph.add_edge("A3C_schema", "A4_pyomo")
    graph.add_edge("A4_pyomo", "A5_datagen")
    graph.add_edge("A5_datagen", "A6_screen")
    # Conditional after A6
    if not skip_a7:
        graph.add_edge("A7_checker", "A8_solver")
    if skip_a9:
        graph.add_edge("A8_solver", END)
    else:
        graph.add_edge("A8_solver", "A9_judge")

    # Add conditional edges for feedback
    graph.add_conditional_edges(
        "A6_screen",
        route_after_a6,
        {
            "A4_pyomo": "A4_pyomo",
            "A5_datagen": "A5_datagen",
            "A7_checker": "A8_solver" if skip_a7 else "A7_checker",
        },
    )

    if not skip_a9:
        graph.add_conditional_edges(
            "A9_judge",
            route_after_a9,
            {
                "A7_checker": "A7_checker",
                "A5_datagen": "A5_datagen",
                "A4_pyomo": "A4_pyomo",
                "END": END,
            },
        )

    # Set entry point
    if merge_a0_a1_a3:
        graph.set_entry_point("A0A1A3_frontend_bundle")
    elif merge_a0_a1:
        graph.set_entry_point("A0A1_specify_extract")
    else:
        graph.set_entry_point("A0_specifier")

    return graph


def create_generation_graph() -> StateGraph:
    """Create generation-only graph (A0->A4) without validation loop agents."""

    graph = StateGraph(GraphState)

    graph.add_node("A0_specifier", run_a0)
    graph.add_node("A1_extractor", run_a1)
    graph.add_node("A2_reviser", run_a2)
    graph.add_node("A3_mathifier", run_a3)
    graph.add_node("A3B_data_extractor", run_a3b)
    graph.add_node("A3C_schema", run_a3c)
    graph.add_node("A4_pyomo", run_a4)

    graph.add_edge("A0_specifier", "A1_extractor")
    graph.add_edge("A1_extractor", "A2_reviser")
    graph.add_edge("A2_reviser", "A3_mathifier")
    graph.add_edge("A3_mathifier", "A3B_data_extractor")
    graph.add_edge("A3B_data_extractor", "A3C_schema")
    graph.add_edge("A3C_schema", "A4_pyomo")
    graph.add_edge("A4_pyomo", END)

    graph.set_entry_point("A0_specifier")
    return graph


def create_app(graph_variant: str = MAIN_FULL_GRAPH_VARIANT):
    """Create compiled app without checkpointing."""
    graph = create_graph(graph_variant=graph_variant)
    return graph.compile()


def create_generation_app():
    """Create compiled generation-only app without checkpointing."""
    graph = create_generation_graph()
    return graph.compile()
