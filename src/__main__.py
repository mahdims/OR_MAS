"""Entry point for the Efficient Modeling Multi-Agent System."""

import asyncio
import argparse
import structlog
from dotenv import load_dotenv

from .llm import llm_client
from .schemas import ModelPack
from .orchestration.graph import MAIN_FULL_GRAPH_VARIANT, create_app, create_generation_app

load_dotenv()
logger = structlog.get_logger(__name__)


def _attach_llm_trace(model_pack: ModelPack, trace_payload: dict[str, object]) -> None:
    model_pack.tests["llm_calls"] = trace_payload.get("calls", [])
    model_pack.tests["llm_usage_summary"] = trace_payload.get("summary", {})


async def run_pipeline(
    problem_text: str,
    target_interface: str = "",
    generation_mode: str = "repair2",
    graph_variant: str = MAIN_FULL_GRAPH_VARIANT,
) -> ModelPack:
    """Run the full modeling pipeline on a natural language problem."""
    logger.info("starting_pipeline", problem_length=len(problem_text))

    # Initialize state
    model_pack = ModelPack()
    model_pack.context["nl_problem"] = problem_text
    target_interface = (target_interface or "").strip()
    if target_interface:
        model_pack.context["target_interface"] = target_interface
        model_pack.context["generation_mode"] = (
            (generation_mode or "repair2").strip() or "repair2"
        )

    # Create and run app
    app = create_app(graph_variant=graph_variant)
    initial_state = {"model_pack": model_pack}

    # Execute pipeline
    trace_token = llm_client.begin_trace()
    result = None
    try:
        result = await app.ainvoke(initial_state)
    finally:
        trace_payload = llm_client.end_trace(trace_token)
        target_model_pack = result["model_pack"] if result is not None else model_pack
        _attach_llm_trace(target_model_pack, trace_payload)

    logger.info("pipeline_complete", status=result["model_pack"].status)
    return result["model_pack"]


async def run_generation_pipeline(
    problem_text: str,
    generation_mode: str = "repair2",
) -> ModelPack:
    """Run the generation-only pipeline (A0->A4) on a natural language problem."""
    logger.info("starting_generation_pipeline", problem_length=len(problem_text))

    model_pack = ModelPack()
    model_pack.context["nl_problem"] = problem_text
    model_pack.context["target_interface"] = "create_model"
    model_pack.context["generation_mode"] = (generation_mode or "repair2").strip() or "repair2"

    app = create_generation_app()
    initial_state = {"model_pack": model_pack}

    trace_token = llm_client.begin_trace()
    result = None
    try:
        result = await app.ainvoke(initial_state)
    finally:
        trace_payload = llm_client.end_trace(trace_token)
        target_model_pack = result["model_pack"] if result is not None else model_pack
        _attach_llm_trace(target_model_pack, trace_payload)

    logger.info("generation_pipeline_complete", status=result["model_pack"].status)
    return result["model_pack"]


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert natural language optimization problems to Pyomo code"
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to file containing the problem description, or problem text directly",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output",
        help="Output directory for generated code (default: output)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(0))

    # Get problem text
    if args.input:
        try:
            with open(args.input, "r") as f:
                problem_text = f.read()
        except FileNotFoundError:
            # Treat as direct text input
            problem_text = args.input
    else:
        print("Enter your optimization problem (Ctrl+D or Ctrl+Z to finish):")
        import sys

        problem_text = sys.stdin.read()

    if not problem_text.strip():
        print("Error: No problem text provided")
        return 1

    # Run pipeline
    result = asyncio.run(run_pipeline(problem_text))

    # Output results
    print(f"\n{'='*60}")
    print("Pipeline Complete!")
    print(f"{'='*60}")
    print(f"Status: {result.status}")

    if result.code.model_builder:
        print(f"\nModel Builder: {result.code.model_builder.filename}")
    if result.code.data_schema:
        print(f"Data Schema: {result.code.data_schema.filename}")
    if result.code.datagen:
        print(f"Data Generator: {result.code.datagen.filename}")
    if result.code.solution_checker:
        print(f"Solution Checker: {result.code.solution_checker.filename}")

    return 0


if __name__ == "__main__":
    exit(main())
