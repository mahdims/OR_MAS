"""Entry point for the Efficient Modeling Multi-Agent System."""

import asyncio
import argparse
from pathlib import Path
import structlog
from dotenv import load_dotenv

from .agents.build_model import _validate_create_model_entrypoint
from .llm import llm_client
from .prompts import PROMPTS, llm_problem_text
from .schemas import CodeBlob, ModelPack

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
logger = structlog.get_logger(__name__)
DEFAULT_GRAPH_VARIANT = "main"


def _attach_llm_trace(model_pack: ModelPack, trace_payload: dict[str, object]) -> None:
    detailed_calls = [
        dict(call) for call in trace_payload.get("calls", []) if isinstance(call, dict)
    ]
    model_pack.tests["llm_trace"] = detailed_calls


async def run_pipeline(
    problem_text: str,
    target_interface: str = "",
    graph_variant: str = DEFAULT_GRAPH_VARIANT,
) -> ModelPack:
    """Run the full modeling pipeline on a natural language problem."""
    logger.info("starting_pipeline", problem_length=len(problem_text))

    # Initialize state
    model_pack = ModelPack()
    model_pack.context["nl_problem"] = problem_text
    target_interface = (target_interface or "").strip()
    if target_interface:
        model_pack.context["target_interface"] = target_interface

    # Create and run app
    from .orchestration.graph import create_app

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
) -> ModelPack:
    """Run the minimal MAS path and stop after model generation."""
    logger.info("starting_generation_pipeline", problem_length=len(problem_text))

    model_pack = ModelPack()
    model_pack.context["nl_problem"] = problem_text
    model_pack.context["target_interface"] = "create_model"

    from .orchestration.graph import create_generation_app

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


async def run_single_agent_generation(
    problem_text: str,
) -> ModelPack:
    """Run a direct single-agent create_model baseline on the provided input."""
    logger.info("starting_single_agent_generation", problem_length=len(problem_text))

    model_pack = ModelPack()
    model_pack.context["nl_problem"] = problem_text
    model_pack.context["target_interface"] = "create_model"

    system_prompt = PROMPTS["single_agent_create_model"]["system"]
    llm_problem = llm_problem_text(
        problem_text,
        preserve_data_generator_contract=True,
    )

    trace_token = llm_client.begin_trace()
    try:
        trace_input = {
            "agent": "single_agent_create_model",
            "upstream_artifacts": [
                {
                    "label": "problem_input",
                    "source": "llm_problem_text(problem_text)",
                    "value": llm_problem,
                }
            ],
        }
        code = llm_client.code_generation_call(
            sys_prompt=system_prompt,
            user_prompt=llm_problem,
            temperature=0.0,
            validate=True,
            trace_input=trace_input,
        )
        valid, diagnostics = _validate_create_model_entrypoint(code)
        if not valid:
            joined = ", ".join(diagnostics)
            raise ValueError(f"single_agent_create_model_validation_failed: {joined}")

        model_pack.code.model_builder = CodeBlob(
            language="python",
            filename="create_model.py",
            source=code,
        )
        model_pack.status = "generated"
    except Exception as exc:
        logger.error("single_agent_generation_error", error=str(exc))
        model_pack.status = "error"
    finally:
        trace_payload = llm_client.end_trace(trace_token)
        _attach_llm_trace(model_pack, trace_payload)

    logger.info("single_agent_generation_complete", status=model_pack.status)
    return model_pack


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
    if result.code.datagen:
        print(f"Data Generator: {result.code.datagen.filename}")
    if result.code.solution_checker:
        print(f"Solution Checker: {result.code.solution_checker.filename}")

    return 0


if __name__ == "__main__":
    exit(main())
