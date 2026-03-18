"""Entry point for the Efficient Modeling Multi-Agent System."""

import asyncio
import argparse
import structlog
from dotenv import load_dotenv

from .agents.agent4_pyomo import _validate_create_model_entrypoint
from .llm import llm_client
from .prompts import PROMPTS, llm_problem_text
from .schemas import CodeBlob, ModelPack
from .orchestration.graph import MAIN_FULL_GRAPH_VARIANT, create_app, create_generation_app

load_dotenv()
logger = structlog.get_logger(__name__)


def _compact_llm_call(call: dict[str, object]) -> dict[str, object]:
    keys = [
        "sequence",
        "call_type",
        "caller",
        "provider",
        "model_name",
        "response_model",
        "temperature",
        "started_at",
        "latency_seconds",
        "success",
        "error",
        "finish_reason",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "usage",
    ]
    return {key: call.get(key) for key in keys}


def _attach_llm_trace(model_pack: ModelPack, trace_payload: dict[str, object]) -> None:
    detailed_calls = [
        dict(call) for call in trace_payload.get("calls", []) if isinstance(call, dict)
    ]
    model_pack.tests["llm_trace"] = detailed_calls
    model_pack.tests["llm_calls"] = [_compact_llm_call(call) for call in detailed_calls]
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
        model_pack.context["generation_mode"] = (generation_mode or "repair2").strip() or "repair2"

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
    """Run the minimal MAS path and stop after A4 Pyomo generation."""
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


async def run_single_agent_generation(
    problem_text: str,
    generation_mode: str = "repair2",
) -> ModelPack:
    """Run a direct single-agent create_model baseline on the provided input."""
    logger.info("starting_single_agent_generation", problem_length=len(problem_text))

    model_pack = ModelPack()
    model_pack.context["nl_problem"] = problem_text
    model_pack.context["target_interface"] = "create_model"

    normalized_mode = (generation_mode or "repair2").strip().lower()
    if normalized_mode not in {"prompt_only", "repair2"}:
        normalized_mode = "repair2"
    model_pack.context["generation_mode"] = normalized_mode

    system_prompt = PROMPTS["single_agent_create_model"]["system"]
    llm_problem = llm_problem_text(problem_text)

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
        if not valid and normalized_mode == "repair2":
            repair_iterations = model_pack.tests.setdefault("repair_iterations", {})
            repair_iterations["single_agent_validation"] = (
                int(repair_iterations.get("single_agent_validation") or 0) + 1
            )
            diagnostic_lines = "\n".join(f"- {item}" for item in diagnostics)
            repair_prompt = f"""{llm_problem}

Validation diagnostics from the previous attempt:
{diagnostic_lines}

Previous code to repair:
```python
{code}
```

Return corrected code only."""
            repair_trace_input = {
                "agent": "single_agent_create_model",
                "upstream_artifacts": trace_input["upstream_artifacts"]
                + [
                    {
                        "label": "validation_diagnostics",
                        "source": "_validate_create_model_entrypoint(previous_code)",
                        "value": diagnostics,
                    },
                    {
                        "label": "previous_code",
                        "source": "previous_llm_output",
                        "value": code,
                    },
                ],
            }
            code = llm_client.code_generation_call(
                sys_prompt=system_prompt,
                user_prompt=repair_prompt,
                temperature=0.0,
                validate=True,
                trace_input=repair_trace_input,
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
