"""Entry point for the Efficient Modeling Multi-Agent System."""
import asyncio
import argparse
import structlog
from dotenv import load_dotenv

from .schemas import ModelPack
from .orchestration.graph import create_app

load_dotenv()
logger = structlog.get_logger(__name__)


async def run_pipeline(problem_text: str) -> ModelPack:
    """Run the full modeling pipeline on a natural language problem."""
    logger.info("starting_pipeline", problem_length=len(problem_text))

    # Initialize state
    model_pack = ModelPack()
    model_pack.context["nl_problem"] = problem_text

    # Create and run app
    app = create_app()
    initial_state = {"model_pack": model_pack}

    # Execute pipeline
    result = await app.ainvoke(initial_state)

    logger.info("pipeline_complete", status=result["model_pack"].status)
    return result["model_pack"]


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert natural language optimization problems to Pyomo code"
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to file containing the problem description, or problem text directly"
    )
    parser.add_argument(
        "-o", "--output",
        default="output",
        help="Output directory for generated code (default: output)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(0)
        )

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
