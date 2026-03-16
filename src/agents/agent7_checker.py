# modelpack/agents/agent7_checker.py
import json

import structlog
from ..schemas import ModelPack, CodeBlob
from ..llm import llm_client
from ..prompts import PROMPTS, compact_feedback_context, runtime_data_note

logger = structlog.get_logger(__name__)


async def a7_checker(state: ModelPack) -> ModelPack:
    """A7 - Solution Checker Author: Generate constraint verification code."""

    logger.info("a7_checker_start", model_id=state.id)

    if not state.components_nl:
        logger.error("a7_missing_prerequisites")
        return state

    try:
        # Get basic constraints
        basic_constraints = state.components_nl.constraints_basic
        basic_constraints_json = json.dumps(
            [{"name": c.name, "desc": c.desc} for c in basic_constraints],
            indent=2,
        )
        feedback_note = ""
        feedback = state.tests.get("last_feedback")
        if feedback and feedback.target_agent == "A7":
            feedback_note = compact_feedback_context(feedback)

        user_prompt_sections = [
            "Basic constraints:",
            basic_constraints_json,
            runtime_data_note(),
        ]
        if feedback_note:
            user_prompt_sections.extend(["Targeted feedback:", feedback_note])
            if state.code.solution_checker and state.code.solution_checker.source:
                user_prompt_sections.extend(
                    [
                        "Existing checker code to repair:",
                        f"```python\n{state.code.solution_checker.source}\n```",
                    ]
                )
        user_prompt_sections.extend(
            [
                "Task:",
                (
                    "Return `SolutionChecker(data, solution, tolerance=1e-6)`.\n"
                    "Check only the listed constraints."
                ),
            ]
        )
        user_prompt = "\n\n".join(user_prompt_sections)

        code = llm_client.code_generation_call(
            sys_prompt=PROMPTS["A7_checker"]["system"],
            user_prompt=user_prompt,
            temperature=0.3,
            validate=True,
        )

        state.code.solution_checker = CodeBlob(
            language="python", filename="solution_checker.py", source=code
        )

        logger.info("a7_checker_success", code_length=len(code))

    except Exception as e:
        logger.error("a7_checker_error", error=str(e))

    return state
