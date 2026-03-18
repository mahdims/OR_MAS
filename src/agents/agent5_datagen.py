# modelpack/agents/agent5_datagen.py
import structlog
from ..schemas import ModelPack, CodeBlob
from ..llm import llm_client
from ..prompts import PROMPTS, compact_feedback_context, llm_problem_text, runtime_data_note

logger = structlog.get_logger(__name__)


async def a5_datagen(state: ModelPack) -> ModelPack:
    """A5 - DataGen Author: Generate data generation code."""

    logger.info("a5_datagen_start", model_id=state.id)

    if not state.components_math:
        logger.error("a5_missing_prerequisites")
        return state

    try:
        # Check for feedback
        feedback_context = ""
        feedback = state.tests.get("last_feedback")
        if feedback and feedback.target_agent == "A5":
            feedback_note = compact_feedback_context(feedback)
            if feedback_note:
                feedback_context = f"Targeted feedback:\n{feedback_note}\n"

        nl_problem = llm_problem_text(state.context.get("nl_problem") or "")
        nl_components_json = (
            state.components_nl.model_dump_json(indent=2)
            if state.components_nl is not None
            else "Not available"
        )
        math_components_json = (
            state.components_math.model_dump_json(indent=2)
            if state.components_math is not None
            else "Not available"
        )
        trace_input = {
            "agent": "A5_datagen",
            "upstream_artifacts": [
                {
                    "label": "problem_input",
                    "source": "llm_problem_text(state.context.nl_problem)",
                    "value": nl_problem,
                },
                {
                    "label": "components_nl",
                    "source": "state.components_nl",
                    "value": nl_components_json,
                },
                {
                    "label": "components_math",
                    "source": "state.components_math",
                    "value": math_components_json,
                },
            ],
        }
        if feedback_context:
            trace_input["upstream_artifacts"].append(
                {
                    "label": "targeted_feedback",
                    "source": "state.tests.last_feedback",
                    "value": feedback_context,
                }
            )

        user_prompt = f"""Problem:
{nl_problem or 'Not available'}

NL:
{nl_components_json}

Math:
{math_components_json}

{runtime_data_note()}

{feedback_context}

Task:
Generate feasible data."""

        code = llm_client.code_generation_call(
            sys_prompt=PROMPTS["A5_datagen"]["system"],
            user_prompt=user_prompt,
            temperature=0.3,
            validate=True,
            trace_input=trace_input,
        )

        state.code.datagen = CodeBlob(language="python", filename="datagen.py", source=code)

        logger.info("a5_datagen_success", code_length=len(code))

    except Exception as e:
        logger.error("a5_datagen_error", error=str(e))

    return state
