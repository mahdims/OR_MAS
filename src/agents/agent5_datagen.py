# modelpack/agents/agent5_datagen.py
import structlog
from ..schemas import ModelPack, CodeBlob
from ..llm import llm_client
from ..prompts import PROMPTS, compact_feedback_context, runtime_data_note

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

        nl_problem = str(state.context.get("nl_problem") or "")
        nl_components_json = (
            state.components_nl.model_dump_json(indent=2)
            if state.components_nl is not None
            else "Not available"
        )

        user_prompt = f"""Problem:
{nl_problem or 'Not available'}

NL:
{nl_components_json}

Math:
{state.components_math.model_dump_json(indent=2) if state.components_math else 'Not available'}

{runtime_data_note()}

{feedback_context}

Task:
Return `DataGen(seed: int) -> dict`.
Generate feasible data."""

        code = llm_client.code_generation_call(
            sys_prompt=PROMPTS["A5_datagen"]["system"],
            user_prompt=user_prompt,
            temperature=0.3,
            validate=True,
        )

        state.code.datagen = CodeBlob(language="python", filename="datagen.py", source=code)

        logger.info("a5_datagen_success", code_length=len(code))

    except Exception as e:
        logger.error("a5_datagen_error", error=str(e))

    return state
