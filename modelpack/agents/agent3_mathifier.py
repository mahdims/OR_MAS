# modelpack/agents/agent3_mathifier.py
import structlog
from ..schemas import ModelPack, ComponentsMATH
from ..llm import llm_client
from ..prompts import PROMPTS

logger = structlog.get_logger(__name__)

async def a3_mathifier(state: ModelPack) -> ModelPack:
    """A3 - Mathifier: Convert NL components to mathematical notation."""

    logger.info("a3_mathifier_start", model_id=state.id)

    if not state.components_nl:
        logger.error("a3_no_components")
        return state

    try:
        user_prompt = f"""Natural Language Components:
{state.components_nl.model_dump_json(indent=2)}

Objective Sense: {state.context.get('objective_sense', 'minimize')}

Convert to mathematical notation in LaTeX. Preserve variable types (integer/continuous/binary)."""

        math_components = llm_client.structured_call(
            sys_prompt=PROMPTS["A3_mathifier"]["system"],
            user_prompt=user_prompt,
            pyd_model=ComponentsMATH,
            temperature=0.5
        )

        # Set objective sense from context
        if not math_components.sense:
            math_components.sense = state.context.get('objective_sense', 'min')

        state.components_math = math_components

        logger.info("a3_mathifier_success",
                   indices=len(math_components.indices),
                   constraints=len(math_components.constraints))

    except Exception as e:
        logger.error("a3_mathifier_error", error=str(e))

    return state
