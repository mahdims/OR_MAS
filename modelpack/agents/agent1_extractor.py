# modelpack/agents/agent1_extractor.py
import structlog
from ..schemas import ModelPack, ComponentsNL
from ..llm import llm_client
from ..prompts import PROMPTS

logger = structlog.get_logger(__name__)

async def a1_extractor(state: ModelPack) -> ModelPack:
    """A1 - Extractor: Extract modeling components from NL."""

    logger.info("a1_extractor_start", model_id=state.id)

    nl_problem = state.context.get("nl_problem", "")
    if not nl_problem:
        logger.error("a1_no_problem")
        return state

    try:
        user_prompt = f"""Natural Language Problem:
{nl_problem}

Context:
- Objective: {state.context.get('objective_sense', 'not specified')}
- Assumptions: {state.context.get('assumptions', [])}

Extract all modeling components. Mark variable types appropriately (integer/continuous/binary)."""

        components = llm_client.structured_call(
            sys_prompt=PROMPTS["A1_extractor"]["system"],
            user_prompt=user_prompt,
            pyd_model=ComponentsNL,
            temperature=0.6
        )

        state.components_nl = components

        logger.info("a1_extractor_success",
                   sets=len(components.sets),
                   params=len(components.parameters),
                   vars=len(components.variables))

    except Exception as e:
        logger.error("a1_extractor_error", error=str(e))

    return state
