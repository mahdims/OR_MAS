# modelpack/agents/agent2_reviser.py
import structlog
from ..schemas import ModelPack, ComponentsNL, ComponentsNLMETA
from ..llm import llm_client
from ..prompts import PROMPTS
import json

logger = structlog.get_logger(__name__)

async def a2_reviser(state: ModelPack) -> ModelPack:
    """A2 - Reviser: Refine extracted components using Reflexion."""

    logger.info("a2_reviser_start", model_id=state.id)

    if not state.components_nl:
        logger.error("a2_no_components")
        return state

    try:
        user_prompt = f"""Original NL Problem:
{state.context.get('nl_problem', '')}

Extracted Components:
{state.components_nl.model_dump_json(indent=2)}

Review and refine. Be conservative - only fix clear issues."""

        # Create a combined model for response
        from pydantic import BaseModel
        class RevisedComponents(BaseModel):
            components: ComponentsNL
            meta: ComponentsNLMETA

        revised = llm_client.structured_call(
            sys_prompt=PROMPTS["A2_reviser"]["system"],
            user_prompt=user_prompt,
            pyd_model=RevisedComponents,
            temperature=0.5
        )

        state.components_nl = revised.components
        state.components_nl_meta = revised.meta

        logger.info("a2_reviser_success", edits=len(revised.meta.edits))

    except Exception as e:
        logger.error("a2_reviser_error", error=str(e))

    return state
