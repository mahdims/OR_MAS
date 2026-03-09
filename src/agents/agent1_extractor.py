# modelpack/agents/agent1_extractor.py
import structlog
from pydantic import BaseModel

from ..schemas import ComponentsNL, ContextContract, ModelPack
from ..llm import llm_client
from ..prompts import PROMPTS, problem_input_note

logger = structlog.get_logger(__name__)


async def a0_a1_specify_extract(state: ModelPack) -> ModelPack:
    """Combined A0+A1 frontend pass used by the benchmarked full graph."""

    logger.info("a0_a1_specify_extract_start", model_id=state.id)

    nl_problem = state.context.get("nl_problem", "")
    if not nl_problem:
        logger.error("a0_a1_no_problem")
        return state

    try:
        input_note = problem_input_note(nl_problem)
        user_prompt = f"""Problem Input:
{nl_problem}

{input_note}

Authoritative interface constraints inside Problem Input:
- The `DataGenerator contract` block and required `create_model(...)` signature are binding interface constraints.
- If those identifiers correspond to actual sets or parameters, preserve them verbatim in ids and names.
- Do not treat wrapper/interface instructions as extra domain content.

Produce both:
1. A normalized problem contract
2. The modeling components needed for optimization"""

        class SpecifiedComponents(BaseModel):
            contract: ContextContract
            components: ComponentsNL

        sys_prompt = (
            f"{PROMPTS['A0_specifier']['system']}\n\n"
            f"{PROMPTS['A1_extractor']['system']}\n\n"
            "Return a JSON object with fields 'contract' and 'components'."
        )

        result = llm_client.structured_call(
            sys_prompt=sys_prompt,
            user_prompt=user_prompt,
            pyd_model=SpecifiedComponents,
            temperature=0.45,
        )

        state.context["assumptions"] = result.contract.assumptions
        state.context["units"] = result.contract.units
        state.context["objective_sense"] = result.contract.objective_sense
        state.context["scope"] = result.contract.scope
        state.context["deliverables"] = result.contract.deliverables
        state.components_nl = result.components

        logger.info(
            "a0_a1_specify_extract_success",
            objective_sense=result.contract.objective_sense,
            sets=len(result.components.sets),
            params=len(result.components.parameters),
            vars=len(result.components.variables),
        )

    except Exception as e:
        logger.error("a0_a1_specify_extract_error", error=str(e))

    return state
