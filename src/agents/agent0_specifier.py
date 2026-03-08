# modelpack/agents/agent0_specifier.py
import structlog
from ..schemas import ModelPack, ContextContract
from ..llm import llm_client
from ..prompts import PROMPTS, problem_input_note

logger = structlog.get_logger(__name__)


async def a0_specifier(state: ModelPack) -> ModelPack:
    """A0 - Specifier: Extract problem contract from the problem input."""

    logger.info("a0_specifier_start", model_id=state.id)

    nl_problem = state.context.get("nl_problem", "")
    if not nl_problem:
        logger.error("a0_no_problem")
        return state

    try:
        input_note = problem_input_note(nl_problem)
        user_prompt = f"""Problem Input:
{nl_problem}

{input_note}

Extract the problem contract."""

        contract = llm_client.structured_call(
            sys_prompt=PROMPTS["A0_specifier"]["system"],
            user_prompt=user_prompt,
            pyd_model=ContextContract,
            temperature=0.5,
        )

        # Update context
        state.context["assumptions"] = contract.assumptions
        state.context["units"] = contract.units
        state.context["objective_sense"] = contract.objective_sense
        state.context["scope"] = contract.scope
        state.context["deliverables"] = contract.deliverables

        logger.info("a0_specifier_success", objective_sense=contract.objective_sense)

    except Exception as e:
        logger.error("a0_specifier_error", error=str(e))

    return state
