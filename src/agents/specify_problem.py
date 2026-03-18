# modelpack/agents/specify_problem.py
import structlog
from pydantic import BaseModel

from ..schemas import ComponentsNL, ContextContract, ModelPack
from ..llm import llm_client
from ..prompts import PROMPTS, llm_problem_text, problem_input_note

logger = structlog.get_logger(__name__)


async def specify_problem(state: ModelPack) -> ModelPack:
    """Specify the problem contract and extract NL components."""

    logger.info("specify_problem_start", model_id=state.id)

    nl_problem = state.context.get("nl_problem", "")
    if not nl_problem:
        logger.error("specify_problem_missing_input")
        return state

    try:
        llm_problem = llm_problem_text(nl_problem)
        input_note = problem_input_note(llm_problem)
        user_prompt = f"""Problem Input:
{llm_problem}

{input_note}

Authoritative interface constraints inside Problem Input:
- The required `create_model(...)` signature is binding interface metadata.
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
        trace_input = {
            "agent": "specify_problem",
            "upstream_artifacts": [
                {
                    "label": "problem_input",
                    "source": "llm_problem_text(state.context.nl_problem)",
                    "value": llm_problem,
                },
                {
                    "label": "problem_input_note",
                    "source": "problem_input_note(llm_problem)",
                    "value": input_note,
                },
            ],
        }

        result = llm_client.structured_call(
            sys_prompt=sys_prompt,
            user_prompt=user_prompt,
            pyd_model=SpecifiedComponents,
            temperature=0.45,
            trace_input=trace_input,
        )

        state.context["assumptions"] = result.contract.assumptions
        state.context["units"] = result.contract.units
        state.context["objective_sense"] = result.contract.objective_sense
        state.context["scope"] = result.contract.scope
        state.context["deliverables"] = result.contract.deliverables
        state.components_nl = result.components

        logger.info(
            "specify_problem_success",
            objective_sense=result.contract.objective_sense,
            sets=len(result.components.sets),
            params=len(result.components.parameters),
            vars=len(result.components.variables),
        )

    except Exception as e:
        logger.error("specify_problem_error", error=str(e))

    return state
