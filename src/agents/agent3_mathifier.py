# modelpack/agents/agent3_mathifier.py
import structlog

from ..schemas import ComponentsMATH, ModelPack
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
        nl_components_json = state.components_nl.model_dump_json(indent=2)
        objective_sense = state.context.get("objective_sense", "minimize")
        user_prompt = f"""Natural Language Components:
{nl_components_json}

Objective Sense: {objective_sense}

Convert to mathematical notation in LaTeX.
Preserve upstream ids via maps_to and preserve tuple/index order exactly as described in the NL components.
Map every explicit NL requirement to a math constraint. Stay compact and non-speculative."""
        trace_input = {
            "agent": "derive_math",
            "upstream_artifacts": [
                {
                    "label": "components_nl",
                    "source": "state.components_nl",
                    "value": nl_components_json,
                },
                {
                    "label": "objective_sense",
                    "source": "state.context.objective_sense",
                    "value": objective_sense,
                },
            ],
        }

        math_components = llm_client.structured_call(
            sys_prompt=PROMPTS["A3_mathifier"]["system"],
            user_prompt=user_prompt,
            pyd_model=ComponentsMATH,
            temperature=0.5,
            trace_input=trace_input,
        )

        # Set objective sense from context
        if not math_components.sense:
            math_components.sense = state.context.get("objective_sense", "min")

        state.components_math = math_components

        logger.info(
            "a3_mathifier_success",
            indices=len(math_components.indices),
            constraints=len(math_components.constraints),
        )

    except Exception as e:
        logger.error("a3_mathifier_error", error=str(e))

    return state
