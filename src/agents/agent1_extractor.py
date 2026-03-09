# modelpack/agents/agent1_extractor.py
import structlog
from pydantic import BaseModel
from ..schemas import ModelPack, ComponentsNL
from ..schemas import ContextContract
from ..llm import llm_client
from ..prompts import PROMPTS, problem_input_note

logger = structlog.get_logger(__name__)


def _frontend_prompt_suffix(state: ModelPack) -> str:
    if state.context.get("frontend_prompt_mode") != "strict":
        return ""

    return """

STRICT FRONTEND RULES:
- Prefer a minimal, faithful model over a broad one.
- Only include sets, parameters, and variables that are explicitly supported by the text or clearly required by the objective/basic constraints.
- Every variable must appear in the objective or at least one constraint.
- Every parameter must correspond to a real quantity, cost, capacity, demand, limit, or coefficient from the problem.
- Avoid speculative auxiliary constraints or placeholders.
- Keep ids stable and human-readable.
"""


async def a1_extractor(state: ModelPack) -> ModelPack:
    """A1 - Extractor: Extract modeling components from the problem input."""

    logger.info("a1_extractor_start", model_id=state.id)

    nl_problem = state.context.get("nl_problem", "")
    if not nl_problem:
        logger.error("a1_no_problem")
        return state

    try:
        prompt_suffix = _frontend_prompt_suffix(state)
        input_note = problem_input_note(nl_problem)
        user_prompt = f"""Problem Input:
{nl_problem}

{input_note}

Context:
- Objective: {state.context.get('objective_sense', 'not specified')}
- Assumptions: {state.context.get('assumptions', [])}

Extract all modeling components. Mark variable types appropriately (integer/continuous/binary).{prompt_suffix}"""

        components = llm_client.structured_call(
            sys_prompt=PROMPTS["A1_extractor"]["system"],
            user_prompt=user_prompt,
            pyd_model=ComponentsNL,
            temperature=0.6,
        )

        state.components_nl = components

        logger.info(
            "a1_extractor_success",
            sets=len(components.sets),
            params=len(components.parameters),
            vars=len(components.variables),
        )

    except Exception as e:
        logger.error("a1_extractor_error", error=str(e))

    return state


async def a0_a1_specify_extract(state: ModelPack) -> ModelPack:
    """Combined A0+A1 frontend pass: problem contract + extracted components."""

    logger.info("a0_a1_specify_extract_start", model_id=state.id)

    nl_problem = state.context.get("nl_problem", "")
    if not nl_problem:
        logger.error("a0_a1_no_problem")
        return state

    try:
        prompt_suffix = _frontend_prompt_suffix(state)
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
2. The modeling components needed for optimization
{prompt_suffix}"""

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
