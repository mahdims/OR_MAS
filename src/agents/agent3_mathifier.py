# modelpack/agents/agent3_mathifier.py
import structlog
from pydantic import BaseModel
from ..schemas import ModelPack, ComponentsMATH
from ..schemas import ComponentsNL, ContextContract
from ..llm import llm_client
from ..prompts import PROMPTS

logger = structlog.get_logger(__name__)


def _mathifier_prompt_suffix(state: ModelPack) -> str:
    if state.context.get("frontend_prompt_mode") != "strict":
        return ""

    return """

STRICT MATHEMATICAL RULES:
- Define only indices that are actually used by variables or parameters.
- Every parameter and variable index must come from a declared set.
- Keep maps_to aligned exactly with existing NL ids.
- Avoid speculative auxiliary constraints.
- Prefer fewer, cleaner constraints over broad inferred structure.
"""


async def a3_mathifier(state: ModelPack) -> ModelPack:
    """A3 - Mathifier: Convert NL components to mathematical notation."""

    logger.info("a3_mathifier_start", model_id=state.id)

    if not state.components_nl:
        logger.error("a3_no_components")
        return state

    try:
        prompt_suffix = _mathifier_prompt_suffix(state)
        user_prompt = f"""Natural Language Components:
{state.components_nl.model_dump_json(indent=2)}

Objective Sense: {state.context.get('objective_sense', 'minimize')}

Convert to mathematical notation in LaTeX. Preserve variable types (integer/continuous/binary).{prompt_suffix}"""

        math_components = llm_client.structured_call(
            sys_prompt=PROMPTS["A3_mathifier"]["system"],
            user_prompt=user_prompt,
            pyd_model=ComponentsMATH,
            temperature=0.5,
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


async def a1_a3_extract_mathify(state: ModelPack) -> ModelPack:
    """Combined A1+A3 pass: NL components + math components."""

    logger.info("a1_a3_extract_mathify_start", model_id=state.id)

    nl_problem = state.context.get("nl_problem", "")
    if not nl_problem:
        logger.error("a1_a3_no_problem")
        return state

    try:
        prompt_suffix = _mathifier_prompt_suffix(state)
        user_prompt = f"""Natural Language Problem:
{nl_problem}

Context:
- Objective: {state.context.get('objective_sense', 'not specified')}
- Assumptions: {state.context.get('assumptions', [])}

Produce both:
1. NL modeling components
2. Mathematical components in LaTeX form
Keep the two representations aligned one-to-one.{prompt_suffix}"""

        class ExtractedMathBundle(BaseModel):
            components: ComponentsNL
            math: ComponentsMATH

        sys_prompt = (
            f"{PROMPTS['A1_extractor']['system']}\n\n"
            f"{PROMPTS['A3_mathifier']['system']}\n\n"
            "Return a JSON object with fields 'components' and 'math'."
        )

        bundle = llm_client.structured_call(
            sys_prompt=sys_prompt,
            user_prompt=user_prompt,
            pyd_model=ExtractedMathBundle,
            temperature=0.45,
        )

        if not bundle.math.sense:
            bundle.math.sense = state.context.get("objective_sense", "min")

        state.components_nl = bundle.components
        state.components_math = bundle.math

        logger.info(
            "a1_a3_extract_mathify_success",
            sets=len(bundle.components.sets),
            params=len(bundle.components.parameters),
            vars=len(bundle.components.variables),
            indices=len(bundle.math.indices),
            constraints=len(bundle.math.constraints),
        )

    except Exception as e:
        logger.error("a1_a3_extract_mathify_error", error=str(e))

    return state


async def a0_a1_a3_frontend_bundle(state: ModelPack) -> ModelPack:
    """Combined A0+A1+A3 pass: contract, NL components, math components."""

    logger.info("a0_a1_a3_frontend_bundle_start", model_id=state.id)

    nl_problem = state.context.get("nl_problem", "")
    if not nl_problem:
        logger.error("a0_a1_a3_no_problem")
        return state

    try:
        prompt_suffix = _mathifier_prompt_suffix(state)
        user_prompt = f"""Natural Language Problem:
{nl_problem}

Produce in one pass:
1. A normalized problem contract
2. NL modeling components
3. Mathematical components in LaTeX form
Keep all three representations mutually consistent.{prompt_suffix}"""

        class FrontendBundle(BaseModel):
            contract: ContextContract
            components: ComponentsNL
            math: ComponentsMATH

        sys_prompt = (
            f"{PROMPTS['A0_specifier']['system']}\n\n"
            f"{PROMPTS['A1_extractor']['system']}\n\n"
            f"{PROMPTS['A3_mathifier']['system']}\n\n"
            "Return a JSON object with fields 'contract', 'components', and 'math'."
        )

        bundle = llm_client.structured_call(
            sys_prompt=sys_prompt,
            user_prompt=user_prompt,
            pyd_model=FrontendBundle,
            temperature=0.4,
        )

        if not bundle.math.sense:
            bundle.math.sense = bundle.contract.objective_sense

        state.context["assumptions"] = bundle.contract.assumptions
        state.context["units"] = bundle.contract.units
        state.context["objective_sense"] = bundle.contract.objective_sense
        state.context["scope"] = bundle.contract.scope
        state.context["deliverables"] = bundle.contract.deliverables
        state.components_nl = bundle.components
        state.components_math = bundle.math

        logger.info(
            "a0_a1_a3_frontend_bundle_success",
            objective_sense=bundle.contract.objective_sense,
            sets=len(bundle.components.sets),
            params=len(bundle.components.parameters),
            vars=len(bundle.components.variables),
            indices=len(bundle.math.indices),
            constraints=len(bundle.math.constraints),
        )

    except Exception as e:
        logger.error("a0_a1_a3_frontend_bundle_error", error=str(e))

    return state
