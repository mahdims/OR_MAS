# modelpack/agents/agent7_checker.py
import structlog
from ..schemas import ModelPack, CodeBlob
from ..llm import llm_client
from ..prompts import PROMPTS, runtime_data_note

logger = structlog.get_logger(__name__)


async def a7_checker(state: ModelPack) -> ModelPack:
    """A7 - Solution Checker Author: Generate constraint verification code."""

    logger.info("a7_checker_start", model_id=state.id)

    if not state.components_nl:
        logger.error("a7_missing_prerequisites")
        return state

    try:
        # Get basic constraints
        basic_constraints = state.components_nl.constraints_basic
        nl_problem = str(state.context.get("nl_problem") or "")

        user_prompt = f"""Problem:
{nl_problem or 'Not available'}

NL:
{state.components_nl.model_dump_json(indent=2)}

Math:
{state.components_math.model_dump_json(indent=2) if state.components_math else 'Not available'}

{runtime_data_note()}

Task:
Return `SolutionChecker(data, solution, tolerance=1e-6)`.
Check only: {[c.name for c in basic_constraints]}."""

        code = llm_client.code_generation_call(
            sys_prompt=PROMPTS["A7_checker"]["system"],
            user_prompt=user_prompt,
            temperature=0.3,
            validate=True,
        )

        state.code.solution_checker = CodeBlob(
            language="python", filename="solution_checker.py", source=code
        )

        logger.info("a7_checker_success", code_length=len(code))

    except Exception as e:
        logger.error("a7_checker_error", error=str(e))

    return state
