# modelpack/agents/agent3c_schema.py
import structlog
from ..schemas import ModelPack, CodeBlob
from ..llm import llm_client
from ..prompts import PROMPTS

logger = structlog.get_logger(__name__)

async def a3c_schema(state: ModelPack) -> ModelPack:
    """A3C - Schema Generator: Generate Data class structure."""

    logger.info("a3c_schema_start", model_id=state.id)

    if not state.components_math:
        logger.error("a3c_no_math_components")
        return state

    try:
        user_prompt = f"""Mathematical Components:
{state.components_math.model_dump_json(indent=2)}

Generate a generic Data class using dictionaries for parameters and lists for sets.
NO hardcoded attributes for specific instances.
Use tuple keys for multi-indexed parameters."""

        code = llm_client.code_generation_call(
            sys_prompt=PROMPTS["A3C_schema"]["system"],
            user_prompt=user_prompt,
            temperature=0.3,
            validate=True
        )

        state.code.data_schema = CodeBlob(
            language="python",
            filename="data_schema.py",
            source=code
        )

        logger.info("a3c_schema_success", code_length=len(code))

    except Exception as e:
        logger.error("a3c_schema_error", error=str(e))

    return state
