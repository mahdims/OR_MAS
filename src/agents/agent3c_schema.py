# modelpack/agents/agent3c_schema.py
import structlog
from ..schemas import ModelPack, CodeBlob
from ..llm import llm_client
from ..prompts import PROMPTS

logger = structlog.get_logger(__name__)


async def a3c_schema(state: ModelPack) -> ModelPack:
    """A3C - Schema Generator: Generate Data class structure."""

    logger.info("a3c_schema_start", model_id=state.id)

    # In benchmark create_model mode the Data class is unused — A4 reads the
    # DataGenerator contract directly from nl_problem. Skip to save an LLM call.
    target_interface = str(state.context.get("target_interface") or "").strip()
    if target_interface == "create_model":
        logger.info("a3c_schema_skipped", reason="benchmark_create_model_mode")
        return state

    if not state.components_math:
        logger.error("a3c_no_math_components")
        return state

    try:
        user_prompt = f"""Mathematical Components:
{state.components_math.model_dump_json(indent=2)}

Generate a generic Data class using dictionaries for parameters and lists for sets.
Use each component's `maps_to` identifier as the attribute name. Never use math symbols like `D_i`, `x_j`, or `c` as field names.
NO hardcoded attributes for specific instances.
Use tuple keys for multi-indexed parameters."""

        code = llm_client.code_generation_call(
            sys_prompt=PROMPTS["A3C_schema"]["system"],
            user_prompt=user_prompt,
            temperature=0.3,
            validate=True,
        )

        state.code.data_schema = CodeBlob(language="python", filename="data_schema.py", source=code)

        logger.info("a3c_schema_success", code_length=len(code))

    except Exception as e:
        logger.error("a3c_schema_error", error=str(e))

    return state
