# modelpack/agents/agent3b_data_extractor.py
import structlog
from ..schemas import ModelPack, ExtractedData
from ..llm import llm_client
from ..prompts import PROMPTS

logger = structlog.get_logger(__name__)

async def a3b_data_extractor(state: ModelPack) -> ModelPack:
    """A3B - Data Extractor: Extract concrete numerical values from NL problem."""

    logger.info("a3b_data_extractor_start", model_id=state.id)

    nl_problem = state.context.get("nl_problem", "")
    if not nl_problem:
        logger.error("a3b_no_problem")
        return state

    try:
        user_prompt = f"""Natural Language Problem:
{nl_problem}

NL Components (for reference):
{state.components_nl.model_dump_json(indent=2) if state.components_nl else 'Not available'}

Extract all concrete numerical values, set members, and data from the problem text."""

        extracted = llm_client.structured_call(
            sys_prompt=PROMPTS["A3B_data_extractor"]["system"],
            user_prompt=user_prompt,
            pyd_model=ExtractedData,
            temperature=0.4
        )

        state.extracted_data = extracted

        logger.info("a3b_data_extractor_success",
                   sets=len(extracted.sets),
                   params=len(extracted.parameters))

    except Exception as e:
        logger.error("a3b_data_extractor_error", error=str(e))
        # Not critical - continue without extracted data
        state.extracted_data = ExtractedData()

    return state
