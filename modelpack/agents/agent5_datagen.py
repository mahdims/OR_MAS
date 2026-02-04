# modelpack/agents/agent5_datagen.py
import structlog
from ..schemas import ModelPack, CodeBlob, Feedback
from ..llm import llm_client
from ..prompts import PROMPTS

logger = structlog.get_logger(__name__)

async def a5_datagen(state: ModelPack) -> ModelPack:
    """A5 - DataGen Author: Generate data generation code."""

    logger.info("a5_datagen_start", model_id=state.id)

    if not state.code.data_schema:
        logger.error("a5_no_schema")
        return state

    try:
        # Check for feedback
        feedback_context = ""
        feedback = state.tests.get("last_feedback")
        if feedback and feedback.target_agent == "A5":
            feedback_context = f"""
FEEDBACK FROM {feedback.source_agent}:
Issue: {feedback.issue}
Evidence: {feedback.evidence}
Proposed Fix: {feedback.proposed_fix}

Please address this feedback in your implementation.
"""

        # Include extracted data if available
        extracted_info = ""
        if state.extracted_data:
            extracted_info = f"""
Extracted Data (use these exact values when available):
{state.extracted_data.model_dump_json(indent=2)}
"""

        user_prompt = f"""Mathematical Specification:
{state.components_math.model_dump_json(indent=2) if state.components_math else 'Not available'}

Data Schema:
```python
{state.code.data_schema.source}
```

{extracted_info}

{feedback_context}

Generate DataGen(seed: int, extracted_data: dict = None) -> Data function.
Priority: Use extracted_data if provided, otherwise generate feasible test data.
Use float for nutritional/continuous values, not int."""

        code = llm_client.code_generation_call(
            sys_prompt=PROMPTS["A5_datagen"]["system"],
            user_prompt=user_prompt,
            temperature=0.3,
            validate=True
        )

        state.code.datagen = CodeBlob(
            language="python",
            filename="datagen.py",
            source=code
        )

        logger.info("a5_datagen_success", code_length=len(code))

    except Exception as e:
        logger.error("a5_datagen_error", error=str(e))

    return state
