# modelpack/agents/agent4_pyomo.py
import structlog
from ..schemas import ModelPack, CodeBlob, Feedback
from ..llm import llm_client
from ..prompts import PROMPTS

logger = structlog.get_logger(__name__)

async def a4_pyomo(state: ModelPack) -> ModelPack:
    """A4 - Pyomo Builder: Generate Pyomo model code."""

    logger.info("a4_pyomo_start", model_id=state.id)

    if not state.components_math or not state.code.data_schema:
        logger.error("a4_missing_prerequisites")
        return state

    try:
        # Check for feedback
        feedback_context = ""
        feedback = state.tests.get("last_feedback")
        if feedback and feedback.target_agent == "A4":
            feedback_context = f"""
FEEDBACK FROM {feedback.source_agent}:
Issue: {feedback.issue}
Evidence: {feedback.evidence}
Proposed Fix: {feedback.proposed_fix}

Please address this feedback in your implementation.
"""

        user_prompt = f"""Mathematical Specification:
{state.components_math.model_dump_json(indent=2)}

Data Schema:
```python
{state.code.data_schema.source}
```

{feedback_context}

Generate ModelBuilder(data: Any) -> pyo.ConcreteModel function.
DO NOT import Data class.
Use correct Pyomo domains based on variable types.
Constraints must return expressions, NOT True/False."""

        code = llm_client.code_generation_call(
            sys_prompt=PROMPTS["A4_pyomo"]["system"],
            user_prompt=user_prompt,
            temperature=0.3,
            validate=True
        )

        state.code.model_builder = CodeBlob(
            language="python",
            filename="model_builder.py",
            source=code
        )

        logger.info("a4_pyomo_success", code_length=len(code))

    except Exception as e:
        logger.error("a4_pyomo_error", error=str(e))

    return state
