# modelpack/agents/agent4_pyomo.py
import ast
import structlog
from ..schemas import ModelPack, CodeBlob
from ..llm import llm_client
from ..prompts import PROMPTS

logger = structlog.get_logger(__name__)


def _validate_create_model_entrypoint(source: str) -> tuple[bool, str]:
    """Validate benchmark-mode code contains exactly one top-level create_model function."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"invalid_python: {exc}"

    fn_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    async_defs = [node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)]
    class_defs = [node for node in tree.body if isinstance(node, ast.ClassDef)]

    if async_defs:
        return False, "top_level_async_functions_not_allowed"
    if class_defs:
        return False, "top_level_classes_not_allowed"
    if len(fn_defs) != 1:
        return False, "must_define_exactly_one_top_level_function"
    if fn_defs[0].name != "create_model":
        return False, "top_level_function_must_be_create_model"

    return True, ""


async def a4_pyomo(state: ModelPack) -> ModelPack:
    """A4 - Pyomo Builder: Generate Pyomo model code."""

    logger.info("a4_pyomo_start", model_id=state.id)

    if not state.components_math or not state.code.data_schema:
        logger.error("a4_missing_prerequisites")
        return state

    try:
        target_interface = str(state.context.get("target_interface") or "").strip()
        benchmark_mode = target_interface == "create_model"

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

        if benchmark_mode:
            user_prompt = f"""Mathematical Specification:
{state.components_math.model_dump_json(indent=2)}

Data Schema:
```python
{state.code.data_schema.source}
```

{feedback_context}

Generate code with exactly one top-level function:
create_model(...) -> pyo.ConcreteModel

Hard requirements:
- no ModelBuilder function
- no file I/O
- no solver calls
- no external calls
- deterministic behavior only
- constraints return expressions, not True/False
"""
            system_prompt = PROMPTS["A4_pyomo_create_model"]["system"]
        else:
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
            system_prompt = PROMPTS["A4_pyomo"]["system"]

        code = llm_client.code_generation_call(
            sys_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            validate=True,
        )

        if benchmark_mode:
            valid, reason = _validate_create_model_entrypoint(code)
            if not valid:
                raise ValueError(f"benchmark_create_model_validation_failed: {reason}")

        state.code.model_builder = CodeBlob(
            language="python",
            filename="create_model.py" if benchmark_mode else "model_builder.py",
            source=code,
        )

        logger.info(
            "a4_pyomo_success",
            code_length=len(code),
            target_interface=target_interface or "default",
        )

    except Exception as e:
        logger.error("a4_pyomo_error", error=str(e))

    return state
