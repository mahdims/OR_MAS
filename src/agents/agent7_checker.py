# modelpack/agents/agent7_checker.py
import json

import structlog
from ..schemas import ModelPack, CodeBlob
from ..llm import llm_client
from ..prompts import PROMPTS, compact_feedback_context, runtime_data_note
from .utils import (
    build_checker_contract,
    build_constraint_catalog,
)

logger = structlog.get_logger(__name__)


async def a7_checker(state: ModelPack) -> ModelPack:
    """A7 - Solution Checker Author: Generate constraint verification code."""

    logger.info("a7_checker_start", model_id=state.id)

    if not state.components_nl:
        logger.error("a7_missing_prerequisites")
        return state

    try:
        constraint_catalog = build_constraint_catalog(state.components_nl)
        constraint_catalog_json = json.dumps(
            constraint_catalog,
            indent=2,
        )
        model_builder_source = (
            state.code.model_builder.source
            if state.code.model_builder and state.code.model_builder.source
            else ""
        )
        checker_contract = state.tests.get("checker_contract")
        if not isinstance(checker_contract, dict):
            checker_contract = build_checker_contract(
                components_nl=state.components_nl,
                model_source=model_builder_source,
            )
            state.tests["checker_contract"] = checker_contract

        feedback_note = ""
        feedback = state.tests.get("last_feedback")
        if feedback and feedback.target_agent == "A7":
            feedback_note = compact_feedback_context(feedback)
        trace_input = {
            "agent": "check_solution",
            "upstream_artifacts": [
                {
                    "label": "constraint_catalog",
                    "source": "build_constraint_catalog(state.components_nl)",
                    "value": constraint_catalog,
                },
                {
                    "label": "checker_contract",
                    "source": "state.tests.checker_contract",
                    "value": checker_contract,
                }
            ],
        }
        if feedback_note:
            trace_input["upstream_artifacts"].append(
                {
                    "label": "targeted_feedback",
                    "source": "state.tests.last_feedback",
                    "value": feedback_note,
                }
            )
            evidence = getattr(feedback, "evidence", None)
            if isinstance(evidence, dict):
                grounded_feedback = {
                    key: evidence.get(key)
                    for key in (
                        "checker_solution_refs",
                        "missing_solution_refs",
                        "solution_keys",
                        "indexed_key_samples",
                        "schema_mismatch_reason",
                        "repeated_violation",
                        "canonical_solution_schema",
                    )
                    if evidence.get(key) is not None
                }
                if grounded_feedback:
                    trace_input["upstream_artifacts"].append(
                        {
                            "label": "grounded_feedback",
                            "source": "state.tests.last_feedback.evidence",
                            "value": grounded_feedback,
                        }
                    )
                model_code_snippet = evidence.get("model_code_snippet")
                if model_code_snippet:
                    trace_input["upstream_artifacts"].append(
                        {
                            "label": "feedback_model_code_snippet",
                            "source": "state.tests.last_feedback.evidence.model_code_snippet",
                            "value": model_code_snippet,
                        }
                    )
        existing_checker_code = (
            state.code.solution_checker.source
            if state.code.solution_checker and state.code.solution_checker.source
            else None
        )
        if existing_checker_code:
            trace_input["upstream_artifacts"].append(
                {
                    "label": "existing_checker_code",
                    "source": "state.code.solution_checker.source",
                    "value": existing_checker_code,
                }
            )

        user_prompt_sections = [
            "Constraint catalog:",
            constraint_catalog_json,
            "Checker contract:",
            json.dumps(checker_contract, indent=2),
            runtime_data_note(),
        ]
        if model_builder_source:
            user_prompt_sections.extend(
                [
                    "Generated model code to ground exact names against:",
                    f"```python\n{model_builder_source}\n```",
                ]
            )
        if feedback_note:
            user_prompt_sections.extend(["Targeted feedback:", feedback_note])
            evidence = getattr(feedback, "evidence", None)
            if isinstance(evidence, dict):
                grounded_feedback = {
                    key: evidence.get(key)
                    for key in (
                        "checker_solution_refs",
                        "missing_solution_refs",
                        "solution_keys",
                        "indexed_key_samples",
                        "schema_mismatch_reason",
                        "repeated_violation",
                        "canonical_solution_schema",
                    )
                    if evidence.get(key) is not None
                }
                if grounded_feedback:
                    user_prompt_sections.extend(
                        [
                            "Grounded repair evidence from A9:",
                            json.dumps(grounded_feedback, indent=2),
                        ]
                    )
                model_code_snippet = evidence.get("model_code_snippet")
                if model_code_snippet:
                    user_prompt_sections.extend(
                        [
                            "Relevant model code snippet from A9:",
                            f"```python\n{model_code_snippet}\n```",
                        ]
                    )
            if existing_checker_code:
                user_prompt_sections.extend(
                    [
                        "Existing checker code to repair:",
                        f"```python\n{existing_checker_code}\n```",
                    ]
                )
        user_prompt_sections.extend(
            [
                "Task:",
                (
                    "Return Python code only.\n"
                    "Define a top-level `CHECKER_METADATA` dict with keys "
                    "`grounded_constraints`, `skipped_constraints`, `solution_names_used`, and `data_names_used`.\n"
                    "Then define `SolutionChecker(data, solution, tolerance=1e-6)`.\n"
                    "Use only exact data and solution names from the checker contract.\n"
                    "Check every listed constraint you can ground from the contract, including logical and auxiliary ones when possible.\n"
                    "If a constraint cannot be grounded confidently, do not guess; add it to `skipped_constraints` with a reason.\n"
                    "Use a helper that reads indexed solution values by trying the native tuple/index key first and then `str(index)`."
                ),
            ]
        )
        user_prompt = "\n\n".join(user_prompt_sections)

        code = llm_client.code_generation_call(
            sys_prompt=PROMPTS["A7_checker"]["system"],
            user_prompt=user_prompt,
            temperature=0.3,
            validate=True,
            trace_input=trace_input,
        )

        state.code.solution_checker = CodeBlob(
            language="python", filename="solution_checker.py", source=code
        )

        logger.info("a7_checker_success", code_length=len(code))

    except Exception as e:
        logger.error("a7_checker_error", error=str(e))

    return state
