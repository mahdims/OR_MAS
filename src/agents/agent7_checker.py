# modelpack/agents/agent7_checker.py
import json

import structlog
from ..schemas import ModelPack, CodeBlob
from ..llm import llm_client
from ..prompts import PROMPTS, compact_feedback_context, runtime_data_note
from .utils import (
    build_canonical_solution_schema,
    extract_model_component_grounding,
    summarize_solution_dict,
)

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
        basic_constraints_json = json.dumps(
            [{"name": c.name, "desc": c.desc} for c in basic_constraints],
            indent=2,
        )
        model_builder_source = (
            state.code.model_builder.source
            if state.code.model_builder and state.code.model_builder.source
            else None
        )
        model_grounding = (
            extract_model_component_grounding(model_builder_source) if model_builder_source else {}
        )
        canonical_solution_schema = build_canonical_solution_schema(model_grounding)
        if model_grounding:
            state.tests["checker_grounding"] = model_grounding
        if canonical_solution_schema:
            state.tests["canonical_solution_schema"] = canonical_solution_schema

        observed_solution_schema = None
        for instance in state.tests.get("instances", []):
            if getattr(instance, "feasible", False) and getattr(instance, "solution_dict", None):
                observed_solution_schema = summarize_solution_dict(instance.solution_dict)
                observed_solution_schema["instance_id"] = instance.id
                state.tests["observed_solution_schema"] = observed_solution_schema
                break

        feedback_note = ""
        feedback = state.tests.get("last_feedback")
        if feedback and feedback.target_agent == "A7":
            feedback_note = compact_feedback_context(feedback)
        trace_input = {
            "agent": "A7_checker",
            "upstream_artifacts": [
                {
                    "label": "basic_constraints",
                    "source": "state.components_nl.constraints_basic",
                    "value": basic_constraints_json,
                },
            ],
        }
        if canonical_solution_schema:
            trace_input["upstream_artifacts"].append(
                {
                    "label": "canonical_solution_schema",
                    "source": "build_canonical_solution_schema(model_builder_source)",
                    "value": canonical_solution_schema,
                }
            )
        if model_grounding:
            trace_input["upstream_artifacts"].append(
                {
                    "label": "model_grounding",
                    "source": "extract_model_component_grounding(model_builder_source)",
                    "value": model_grounding,
                }
            )
        if observed_solution_schema:
            trace_input["upstream_artifacts"].append(
                {
                    "label": "observed_solution_schema",
                    "source": "state.tests.instances[*].solution_dict",
                    "value": observed_solution_schema,
                }
            )
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
            "Basic constraints:",
            basic_constraints_json,
            "Exact canonical solution schema from the generated model:",
            json.dumps(canonical_solution_schema, indent=2),
            runtime_data_note(),
        ]
        if model_grounding:
            user_prompt_sections.extend(
                [
                    "Exact model grounding artifact from the generated model code:",
                    json.dumps(model_grounding, indent=2),
                ]
            )
        if model_builder_source:
            user_prompt_sections.extend(
                [
                    "Generated model code to ground exact names against:",
                    f"```python\n{model_builder_source}\n```",
                ]
            )
        if observed_solution_schema:
            user_prompt_sections.extend(
                [
                    "Observed solution_dict schema from solved instances:",
                    json.dumps(observed_solution_schema, indent=2),
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
                    "Return `SolutionChecker(data, solution, tolerance=1e-6)`.\n"
                    "Check only the listed constraints.\n"
                    "Use only exact solution variable names shown in the canonical schema or observed solution_dict schema.\n"
                    "Do not invent aliases, renamed variables, or paraphrased field names.\n"
                    "When reading indexed decision values, try the raw tuple/native index first and then the shown string fallback.\n"
                    "If a listed constraint cannot be grounded confidently to the provided exact names, skip it instead of guessing."
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
