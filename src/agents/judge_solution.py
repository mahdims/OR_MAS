# modelpack/agents/judge_solution.py
import structlog

from ..schemas import Feedback, ModelPack
from .utils import (
    assign_solution_to_model,
    build_checker_contract,
    build_model_from_instance,
    evaluate_model_deterministically,
    extract_checker_data_refs,
    extract_checker_solution_refs,
    find_infeasible_solution_mutations,
    load_modules_with_shared_namespace,
    match_violation_to_constraints,
    normalize_checker_metadata,
    summarize_data_dict,
    summarize_solution_dict,
)

logger = structlog.get_logger(__name__)


def _feedback_retry_key(target_agent: str) -> str:
    return (
        "judge_solution_to_build_model"
        if target_agent == "build_model"
        else "judge_solution_to_check_solution"
    )


async def judge_solution(state: ModelPack) -> ModelPack:
    """Judge checker and model consistency deterministically."""

    logger.info("judge_solution_start", model_id=state.id)

    MAX_RETRIES = 2

    if not state.code.solution_checker:
        logger.info("judge_solution_no_checker_skip")
        state.status = "completed"
        return state

    try:
        namespace = load_modules_with_shared_namespace(state.code)
        SolutionChecker = namespace.get("SolutionChecker")
        checker_metadata = normalize_checker_metadata(namespace.get("CHECKER_METADATA"))

        if not SolutionChecker:
            logger.warning("judge_solution_checker_not_found")
            state.status = "completed"
            return state

        checker_contract = state.tests.get("checker_contract")
        if not isinstance(checker_contract, dict):
            checker_contract = build_checker_contract(
                components_nl=state.components_nl,
                model_source=(
                    state.code.model_builder.source
                    if state.code.model_builder and state.code.model_builder.source
                    else ""
                ),
            )
            state.tests["checker_contract"] = checker_contract

        constraint_catalog = list(checker_contract.get("constraint_catalog") or [])
        canonical_solution_schema = dict(checker_contract.get("canonical_solution_schema") or {})

        checker_source = (
            state.code.solution_checker.source
            if state.code.solution_checker and state.code.solution_checker.source
            else ""
        )
        checker_solution_refs = (
            checker_metadata.get("solution_names_used") or extract_checker_solution_refs(checker_source)
        )
        checker_data_refs = (
            checker_metadata.get("data_names_used") or extract_checker_data_refs(checker_source)
        )

        solved_instances = [
            instance
            for instance in state.tests.get("instances", [])
            if str(getattr(instance, "id", "")).startswith("solve_")
            and getattr(instance, "feasible", False)
            and getattr(instance, "solution_dict", None)
        ]

        if not solved_instances:
            logger.info("judge_solution_no_solved_instances")
            state.status = "completed"
            return state

        state.tests["last_feedback"] = None

        positive_mismatches = []
        checker_runtime_failures = []
        deterministic_positive_failures = []
        checker_false_positive_examples = []
        missing_solution_refs = set()
        missing_data_refs = set()
        solution_keys = set()
        data_keys = set()
        indexed_key_samples = {}
        data_key_samples = {}
        reference_instance = None

        for instance in solved_instances[:3]:
            data_summary = summarize_data_dict(instance.data_dict)
            solution_summary = summarize_solution_dict(instance.solution_dict)
            data_keys.update(data_summary.get("data_keys") or [])
            solution_keys.update(solution_summary.get("solution_keys") or [])
            indexed_key_samples.update(solution_summary.get("indexed_key_samples") or {})
            data_key_samples.update(data_summary.get("indexed_key_samples") or {})
            missing_solution_refs.update(
                ref for ref in checker_solution_refs if ref not in solution_summary.get("solution_keys", [])
            )
            missing_data_refs.update(
                ref for ref in checker_data_refs if ref not in data_summary.get("data_keys", [])
            )

            try:
                model = build_model_from_instance(namespace, instance.data_dict)
                assignment_issues = assign_solution_to_model(model, instance.solution_dict)
                deterministic_result = evaluate_model_deterministically(model)
            except Exception as exc:
                deterministic_positive_failures.append(
                    {
                        "instance_id": instance.id,
                        "error": str(exc),
                    }
                )
                continue

            if assignment_issues or not deterministic_result.get("feasible", False):
                deterministic_positive_failures.append(
                    {
                        "instance_id": instance.id,
                        "assignment_issues": assignment_issues,
                        "deterministic_violations": deterministic_result.get("violations", []),
                    }
                )
                continue

            if reference_instance is None:
                reference_instance = instance

            try:
                checker_result = SolutionChecker(instance.data_dict, instance.solution_dict)
            except Exception as exc:
                checker_runtime_failures.append(
                    {
                        "instance_id": instance.id,
                        "error": str(exc),
                    }
                )
                continue

            checker_feasible = bool(checker_result.get("feasible", False))
            violations = str(checker_result.get("violations", "") or "").strip()
            matched_constraints = match_violation_to_constraints(violations, constraint_catalog)

            if not checker_feasible:
                positive_mismatches.append(
                    {
                        "instance_id": instance.id,
                        "checker_says": "infeasible",
                        "violations": violations,
                        "matched_constraints": matched_constraints,
                    }
                )

        negative_examples = []
        if reference_instance is not None:
            negative_examples = find_infeasible_solution_mutations(
                namespace,
                reference_instance.data_dict,
                reference_instance.solution_dict,
                canonical_solution_schema,
            )
            for example in negative_examples:
                try:
                    checker_result = SolutionChecker(
                        reference_instance.data_dict,
                        example["solution"],
                    )
                except Exception as exc:
                    checker_runtime_failures.append(
                        {
                            "instance_id": reference_instance.id,
                            "phase": "negative_example",
                            "error": str(exc),
                            "mutation": example.get("mutation"),
                        }
                    )
                    continue

                if bool(checker_result.get("feasible", False)):
                    checker_false_positive_examples.append(
                        {
                            "instance_id": reference_instance.id,
                            "mutation": example.get("mutation"),
                            "deterministic_violations": example.get("deterministic_violations", []),
                        }
                    )

        validation_report = {
            "positive_mismatches": positive_mismatches,
            "checker_runtime_failures": checker_runtime_failures,
            "deterministic_positive_failures": deterministic_positive_failures,
            "checker_false_positive_examples": checker_false_positive_examples,
            "negative_examples_tested": len(negative_examples),
            "checker_metadata": checker_metadata,
            "checker_solution_refs": checker_solution_refs,
            "checker_data_refs": checker_data_refs,
            "missing_solution_refs": sorted(missing_solution_refs),
            "missing_data_refs": sorted(missing_data_refs),
            "solution_keys": sorted(solution_keys),
            "data_keys": sorted(data_keys),
            "indexed_key_samples": indexed_key_samples,
            "data_key_samples": data_key_samples,
            "canonical_solution_schema": canonical_solution_schema,
        }
        state.tests["checker_validation"] = validation_report

        schema_mismatch_reason = None
        if checker_runtime_failures:
            schema_mismatch_reason = "checker_runtime_failure"
        elif missing_solution_refs or missing_data_refs:
            schema_mismatch_reason = "checker_refs_missing_from_contract"
        elif not (
            checker_metadata.get("grounded_constraints") or checker_metadata.get("skipped_constraints")
        ):
            schema_mismatch_reason = "checker_metadata_missing"

        if schema_mismatch_reason:
            retry_key = _feedback_retry_key("check_solution")
            retry_count = state.tests.get("retry_counts", {}).get(retry_key, 0)
            if retry_count >= MAX_RETRIES:
                logger.warning(
                    "judge_solution_max_retries",
                    retries=retry_count,
                    target_agent="check_solution",
                )
                state.tests["last_feedback"] = None
                state.tests["retry_counts"][retry_key] = 0
                state.status = "completed"
                return state

            repair_iterations = state.tests.setdefault("repair_iterations", {})
            repair_iterations["judge_solution_to_check_solution"] = (
                int(repair_iterations.get("judge_solution_to_check_solution") or 0) + 1
            )
            feedback = Feedback(
                source_agent="judge_solution",
                target_agent="check_solution",
                issue="checker_schema_mismatch",
                evidence={
                    **validation_report,
                    "schema_mismatch_reason": schema_mismatch_reason,
                    "model_code_snippet": state.code.model_builder.source[:4000],
                },
                proposed_fix=(
                    "Checker must use the checker contract exactly and return CHECKER_METADATA. "
                    "Repair the data/solution names and metadata before trying to audit the model."
                ),
                retry_count=retry_count,
            )
            state.tests["last_feedback"] = feedback
            state.tests["retry_counts"][retry_key] = retry_count + 1
            return state

        if checker_false_positive_examples:
            retry_key = _feedback_retry_key("check_solution")
            retry_count = state.tests.get("retry_counts", {}).get(retry_key, 0)
            if retry_count >= MAX_RETRIES:
                logger.warning(
                    "judge_solution_max_retries",
                    retries=retry_count,
                    target_agent="check_solution",
                )
                state.tests["last_feedback"] = None
                state.tests["retry_counts"][retry_key] = 0
                state.status = "completed"
                return state

            repair_iterations = state.tests.setdefault("repair_iterations", {})
            repair_iterations["judge_solution_to_check_solution"] = (
                int(repair_iterations.get("judge_solution_to_check_solution") or 0) + 1
            )
            feedback = Feedback(
                source_agent="judge_solution",
                target_agent="check_solution",
                issue="checker_false_positive",
                evidence={
                    **validation_report,
                    "model_code_snippet": state.code.model_builder.source[:4000],
                },
                proposed_fix=(
                    "Checker missed deterministically infeasible corrupted solutions. "
                    "Repair the checker logic while staying grounded to the checker contract."
                ),
                retry_count=retry_count,
            )
            state.tests["last_feedback"] = feedback
            state.tests["retry_counts"][retry_key] = retry_count + 1
            return state

        if deterministic_positive_failures:
            retry_key = _feedback_retry_key("build_model")
            retry_count = state.tests.get("retry_counts", {}).get(retry_key, 0)
            if retry_count >= MAX_RETRIES:
                logger.warning(
                    "judge_solution_max_retries",
                    retries=retry_count,
                    target_agent="build_model",
                )
                state.tests["last_feedback"] = None
                state.tests["retry_counts"][retry_key] = 0
                state.status = "completed"
                return state

            repair_iterations = state.tests.setdefault("repair_iterations", {})
            repair_iterations["judge_solution_to_build_model"] = (
                int(repair_iterations.get("judge_solution_to_build_model") or 0) + 1
            )
            feedback = Feedback(
                source_agent="judge_solution",
                target_agent="build_model",
                issue="model_constraint_mismatch",
                evidence={
                    **validation_report,
                    "model_code_snippet": state.code.model_builder.source[:4000],
                },
                proposed_fix=(
                    "The extracted solver solution does not satisfy the generated Pyomo model under "
                    "deterministic re-evaluation. Repair the model before relying on the checker."
                ),
                retry_count=retry_count,
            )
            state.tests["last_feedback"] = feedback
            state.tests["retry_counts"][retry_key] = retry_count + 1
            return state

        if positive_mismatches:
            # If the model is deterministically correct (no deterministic_positive_failures)
            # but the checker disagrees, the checker is likely wrong — fix the checker,
            # not the model.
            if not deterministic_positive_failures:
                target = "check_solution"
                retry_key = _feedback_retry_key("check_solution")
                proposed_fix = (
                    "Checker rejects a solver-feasible solution that passes deterministic "
                    "re-evaluation. The model is likely correct; repair the checker constraints "
                    "to match the model's actual variable/constraint semantics."
                )
            else:
                target = "build_model"
                retry_key = _feedback_retry_key("build_model")
                proposed_fix = (
                    "Checker is grounded and rejects a solver-feasible solution, so the generated model "
                    "is likely missing or weakening an NL requirement. Repair the candidate model."
                )
            retry_count = state.tests.get("retry_counts", {}).get(retry_key, 0)
            if retry_count >= MAX_RETRIES:
                logger.warning(
                    "judge_solution_max_retries",
                    retries=retry_count,
                    target_agent=target,
                )
                state.tests["last_feedback"] = None
                state.tests["retry_counts"][retry_key] = 0
                state.status = "completed"
                return state

            repair_iterations = state.tests.setdefault("repair_iterations", {})
            iteration_key = f"judge_solution_to_{target}"
            repair_iterations[iteration_key] = (
                int(repair_iterations.get(iteration_key) or 0) + 1
            )
            feedback = Feedback(
                source_agent="judge_solution",
                target_agent=target,
                issue="model_constraint_mismatch",
                evidence={
                    **validation_report,
                    "model_code_snippet": state.code.model_builder.source[:4000],
                },
                proposed_fix=proposed_fix,
                retry_count=retry_count,
            )
            state.tests["last_feedback"] = feedback
            state.tests["retry_counts"][retry_key] = retry_count + 1
            return state

        state.tests["last_feedback"] = None
        state.tests["retry_counts"][_feedback_retry_key("check_solution")] = 0
        state.tests["retry_counts"][_feedback_retry_key("build_model")] = 0
        state.status = "completed"

    except Exception as exc:
        logger.error("judge_solution_error", error=str(exc))
        state.status = "completed"

    return state
