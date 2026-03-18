# modelpack/agents/agent9_judge.py
import structlog
from ..schemas import ModelPack, Feedback
from .utils import (
    build_canonical_solution_schema,
    extract_checker_solution_refs,
    extract_model_component_grounding,
    load_modules_with_shared_namespace,
    summarize_solution_dict,
    violation_matches_model_grounding,
)

logger = structlog.get_logger(__name__)


async def a9_judge(state: ModelPack) -> ModelPack:
    """A9 - Judge: Cross-validate solver solutions against checker."""

    logger.info("a9_judge_start", model_id=state.id)

    MAX_RETRIES = 2
    retry_key = "A9_to_A7"
    retry_count = state.tests.get("retry_counts", {}).get(retry_key, 0)

    if not state.code.solution_checker:
        logger.info("a9_no_checker_skip")
        state.status = "completed"
        return state

    try:
        # Load modules
        namespace = load_modules_with_shared_namespace(state.code)
        model_source = (
            state.code.model_builder.source
            if state.code.model_builder and state.code.model_builder.source
            else ""
        )
        model_grounding = extract_model_component_grounding(model_source)
        canonical_solution_schema = build_canonical_solution_schema(model_grounding)
        checker_source = (
            state.code.solution_checker.source
            if state.code.solution_checker and state.code.solution_checker.source
            else ""
        )
        checker_solution_refs = extract_checker_solution_refs(checker_source)

        SolutionChecker = namespace.get("SolutionChecker")

        if not SolutionChecker:
            logger.warning("a9_checker_not_found")
            state.status = "completed"
            return state

        # Get solved instances
        solved_instances = [
            i
            for i in state.tests.get("instances", [])
            if i.id.startswith("solve_") and i.feasible and i.solution_dict
        ]

        if not solved_instances:
            logger.info("a9_no_solved_instances")
            state.status = "completed"
            return state

        # Cross-validate
        mismatches = []
        observed_solution_schema = None
        solution_keys = set()
        missing_solution_refs = set()
        violation_counts = {}
        for instance in solved_instances[:3]:  # Check first 3
            try:
                # Reconstruct data object
                data_dict = instance.data_dict
                if instance.solution_dict:
                    solution_summary = summarize_solution_dict(instance.solution_dict)
                    if observed_solution_schema is None:
                        observed_solution_schema = dict(solution_summary)
                        observed_solution_schema["instance_id"] = instance.id
                    instance_solution_keys = solution_summary.get("solution_keys") or []
                    solution_keys.update(instance_solution_keys)
                    missing_solution_refs.update(
                        ref for ref in checker_solution_refs if ref not in instance_solution_keys
                    )

                # Run checker
                result = SolutionChecker(data_dict, instance.solution_dict)

                checker_feasible = result.get("feasible", False)
                solver_feasible = instance.feasible

                if solver_feasible and not checker_feasible:
                    violations = str(result.get("violations", "") or "").strip()
                    if violations:
                        violation_counts[violations] = violation_counts.get(violations, 0) + 1
                    # False negative - checker rejected valid solution
                    mismatches.append(
                        {
                            "instance_id": instance.id,
                            "solver_says": "feasible",
                            "checker_says": "infeasible",
                            "violations": violations,
                        }
                    )
                    logger.warning("a9_mismatch_false_negative", instance=instance.id)

            except Exception as e:
                logger.warning("a9_check_failed", instance=instance.id, error=str(e))

        # Create feedback if mismatches found
        if mismatches:
            repeated_violation = None
            repeated_violation_count = 0
            for violation_text, count in violation_counts.items():
                if count > repeated_violation_count:
                    repeated_violation = violation_text
                    repeated_violation_count = count

            shared_evidence = {
                "mismatches": mismatches,
                "total_checked": len(solved_instances),
                "checker_solution_refs": checker_solution_refs,
                "missing_solution_refs": sorted(missing_solution_refs),
                "solution_keys": sorted(solution_keys),
                "indexed_key_samples": (
                    observed_solution_schema.get("indexed_key_samples", {})
                    if isinstance(observed_solution_schema, dict)
                    else {}
                ),
                "canonical_solution_schema": canonical_solution_schema,
                "model_code_snippet": model_source[:4000] if model_source else "",
            }

            schema_mismatch_reason = None
            if missing_solution_refs:
                schema_mismatch_reason = "checker_refs_missing_from_solution_dict"
            elif (
                repeated_violation
                and repeated_violation_count >= 2
                and violation_matches_model_grounding(repeated_violation, model_grounding)
            ):
                schema_mismatch_reason = "repeated_violation_matches_existing_candidate_constraint"

            if schema_mismatch_reason:
                shared_evidence["schema_mismatch_reason"] = schema_mismatch_reason
                if repeated_violation:
                    shared_evidence["repeated_violation"] = repeated_violation
                feedback = Feedback(
                    source_agent="A9",
                    target_agent="A7",
                    issue="checker_schema_mismatch",
                    evidence=shared_evidence,
                    proposed_fix=(
                        "Checker is not grounded to the candidate model schema. "
                        "Use the exact solution variable names and observed key encoding; stop retrying "
                        "the same checker loop."
                    ),
                    retry_count=retry_count,
                )
                state.tests["last_feedback"] = feedback
                state.tests["retry_counts"][retry_key] = 0
                state.status = "completed"
                return state
            if retry_count >= MAX_RETRIES:
                logger.warning("a9_max_retries", retries=retry_count)
                state.tests["last_feedback"] = None
                state.status = "completed"
                return state
            repair_iterations = state.tests.setdefault("repair_iterations", {})
            repair_iterations["A9_to_A7"] = int(repair_iterations.get("A9_to_A7") or 0) + 1
            feedback = Feedback(
                source_agent="A9",
                target_agent="A7",
                issue="checker_false_negative",
                evidence=shared_evidence,
                proposed_fix=(
                    "Checker is too strict or has logic errors. "
                    "Repair it against the grounded model schema, exact solution keys, and sampled index forms."
                ),
                retry_count=retry_count,
            )
            state.tests["last_feedback"] = feedback
            state.tests["retry_counts"][retry_key] = retry_count + 1
        else:
            state.tests["last_feedback"] = None
            state.tests["retry_counts"][retry_key] = 0
            state.status = "completed"

    except Exception as e:
        logger.error("a9_judge_error", error=str(e))
        state.status = "completed"  # Don't block on judge errors

    return state
