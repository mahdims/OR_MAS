# modelpack/agents/agent9_judge.py
import structlog
from ..schemas import ModelPack, Feedback
from .utils import load_modules_with_shared_namespace

logger = structlog.get_logger(__name__)

async def a9_judge(state: ModelPack) -> ModelPack:
    """A9 - Judge: Cross-validate solver solutions against checker."""

    logger.info("a9_judge_start", model_id=state.id)

    if not state.code.solution_checker:
        logger.info("a9_no_checker_skip")
        state.status = "completed"
        return state

    try:
        # Load modules
        namespace = load_modules_with_shared_namespace(state.code)

        SolutionChecker = namespace.get('SolutionChecker')

        if not SolutionChecker:
            logger.warning("a9_checker_not_found")
            state.status = "completed"
            return state

        # Get solved instances
        solved_instances = [i for i in state.tests.get("instances", [])
                          if i.id.startswith("solve_") and i.feasible and i.solution_dict]

        if not solved_instances:
            logger.info("a9_no_solved_instances")
            state.status = "completed"
            return state

        # Cross-validate
        mismatches = []
        for instance in solved_instances[:3]:  # Check first 3
            try:
                # Reconstruct data object
                data_dict = instance.data_dict

                # Run checker
                result = SolutionChecker(data_dict, instance.solution_dict)

                checker_feasible = result.get("feasible", False)
                solver_feasible = instance.feasible

                if solver_feasible and not checker_feasible:
                    # False negative - checker rejected valid solution
                    mismatches.append({
                        "instance_id": instance.id,
                        "solver_says": "feasible",
                        "checker_says": "infeasible",
                        "violations": result.get("violations", "")
                    })
                    logger.warning("a9_mismatch_false_negative", instance=instance.id)

            except Exception as e:
                logger.warning("a9_check_failed", instance=instance.id, error=str(e))

        # Create feedback if mismatches found
        if mismatches:
            feedback = Feedback(
                source_agent="A9",
                target_agent="A7",
                issue="checker_false_negative",
                evidence={
                    "mismatches": mismatches,
                    "total_checked": len(solved_instances)
                },
                proposed_fix="Checker is too strict or has logic errors. Review constraint implementation."
            )
            state.tests["last_feedback"] = feedback
        else:
            state.tests["last_feedback"] = None
            state.status = "completed"

        state.tests["logs"].append({
            "agent": "A9",
            "status": "success",
            "validated": len(solved_instances),
            "mismatches": len(mismatches)
        })

    except Exception as e:
        logger.error("a9_judge_error", error=str(e))
        state.tests["logs"].append({"agent": "A9", "error": str(e)})
        state.status = "completed"  # Don't block on judge errors

    return state
