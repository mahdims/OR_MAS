# modelpack/agents/screen_data.py
import structlog
import traceback
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from ..schemas import ModelPack, Feedback, TestInstance
from .utils import load_modules_with_shared_namespace, resolve_solver

logger = structlog.get_logger(__name__)


async def screen_data(state: ModelPack) -> ModelPack:
    """Screen generated data and model feasibility."""

    logger.info("screen_data_start", model_id=state.id)

    # Check retry limit
    MAX_RETRIES = 2
    retry_key = "screen_data_to_build_model"
    retry_count = state.tests.get("retry_counts", {}).get(retry_key, 0)
    existing_feedback = state.tests.get("last_feedback")
    state.tests["instances"] = [
        instance
        for instance in state.tests.get("instances", [])
        if not str(getattr(instance, "id", "")).startswith("screen_")
    ]

    if not all([state.code.model_builder, state.code.datagen]):
        if getattr(existing_feedback, "target_agent", None) == "build_model":
            logger.warning(
                "screen_data_missing_code_with_feedback",
                issue=getattr(existing_feedback, "issue", None),
                source_agent=getattr(existing_feedback, "source_agent", None),
            )
            state.tests["last_feedback"] = existing_feedback
            return state
        state.tests["last_feedback"] = None
        logger.error("screen_data_missing_code")
        return state

    try:
        state.tests["last_feedback"] = None
        # Load all modules in shared namespace
        namespace = load_modules_with_shared_namespace(state.code)

        DataGen = namespace.get("DataGen")
        ModelBuilder = namespace.get("ModelBuilder")
        create_model_fn = namespace.get("create_model")

        if not DataGen:
            raise ValueError("DataGen not found in namespace")
        if not ModelBuilder and not create_model_fn:
            raise ValueError("ModelBuilder or create_model not found in namespace")

        # Try to build with test instance to catch errors early
        test_data = DataGen(0)

        try:
            # This is where Pyomo errors occur
            if ModelBuilder:
                test_model = ModelBuilder(test_data)
            else:
                if isinstance(test_data, dict):
                    test_kwargs = dict(test_data)
                elif hasattr(test_data, "__dict__"):
                    test_kwargs = vars(test_data)
                else:
                    raise TypeError("DataGen output must be dict-like for create_model mode")
                test_model = create_model_fn(**test_kwargs)
            logger.info("screen_data_model_build_success")

        except Exception as pyomo_error:
            error_str = str(pyomo_error)
            error_trace = traceback.format_exc()

            # Check retry limit
            if retry_count >= MAX_RETRIES:
                logger.error("screen_data_max_retries", retries=retry_count)
                state.tests["last_feedback"] = None
                return state

            # Create specific feedback based on error type
            if "Invalid constraint expression" in error_str or "trivial Boolean" in error_str:
                feedback_issue = "pyomo_build_error"
                fix = "Constraints must return Pyomo expressions, not True/False. Use pyo.Constraint.Skip or pyo.Constraint.Feasible."
            elif "AttributeError" in error_str:
                feedback_issue = "code_build_error"
                fix = "Check runtime data access. ModelBuilder may not match the generated data keys or attributes."
            elif "must be integer" in error_str:
                feedback_issue = "type_mismatch"
                fix = (
                    "Data type mismatch during screen build. Ensure the generated Pyomo model "
                    "accepts the runtime data types emitted by DataGen."
                )
            else:
                feedback_issue = "pyomo_build_error"
                fix = f"ModelBuilder failed: {error_str}\nCheck Pyomo syntax and data access patterns."

            # Create feedback for build_model
            feedback = Feedback(
                source_agent="screen_data",
                target_agent="build_model",
                issue=feedback_issue,
                evidence={
                    "error_type": type(pyomo_error).__name__,
                    "error_message": error_str,
                    "traceback": error_trace[:1500],
                    "retry_attempt": retry_count + 1,
                },
                proposed_fix=fix,
                retry_count=retry_count,
            )

            repair_iterations = state.tests.setdefault("repair_iterations", {})
            repair_iterations["screen_data_to_build_model"] = (
                int(repair_iterations.get("screen_data_to_build_model") or 0) + 1
            )
            state.tests["last_feedback"] = feedback
            state.tests["retry_counts"][retry_key] = retry_count + 1

            logger.warning(
                "screen_data_model_build_failed",
                error_type=type(pyomo_error).__name__,
                retry=retry_count + 1,
            )
            return state

        # If model builds, test feasibility
        solver_name, solver = resolve_solver()
        if not solver:
            return state
        logger.info("screen_data_solver_selected", solver=solver_name)

        infeasible_count = 0

        for seed in range(4):
            try:
                data = DataGen(seed)
                if ModelBuilder:
                    model = ModelBuilder(data)
                else:
                    if isinstance(data, dict):
                        data_kwargs = dict(data)
                    elif hasattr(data, "__dict__"):
                        data_kwargs = vars(data)
                    else:
                        raise TypeError("DataGen output must be dict-like for create_model mode")
                    model = create_model_fn(**data_kwargs)

                results = solver.solve(model, tee=False, timelimit=10)

                feasible = (
                    results.solver.status == SolverStatus.ok
                    and results.solver.termination_condition
                    in [TerminationCondition.optimal, TerminationCondition.feasible]
                )

                # Store instance
                if isinstance(data, dict):
                    data_dict = dict(data)
                elif hasattr(data, "__dict__"):
                    data_dict = vars(data)
                else:
                    data_dict = {}
                instance = TestInstance(
                    id=f"screen_{seed}",
                    data_dict=data_dict,
                    feasible=feasible,
                    solver_status=str(results.solver.termination_condition),
                    objective_value=(
                        pyo.value(model.objective)
                        if feasible and hasattr(model, "objective")
                        else None
                    ),
                )
                state.tests["instances"].append(instance)

                if not feasible:
                    infeasible_count += 1

            except Exception as e:
                logger.warning("screen_data_instance_test_failed", seed=seed, error=str(e))
                infeasible_count += 1

        # Reset retry count on success
        state.tests["retry_counts"][retry_key] = 0

        # Check for data issues
        if infeasible_count > 2:
            if retry_count >= MAX_RETRIES:
                logger.warning("screen_data_max_retries", retries=retry_count)
                state.tests["last_feedback"] = None
            else:
                repair_iterations = state.tests.setdefault("repair_iterations", {})
                repair_iterations["screen_data_to_build_model"] = (
                    int(repair_iterations.get("screen_data_to_build_model") or 0) + 1
                )
                feedback = Feedback(
                    source_agent="screen_data",
                    target_agent="build_model",
                    issue="data_infeasible",
                    evidence={"infeasible_count": infeasible_count, "total_tested": 4},
                    proposed_fix=(
                        "Screen instances are mostly infeasible. Review the generated model's "
                        "assumptions and constraints against the runtime data contract."
                    ),
                    retry_count=retry_count,
                )
                state.tests["last_feedback"] = feedback
                state.tests["retry_counts"][retry_key] = retry_count + 1
        else:
            state.tests["last_feedback"] = None

    except Exception as e:
        logger.error("screen_data_error", error=str(e))

    return state
