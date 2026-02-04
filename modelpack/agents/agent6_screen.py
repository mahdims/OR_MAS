# modelpack/agents/agent6_screen.py
import structlog
import os
import traceback
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from ..schemas import ModelPack, Feedback, TestInstance
from .utils import load_modules_with_shared_namespace

logger = structlog.get_logger(__name__)

async def a6_screen(state: ModelPack) -> ModelPack:
    """A6 - Feasibility Screener with comprehensive error handling."""

    logger.info("a6_screen_start", model_id=state.id)

    # Check retry limit
    MAX_RETRIES = 2
    retry_key = "A6_to_A4"
    retry_count = state.tests.get("retry_counts", {}).get(retry_key, 0)

    if not all([state.code.model_builder, state.code.datagen]):
        logger.error("a6_missing_code")
        return state

    try:
        # Load all modules in shared namespace
        namespace = load_modules_with_shared_namespace(state.code)

        DataGen = namespace.get('DataGen')
        ModelBuilder = namespace.get('ModelBuilder')

        if not DataGen or not ModelBuilder:
            raise ValueError("DataGen or ModelBuilder not found in namespace")

        # Try to build with test instance to catch errors early
        extracted = state.extracted_data.dict() if state.extracted_data else {}
        test_data = DataGen(0, extracted_data=extracted)

        try:
            # This is where Pyomo errors occur
            test_model = ModelBuilder(test_data)
            logger.info("a6_model_builds_successfully")

        except Exception as pyomo_error:
            error_str = str(pyomo_error)
            error_trace = traceback.format_exc()

            # Check retry limit
            if retry_count >= MAX_RETRIES:
                logger.error("a6_max_retries", retries=retry_count)
                state.tests["last_feedback"] = None
                return state

            # Create specific feedback based on error type
            if "Invalid constraint expression" in error_str or "trivial Boolean" in error_str:
                feedback_issue = "pyomo_build_error"
                fix = "Constraints must return Pyomo expressions, not True/False. Use pyo.Constraint.Skip or pyo.Constraint.Feasible."
            elif "AttributeError" in error_str:
                feedback_issue = "code_build_error"
                fix = "Check data attribute access. The Data class structure may not match what ModelBuilder expects."
            elif "must be integer" in error_str:
                feedback_issue = "type_mismatch"
                fix = "Data type mismatch. Use float for continuous values, not int."
                # Route to A5 for data type issues
                state.tests["last_feedback"] = Feedback(
                    source_agent="A6",
                    target_agent="A5",
                    issue=feedback_issue,
                    evidence={"error": error_str},
                    proposed_fix=fix,
                    retry_count=retry_count
                )
                state.tests["retry_counts"][retry_key] = retry_count + 1
                return state
            else:
                feedback_issue = "pyomo_build_error"
                fix = f"ModelBuilder failed: {error_str}\nCheck Pyomo syntax and data access patterns."

            # Create feedback for A4
            feedback = Feedback(
                source_agent="A6",
                target_agent="A4",
                issue=feedback_issue,
                evidence={
                    "error_type": type(pyomo_error).__name__,
                    "error_message": error_str,
                    "traceback": error_trace[:1500],
                    "retry_attempt": retry_count + 1
                },
                proposed_fix=fix,
                retry_count=retry_count
            )

            state.tests["last_feedback"] = feedback
            state.tests["retry_counts"][retry_key] = retry_count + 1

            logger.warning("a6_model_build_failed",
                         error_type=type(pyomo_error).__name__,
                         retry=retry_count + 1)
            return state

        # If model builds, test feasibility
        solver_name = os.getenv("SOLVER", "glpk")
        infeasible_count = 0

        for seed in range(4):
            try:
                data = DataGen(seed, extracted_data=extracted)
                model = ModelBuilder(data)

                if not pyo.SolverFactory(solver_name).available():
                    logger.error(f"Solver {solver_name} not available")
                    continue

                solver = pyo.SolverFactory(solver_name)
                results = solver.solve(model, tee=False, timelimit=10)

                feasible = (results.solver.status == SolverStatus.ok and
                           results.solver.termination_condition in
                           [TerminationCondition.optimal, TerminationCondition.feasible])

                # Store instance
                data_dict = vars(data) if hasattr(data, '__dict__') else {}
                instance = TestInstance(
                    id=f"screen_{seed}",
                    data_dict=data_dict,
                    feasible=feasible,
                    solver_status=str(results.solver.termination_condition),
                    objective_value=pyo.value(model.objective) if feasible and hasattr(model, 'objective') else None
                )
                state.tests["instances"].append(instance)

                if not feasible:
                    infeasible_count += 1

            except Exception as e:
                logger.warning("a6_instance_test_failed", seed=seed, error=str(e))
                infeasible_count += 1

        # Reset retry count on success
        state.tests["retry_counts"][retry_key] = 0

        # Check for data issues
        if infeasible_count > 2:
            feedback = Feedback(
                source_agent="A6",
                target_agent="A5",
                issue="data_infeasible",
                evidence={
                    "infeasible_count": infeasible_count,
                    "total_tested": 4
                },
                proposed_fix="Adjust data generation for feasibility"
            )
            state.tests["last_feedback"] = feedback
        else:
            state.tests["last_feedback"] = None

        state.tests["logs"].append({
            "agent": "A6",
            "status": "success",
            "feasible_count": 4 - infeasible_count
        })

    except Exception as e:
        logger.error("a6_screen_error", error=str(e))
        state.tests["logs"].append({"agent": "A6", "error": str(e)})

    return state
