# modelpack/agents/agent8_solver.py
import structlog
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from ..schemas import ModelPack, TestInstance
from .utils import load_modules_with_shared_namespace, resolve_solver

logger = structlog.get_logger(__name__)


async def a8_solver(state: ModelPack) -> ModelPack:
    """A8 - Solver: Run optimization and extract solutions."""

    logger.info("a8_solver_start", model_id=state.id)

    if not all([state.code.model_builder, state.code.datagen]):
        logger.error("a8_missing_code")
        return state

    try:
        # Load modules
        namespace = load_modules_with_shared_namespace(state.code)

        DataGen = namespace.get("DataGen")
        ModelBuilder = namespace.get("ModelBuilder")
        create_model_fn = namespace.get("create_model")

        if not DataGen:
            raise ValueError("DataGen not found")
        if not ModelBuilder and not create_model_fn:
            raise ValueError("ModelBuilder or create_model not found")

        solver_name, solver = resolve_solver()
        if not solver:
            state.tests["logs"].append({"agent": "A8", "error": "No solver available"})
            return state
        logger.info("a8_solver_selected", solver=solver_name)

        # Solve multiple instances
        for seed in range(3):
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

                results = solver.solve(model, tee=False, timelimit=30)

                feasible = (
                    results.solver.status == SolverStatus.ok
                    and results.solver.termination_condition
                    in [TerminationCondition.optimal, TerminationCondition.feasible]
                )

                # Extract solution
                solution_dict = {}
                if feasible:
                    for var in model.component_objects(pyo.Var, active=True):
                        var_name = var.name
                        solution_dict[var_name] = {}
                        for index in var:
                            solution_dict[var_name][str(index)] = pyo.value(var[index])

                obj_value = None
                if feasible and hasattr(model, "objective"):
                    obj_value = pyo.value(model.objective)

                # Store instance
                if isinstance(data, dict):
                    data_dict = dict(data)
                elif hasattr(data, "__dict__"):
                    data_dict = vars(data)
                else:
                    data_dict = {}
                instance = TestInstance(
                    id=f"solve_{seed}",
                    data_dict=data_dict,
                    solution_dict=solution_dict if feasible else None,
                    feasible=feasible,
                    solver_status=str(results.solver.termination_condition),
                    objective_value=obj_value,
                )

                state.tests["instances"].append(instance)

                logger.info("a8_instance_solved", seed=seed, feasible=feasible, obj_value=obj_value)

            except Exception as e:
                logger.warning("a8_solve_failed", seed=seed, error=str(e))

        state.tests["logs"].append(
            {
                "agent": "A8",
                "status": "success",
                "instances_solved": len(
                    [i for i in state.tests["instances"] if i.id.startswith("solve_")]
                ),
            }
        )

    except Exception as e:
        logger.error("a8_solver_error", error=str(e))
        state.tests["logs"].append({"agent": "A8", "error": str(e)})

    return state
