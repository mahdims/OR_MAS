# modelpack/agents/solve_model.py
import structlog
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from ..schemas import ModelPack, TestInstance
from .utils import (
    build_checker_contract,
    load_modules_with_shared_namespace,
    resolve_solver,
    store_solution_entry,
    summarize_solution_dict,
)

logger = structlog.get_logger(__name__)


async def solve_model(state: ModelPack) -> ModelPack:
    """Solve the generated model and extract solutions."""

    logger.info("solve_model_start", model_id=state.id)

    if not all([state.code.model_builder, state.code.datagen]):
        logger.error("solve_model_missing_code")
        return state

    try:
        state.tests["instances"] = [
            instance
            for instance in state.tests.get("instances", [])
            if not str(getattr(instance, "id", "")).startswith("solve_")
        ]

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
            return state
        logger.info("solve_model_solver_selected", solver=solver_name)

        checker_contract_written = False

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
                        if var.is_indexed():
                            solution_dict[var_name] = {}
                            for index in var:
                                store_solution_entry(
                                    solution_dict[var_name],
                                    index,
                                    pyo.value(var[index]),
                                )
                        else:
                            solution_dict[var_name] = pyo.value(var)

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
                if feasible and solution_dict:
                    observed_solution_schema = summarize_solution_dict(solution_dict)
                    observed_solution_schema["instance_id"] = instance.id
                    state.tests["observed_solution_schema"] = observed_solution_schema
                    if not checker_contract_written:
                        state.tests["checker_contract"] = build_checker_contract(
                            components_nl=state.components_nl,
                            model_source=state.code.model_builder.source,
                            data_dict=data_dict,
                            solution_dict=solution_dict,
                        )
                        checker_contract_written = True

                logger.info(
                    "solve_model_instance_solved",
                    seed=seed,
                    feasible=feasible,
                    obj_value=obj_value,
                )

            except Exception as e:
                logger.warning("solve_model_failed", seed=seed, error=str(e))

        if not checker_contract_written:
            state.tests["checker_contract"] = build_checker_contract(
                components_nl=state.components_nl,
                model_source=state.code.model_builder.source,
            )

    except Exception as e:
        logger.error("solve_model_error", error=str(e))

    return state
