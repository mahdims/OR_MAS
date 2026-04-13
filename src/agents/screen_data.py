# modelpack/agents/screen_data.py
import re
import structlog
import traceback
from typing import Any, Dict, List, Optional

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from ..schemas import ModelPack, Feedback, TestInstance
from .utils import load_modules_with_shared_namespace, resolve_solver

logger = structlog.get_logger(__name__)


def _coerce_data_kwargs(data: Any) -> Dict[str, Any]:
    if isinstance(data, dict):
        return dict(data)
    if hasattr(data, "__dict__"):
        return vars(data)
    raise TypeError("DataGen output must be dict-like for create_model mode")


def _preview_items(values: List[Any], *, limit: int = 4) -> List[str]:
    return [repr(value) for value in values[:limit]]


def _shape_summary(value: Any) -> Dict[str, Any]:
    if isinstance(value, list):
        return {
            "kind": "list",
            "len": len(value),
            "sample": _preview_items(list(value)),
        }

    if not isinstance(value, dict):
        return {"kind": type(value).__name__}

    keys = list(value.keys())
    summary: Dict[str, Any] = {"kind": "dict", "len": len(value)}
    if not keys:
        summary["key_shape"] = "empty"
        return summary

    summary["sample_keys"] = _preview_items(keys)
    if all(isinstance(key, tuple) for key in keys):
        summary["key_shape"] = "tuple"
        summary["tuple_arity"] = sorted({len(key) for key in keys})
        return summary

    if all(isinstance(inner, dict) for inner in value.values()):
        summary["key_shape"] = "nested_dict"
        inner_keys: List[Any] = []
        for inner in value.values():
            inner_keys.extend(list(inner.keys())[:2])
            if len(inner_keys) >= 4:
                break
        summary["sample_inner_keys"] = _preview_items(inner_keys)
        return summary

    summary["key_shape"] = "scalar"
    return summary


def _matching_arg_summaries(missing_key: Any, data_kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    for name, value in data_kwargs.items():
        summary = _shape_summary(value)
        if summary.get("kind") != "dict":
            continue
        key_shape = summary.get("key_shape")
        if isinstance(missing_key, tuple):
            if key_shape == "tuple":
                arities = summary.get("tuple_arity") or []
                if len(missing_key) in arities:
                    matches.append({"name": name, **summary})
            continue
        if key_shape in {"scalar", "nested_dict"}:
            matches.append({"name": name, **summary})
    return matches[:4]


def _extract_component_name(error_text: str, error_trace: str) -> Optional[str]:
    for pattern in (
        r"component '([^']+)'",
        r"Constraint '([^']+)'",
        r"Var '([^']+)'",
        r"Param '([^']+)'",
    ):
        match = re.search(pattern, error_text) or re.search(pattern, error_trace)
        if match:
            return match.group(1)
    return None


def _trace_literal_arg_name(
    missing_key: Any,
    data_kwargs: Dict[str, Any],
    error_trace: str,
) -> Optional[str]:
    if not isinstance(missing_key, int):
        return None

    literal = re.escape(str(missing_key))
    for arg_name in data_kwargs:
        pattern = rf"\b{re.escape(arg_name)}\s*\[\s*{literal}\s*\]"
        if re.search(pattern, error_trace):
            return arg_name
    return None


def _select_likely_arg_summary(
    missing_key: Any,
    candidate_args: List[Dict[str, Any]],
    data_kwargs: Dict[str, Any],
    error_trace: str,
) -> Optional[Dict[str, Any]]:
    trace_arg_name = _trace_literal_arg_name(missing_key, data_kwargs, error_trace)
    if trace_arg_name:
        for summary in candidate_args:
            if summary["name"] == trace_arg_name:
                return summary

    if len(candidate_args) == 1:
        return candidate_args[0]

    tuple_matches = [
        summary for summary in candidate_args if summary.get("key_shape") == "tuple"
    ]
    if len(tuple_matches) == 1:
        return tuple_matches[0]

    nested_matches = [
        summary for summary in candidate_args if summary.get("key_shape") == "nested_dict"
    ]
    if len(nested_matches) == 1:
        return nested_matches[0]

    return None


def _compact_arg_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    compact = {
        "name": summary["name"],
        "key_shape": summary.get("key_shape"),
    }
    if "sample_keys" in summary:
        compact["sample_keys"] = summary["sample_keys"]
    if "sample_inner_keys" in summary:
        compact["sample_inner_keys"] = summary["sample_inner_keys"]
    if "tuple_arity" in summary:
        compact["tuple_arity"] = summary["tuple_arity"]
    return compact


def _keyerror_feedback(
    pyomo_error: Exception,
    error_str: str,
    error_trace: str,
    test_kwargs: Dict[str, Any],
) -> tuple[str, Dict[str, Any]]:
    missing_key = pyomo_error.args[0] if getattr(pyomo_error, "args", ()) else None
    component_name = _extract_component_name(error_str, error_trace)
    candidate_args = _matching_arg_summaries(missing_key, test_kwargs)
    likely_arg = _select_likely_arg_summary(
        missing_key,
        candidate_args,
        test_kwargs,
        error_trace,
    )

    evidence: Dict[str, Any] = {
        "missing_key": repr(missing_key),
        "missing_key_type": type(missing_key).__name__,
    }
    if component_name:
        evidence["component_name"] = component_name
    if likely_arg:
        evidence["likely_arg"] = _compact_arg_summary(likely_arg)
    elif candidate_args:
        evidence["candidate_args"] = [
            _compact_arg_summary(summary) for summary in candidate_args[:3]
        ]

    if likely_arg and likely_arg.get("key_shape") == "tuple":
        fix = (
            f"Likely culprit: {likely_arg['name']} is a tuple-keyed dict with runtime keys "
            f"{likely_arg.get('sample_keys', [])}. Build a support set from "
            f"{likely_arg['name']}.keys() and index variables or constraints over that support "
            "instead of a full cartesian product."
        )
    elif likely_arg and likely_arg.get("key_shape") == "nested_dict":
        fix = (
            f"Likely culprit: {likely_arg['name']} is a nested dict with outer keys "
            f"{likely_arg.get('sample_keys', [])}. Do not hardcode literal dict keys like "
            f"{likely_arg['name']}[{missing_key!r}]; derive outer keys from the runtime data and "
            "then inspect inner keys from an observed entry."
        )
    elif likely_arg and likely_arg.get("key_shape") == "scalar":
        fix = (
            f"Likely culprit: {likely_arg['name']} is a scalar-keyed dict with runtime keys "
            f"{likely_arg.get('sample_keys', [])}. Do not assume missing literal keys exist; "
            "derive indices from the observed runtime support."
        )
    elif isinstance(missing_key, tuple):
        fix = (
            "A tuple key was requested during model build. Use tuple-keyed inputs only over "
            "their actual runtime support, and do not reshape scalar-keyed dict inputs into "
            "tuple-indexed Params or constraints."
        )
    else:
        fix = (
            "A scalar dict key was requested during model build. Do not hardcode literal dict "
            "keys like a[0]; derive keys from the runtime data and use the observed support."
        )

    if component_name:
        fix = f"{fix} Failing component: {component_name}."
    return fix, evidence


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

        test_kwargs: Dict[str, Any] = {}
        try:
            # This is where Pyomo errors occur
            if ModelBuilder:
                test_model = ModelBuilder(test_data)
            else:
                test_kwargs = _coerce_data_kwargs(test_data)
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
            evidence_details: Dict[str, Any] = {}
            if isinstance(pyomo_error, KeyError):
                feedback_issue = "code_build_error"
                fix, evidence_details = _keyerror_feedback(
                    pyomo_error,
                    error_str=error_str,
                    error_trace=error_trace,
                    test_kwargs=test_kwargs,
                )
            elif "Invalid constraint expression" in error_str or "trivial Boolean" in error_str:
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
                    **evidence_details,
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
                    data_kwargs = _coerce_data_kwargs(data)
                    model = create_model_fn(**data_kwargs)

                results = solver.solve(model, tee=False, timelimit=60)

                feasible = (
                    results.solver.status == SolverStatus.ok
                    and results.solver.termination_condition
                    in [TerminationCondition.optimal, TerminationCondition.feasible]
                )

                # Store instance
                data_dict = _coerce_data_kwargs(data)
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
        if infeasible_count >= 4:
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
