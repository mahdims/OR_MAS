# modelpack/agents/utils.py
import ast
import copy
import tempfile
import importlib.util
import sys
import os
import re
import math
from typing import Any, Dict, List, Mapping
import pyomo.environ as pyo
import structlog

logger = structlog.get_logger(__name__)

DEFAULT_GENERATION_MODE = "repair_once"
SUPPORTED_GENERATION_MODES: tuple[str, ...] = ("single_pass", "repair_once")


def normalize_generation_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower()
    if normalized in SUPPORTED_GENERATION_MODES:
        return normalized
    return DEFAULT_GENERATION_MODE


def load_module_from_source(name: str, source: str) -> Any:
    """Load Python module from source code string."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(source)
        spec = importlib.util.spec_from_file_location(name, f.name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def load_modules_with_shared_namespace(code_pack) -> Dict[str, Any]:
    """Load all modules in shared namespace so they can reference each other."""
    import pyomo.environ as pyo
    import numpy as np

    namespace = {
        "__name__": "__main__",
        "pyo": pyo,
        "np": np,
        "List": list,
        "Dict": dict,
        "Optional": type(None),
        "Any": object,
        "Tuple": tuple,
    }

    if code_pack.model_builder:
        # Remove bad import attempts
        import re

        code = code_pack.model_builder.source
        code = re.sub(r"^from\s+[dD]ata\s+import\s+.*", "", code, flags=re.MULTILINE)
        code = re.sub(r"^import\s+[dD]ata.*", "", code, flags=re.MULTILINE)
        exec(code, namespace)

    if code_pack.datagen:
        exec(code_pack.datagen.source, namespace)

    if code_pack.solution_checker:
        exec(code_pack.solution_checker.source, namespace)

    return namespace


def resolve_solver():
    """
    Resolve a usable solver.

    Priority:
    1) SOLVER env var if provided
    2) scip
    3) highs
    """
    explicit = os.getenv("SOLVER")
    if explicit:
        solver = pyo.SolverFactory(explicit)
        if solver.available():
            return explicit, solver
        logger.error(f"Solver {explicit} not available")
        return None, None

    for name in ("scip", "highs"):
        solver = pyo.SolverFactory(name)
        if solver.available():
            return name, solver

    logger.error("No solver available (tried: scip, highs)")
    return None, None


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _literal_component_name(node: ast.AST) -> str:
    text = ast.unparse(node).strip() if node is not None else ""
    if text.startswith("model."):
        return text.split(".", maxsplit=1)[1]
    return text


def extract_model_component_grounding(source: str) -> Dict[str, Any]:
    grounding: Dict[str, Any] = {
        "sets": [],
        "parameters": [],
        "variables": [],
        "constraints": [],
        "objectives": [],
        "parse_error": None,
    }
    if not (source or "").strip():
        return grounding

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        grounding["parse_error"] = str(exc)
        return grounding

    component_types = {
        "pyo.Set": "sets",
        "pyomo.environ.Set": "sets",
        "pyo.Param": "parameters",
        "pyomo.environ.Param": "parameters",
        "pyo.Var": "variables",
        "pyomo.environ.Var": "variables",
        "pyo.Constraint": "constraints",
        "pyomo.environ.Constraint": "constraints",
        "pyo.Objective": "objectives",
        "pyomo.environ.Objective": "objectives",
    }

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Attribute):
            continue

        target = node.targets[0]
        if not isinstance(target.value, ast.Name) or target.value.id != "model":
            continue

        bucket = component_types.get(_call_name(node.value.func))
        if bucket is None:
            continue

        entry: Dict[str, Any] = {
            "name": target.attr,
            "index_sets": [_literal_component_name(arg) for arg in node.value.args],
            "index_arity": len(node.value.args),
        }
        for keyword in node.value.keywords:
            if keyword.arg == "domain":
                entry["domain"] = _literal_component_name(keyword.value)
            elif keyword.arg == "rule":
                entry["rule"] = _literal_component_name(keyword.value)
            elif keyword.arg == "sense":
                entry["sense"] = _literal_component_name(keyword.value)
            elif keyword.arg == "initialize":
                entry["initialize"] = _literal_component_name(keyword.value)

        grounding[bucket].append(entry)

    return grounding


def build_canonical_solution_schema(grounding: Mapping[str, Any]) -> Dict[str, Any]:
    schema: Dict[str, Any] = {}
    for variable in grounding.get("variables", []) or []:
        if not isinstance(variable, Mapping):
            continue
        name = str(variable.get("name") or "").strip()
        if not name:
            continue
        index_sets = [str(item) for item in variable.get("index_sets") or []]
        index_arity = int(variable.get("index_arity") or 0)
        schema[name] = {
            "domain": variable.get("domain"),
            "index_sets": index_sets,
            "index_arity": index_arity,
            "solution_container": "dict" if index_arity else "scalar",
            "allowed_key_forms": (
                ["native_index", "str(native_index)"] if index_arity else ["scalar_value"]
            ),
        }
    return schema


def store_solution_entry(container: Dict[Any, Any], index: Any, value: Any) -> None:
    container[index] = value
    text_index = str(index)
    if text_index != index:
        container[text_index] = value


def summarize_solution_dict(
    solution_dict: Mapping[str, Any],
    *,
    max_index_samples: int = 3,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "solution_keys": [],
        "indexed_key_samples": {},
        "variables": {},
    }
    if not isinstance(solution_dict, Mapping):
        return summary

    solution_keys = [
        str(key)
        for key in solution_dict.keys()
        if isinstance(key, str) and not key.startswith("__")
    ]
    summary["solution_keys"] = solution_keys

    for var_name in solution_keys:
        value = solution_dict.get(var_name)
        variable_summary: Dict[str, Any] = {
            "container_type": type(value).__name__,
        }
        if isinstance(value, Mapping):
            native_samples: List[str] = []
            string_samples: List[str] = []
            has_tuple_keys = False
            has_stringified_tuple_keys = False
            for key in value.keys():
                if isinstance(key, str):
                    string_samples.append(key)
                    if key.startswith("(") and key.endswith(")"):
                        has_stringified_tuple_keys = True
                else:
                    native_samples.append(repr(key))
                    if isinstance(key, tuple):
                        has_tuple_keys = True

            variable_summary.update(
                {
                    "sample_native_keys": native_samples[:max_index_samples],
                    "sample_string_keys": string_samples[:max_index_samples],
                    "has_tuple_keys": has_tuple_keys,
                    "has_stringified_tuple_keys": has_stringified_tuple_keys,
                }
            )
            summary["indexed_key_samples"][var_name] = {
                "native": native_samples[:max_index_samples],
                "string": string_samples[:max_index_samples],
            }
        else:
            variable_summary["sample_value"] = value

        summary["variables"][var_name] = variable_summary

    return summary


def summarize_data_dict(
    data_dict: Mapping[str, Any],
    *,
    max_index_samples: int = 3,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "data_keys": [],
        "key_types": {},
        "indexed_key_samples": {},
    }
    if not isinstance(data_dict, Mapping):
        return summary

    for key, value in data_dict.items():
        key_name = str(key)
        summary["data_keys"].append(key_name)
        summary["key_types"][key_name] = type(value).__name__
        if isinstance(value, Mapping):
            samples = [repr(item) for item in list(value.keys())[:max_index_samples]]
            summary["indexed_key_samples"][key_name] = samples

    return summary


def build_constraint_catalog(components_nl: Any) -> List[Dict[str, Any]]:
    if components_nl is None:
        return []

    catalog: List[Dict[str, Any]] = []
    for bucket_name, constraint_type in (
        ("constraints_basic", "basic"),
        ("constraints_logical", "logical"),
        ("constraints_aux", "aux"),
    ):
        for item in getattr(components_nl, bucket_name, []) or []:
            catalog.append(
                {
                    "id": getattr(item, "id", None),
                    "name": getattr(item, "name", None),
                    "desc": getattr(item, "desc", None),
                    "type": constraint_type,
                }
            )
    return catalog


def build_checker_contract(
    *,
    components_nl: Any,
    model_source: str,
    data_dict: Mapping[str, Any] | None = None,
    solution_dict: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    model_grounding = extract_model_component_grounding(model_source)
    return {
        "constraint_catalog": build_constraint_catalog(components_nl),
        "model_grounding": model_grounding,
        "canonical_solution_schema": build_canonical_solution_schema(model_grounding),
        "observed_solution_schema": summarize_solution_dict(solution_dict or {}),
        "observed_data_schema": summarize_data_dict(data_dict or {}),
    }


def extract_checker_solution_refs(checker_source: str) -> List[str]:
    if not (checker_source or "").strip():
        return []

    patterns = [
        r"(?:get|getattr|get_param)\(\s*solution\s*,\s*['\"]([^'\"]+)['\"]",
        r"solution\.get\(\s*['\"]([^'\"]+)['\"]",
        r"solution\[['\"]([^'\"]+)['\"]\]",
    ]
    seen: List[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, checker_source):
            if match not in seen:
                seen.append(match)
    return seen


def extract_checker_data_refs(checker_source: str) -> List[str]:
    if not (checker_source or "").strip():
        return []

    patterns = [
        r"(?:get|getattr|get_param)\(\s*data\s*,\s*['\"]([^'\"]+)['\"]",
        r"data\.get\(\s*['\"]([^'\"]+)['\"]",
        r"data\[['\"]([^'\"]+)['\"]\]",
    ]
    seen: List[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, checker_source):
            if match not in seen:
                seen.append(match)
    return seen


def normalize_checker_metadata(metadata: Any) -> Dict[str, Any]:
    if not isinstance(metadata, Mapping):
        return {
            "grounded_constraints": [],
            "skipped_constraints": [],
            "solution_names_used": [],
            "data_names_used": [],
        }

    normalized = {
        "grounded_constraints": list(metadata.get("grounded_constraints") or []),
        "skipped_constraints": list(metadata.get("skipped_constraints") or []),
        "solution_names_used": list(metadata.get("solution_names_used") or []),
        "data_names_used": list(metadata.get("data_names_used") or []),
    }
    return normalized


def _tokenize_schema_text(text: Any) -> List[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "as",
        "be",
        "but",
        "by",
        "can",
        "each",
        "for",
        "if",
        "in",
        "is",
        "it",
        "its",
        "must",
        "of",
        "on",
        "or",
        "the",
        "to",
    }
    return [
        token
        for token in re.findall(r"[A-Za-z0-9]+", str(text or "").lower())
        if token and token not in stopwords
    ]


def violation_matches_model_grounding(
    violation: str,
    grounding: Mapping[str, Any],
) -> bool:
    violation_tokens = set(_tokenize_schema_text(violation))
    if not violation_tokens:
        return False

    for bucket in ("constraints", "variables"):
        for item in grounding.get(bucket, []) or []:
            if not isinstance(item, Mapping):
                continue
            name_tokens = set(_tokenize_schema_text(item.get("name")))
            if not name_tokens:
                continue
            overlap = violation_tokens & name_tokens
            if len(overlap) >= min(3, len(name_tokens)):
                return True
    return False


def match_violation_to_constraints(
    violation: str,
    constraint_catalog: List[Mapping[str, Any]],
) -> List[str]:
    violation_tokens = set(_tokenize_schema_text(violation))
    matches: List[str] = []
    if not violation_tokens:
        return matches

    for item in constraint_catalog:
        name = str(item.get("name") or "").strip()
        desc = str(item.get("desc") or "").strip()
        tokens = set(_tokenize_schema_text(name)) | set(_tokenize_schema_text(desc))
        if not tokens:
            continue
        overlap = violation_tokens & tokens
        if len(overlap) >= min(3, max(1, len(tokens) // 3)):
            matches.append(name or str(item.get("id") or "constraint"))
    return matches


def lookup_solution_value(container: Any, key: Any) -> tuple[bool, Any]:
    if isinstance(container, Mapping):
        if key in container:
            return True, container[key]
        text_key = str(key)
        if text_key in container:
            return True, container[text_key]
    else:
        try:
            return True, container[key]
        except Exception:
            pass
    return False, None


def build_model_from_instance(namespace: Mapping[str, Any], data_dict: Mapping[str, Any]) -> Any:
    ModelBuilder = namespace.get("ModelBuilder")
    create_model_fn = namespace.get("create_model")

    if ModelBuilder:
        try:
            return ModelBuilder(data_dict)
        except Exception:
            data_obj = type("RuntimeData", (), dict(data_dict))()
            return ModelBuilder(data_obj)
    if create_model_fn:
        return create_model_fn(**dict(data_dict))
    raise ValueError("ModelBuilder or create_model not found")


def assign_solution_to_model(
    model: Any,
    solution_dict: Mapping[str, Any],
) -> List[str]:
    issues: List[str] = []
    for var in model.component_objects(pyo.Var, active=True):
        var_name = var.name
        if var_name not in solution_dict:
            issues.append(f"missing_solution_variable:{var_name}")
            continue

        container = solution_dict[var_name]
        if var.is_indexed():
            for index in var:
                found, value = lookup_solution_value(container, index)
                if not found:
                    issues.append(f"missing_solution_index:{var_name}[{index!r}]")
                    continue
                var[index].set_value(value, skip_validation=True)
        else:
            var.set_value(container, skip_validation=True)
    return issues


def _domain_name(var_data: Any) -> str:
    domain = getattr(var_data, "domain", None)
    name = getattr(domain, "name", None)
    return str(name or domain or "")


def evaluate_model_deterministically(
    model: Any,
    *,
    tolerance: float = 1e-6,
    max_violations: int = 8,
) -> Dict[str, Any]:
    violations: List[str] = []
    checked_constraints = 0

    for var_data in model.component_data_objects(pyo.Var, active=True):
        value = pyo.value(var_data, exception=False)
        if value is None:
            violations.append(f"uninitialized_variable:{var_data.name}")
            if len(violations) >= max_violations:
                break
            continue
        lb = pyo.value(var_data.lb, exception=False) if var_data.lb is not None else None
        ub = pyo.value(var_data.ub, exception=False) if var_data.ub is not None else None
        if lb is not None and value < lb - tolerance:
            violations.append(f"lower_bound_violation:{var_data.name}")
        if ub is not None and value > ub + tolerance:
            violations.append(f"upper_bound_violation:{var_data.name}")

        domain_name = _domain_name(var_data)
        if "Binary" in domain_name and min(abs(value), abs(value - 1)) > tolerance:
            violations.append(f"binary_domain_violation:{var_data.name}")
        elif "Integer" in domain_name and abs(value - round(value)) > tolerance:
            violations.append(f"integer_domain_violation:{var_data.name}")

        if len(violations) >= max_violations:
            break

    if len(violations) < max_violations:
        for constraint in model.component_data_objects(pyo.Constraint, active=True):
            checked_constraints += 1
            body_value = pyo.value(constraint.body, exception=False)
            lower_value = pyo.value(constraint.lower, exception=False) if constraint.has_lb() else None
            upper_value = pyo.value(constraint.upper, exception=False) if constraint.has_ub() else None

            if body_value is None:
                violations.append(f"unevaluable_constraint:{constraint.name}")
            elif lower_value is not None and body_value < lower_value - tolerance:
                violations.append(f"constraint_lb_violation:{constraint.name}")
            elif upper_value is not None and body_value > upper_value + tolerance:
                violations.append(f"constraint_ub_violation:{constraint.name}")

            if len(violations) >= max_violations:
                break

    return {
        "feasible": len(violations) == 0,
        "violations": violations,
        "checked_constraints": checked_constraints,
    }


def _iter_solution_locations(solution_dict: Mapping[str, Any]) -> List[tuple[str, Any, Any]]:
    locations: List[tuple[str, Any, Any]] = []
    for var_name, container in solution_dict.items():
        if isinstance(container, Mapping):
            native_keys = [key for key in container.keys() if not isinstance(key, str)]
            keys = native_keys or list(container.keys())
            seen = set()
            for key in keys:
                key_marker = repr(key)
                if key_marker in seen:
                    continue
                seen.add(key_marker)
                locations.append((str(var_name), key, container[key]))
        else:
            locations.append((str(var_name), None, container))
    return locations


def _mutation_values(current_value: Any, domain: str) -> List[Any]:
    try:
        numeric = float(current_value)
    except (TypeError, ValueError):
        return []

    if "Binary" in str(domain):
        return [0.0 if numeric >= 0.5 else 1.0]
    if "Integer" in str(domain):
        candidates = [math.floor(numeric) + 1, math.ceil(numeric) - 1, 0]
    else:
        candidates = [numeric + 1.0, numeric - 1.0, 0.0]

    mutations: List[Any] = []
    for candidate in candidates:
        if abs(float(candidate) - numeric) > 1e-9 and candidate not in mutations:
            mutations.append(candidate)
    return mutations


def find_infeasible_solution_mutations(
    namespace: Mapping[str, Any],
    data_dict: Mapping[str, Any],
    solution_dict: Mapping[str, Any],
    canonical_solution_schema: Mapping[str, Any],
    *,
    tolerance: float = 1e-6,
    max_examples: int = 2,
    max_locations: int = 12,
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    locations = _iter_solution_locations(solution_dict)[:max_locations]

    for var_name, key, current_value in locations:
        schema_entry = canonical_solution_schema.get(var_name, {})
        domain = str(schema_entry.get("domain") or "")
        for candidate_value in _mutation_values(current_value, domain):
            mutated = copy.deepcopy(solution_dict)
            if key is None:
                mutated[var_name] = candidate_value
            else:
                mutated[var_name][key] = candidate_value
                text_key = str(key)
                if isinstance(mutated[var_name], Mapping) and text_key in mutated[var_name]:
                    mutated[var_name][text_key] = candidate_value

            try:
                model = build_model_from_instance(namespace, data_dict)
                assign_solution_to_model(model, mutated)
                deterministic = evaluate_model_deterministically(model, tolerance=tolerance)
            except Exception:
                continue

            if deterministic["feasible"]:
                continue

            examples.append(
                {
                    "solution": mutated,
                    "mutation": {
                        "variable": var_name,
                        "key": repr(key) if key is not None else None,
                        "old_value": current_value,
                        "new_value": candidate_value,
                    },
                    "deterministic_violations": deterministic["violations"],
                }
            )
            if len(examples) >= max_examples:
                return examples

    return examples
