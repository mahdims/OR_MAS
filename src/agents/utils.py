# modelpack/agents/utils.py
import ast
import tempfile
import importlib.util
import sys
import os
import re
from typing import Any, Dict, List, Mapping
import pyomo.environ as pyo
import structlog

logger = structlog.get_logger(__name__)


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
