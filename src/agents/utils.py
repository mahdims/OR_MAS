# modelpack/agents/utils.py
import tempfile
import importlib.util
import sys
import os
from typing import Any, Dict
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
