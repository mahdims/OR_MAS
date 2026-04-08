# modelpack/prompts.py
import json
import re
from typing import Any


def problem_input_note(problem_text: str) -> str:
    text = str(problem_text or "").lstrip()
    if text.startswith("Structured optimization problem specification:"):
        return (
            "Input mode: structured optimization specification.\n"
            "Use the entities, data parameters, decisions, objective, and business "
            "requirements directly."
        )
    if text.startswith("Ambiguous natural language optimization problem:"):
        return "Input mode: natural-language optimization problem description."
    if text.startswith("Natural language optimization problem:"):
        return "Input mode: natural-language optimization problem description."
    return "Input mode: optimization problem input."


def runtime_data_note() -> str:
    return """Data shape:
- use upstream ids / `maps_to` names
- sets: lists
- scalar parameters: scalars
- indexed parameters: dicts
- tuple-indexed parameters: tuple keys in upstream order
- support dict-style or attribute-style access"""


def llm_problem_text(
    problem_text: str,
    *,
    preserve_data_generator_contract: bool = False,
) -> str:
    text = str(problem_text or "").strip()
    if not text:
        return ""
    if not preserve_data_generator_contract:
        text = re.sub(
            r"\nDataGenerator contract \(source of truth\):\n.*?\n\nRequired create_model signature:\n",
            "\nRequired create_model signature:\n",
            text,
            flags=re.DOTALL,
        )
    text = re.sub(
        r"\n\nHard interface requirements:\n(?:(?:[ \t]*-.*|[ \t]+.*)(?:\n|$))*",
        "",
        text,
    )
    return text.strip()


def _truncate_text(value: str, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."


def compact_feedback_context(
    feedback: Any,
    *,
    max_evidence_chars: int = 400,
    max_traceback_chars: int = 240,
) -> str:
    if feedback is None:
        return ""

    evidence = getattr(feedback, "evidence", None)
    compact_evidence = {}
    if isinstance(evidence, dict):
        for key, value in evidence.items():
            if key == "traceback" and value:
                compact_evidence[key] = _truncate_text(str(value), max_traceback_chars)
                continue
            if key == "mismatches" and isinstance(value, list):
                compact_evidence[key] = value[:2]
                if len(value) > 2:
                    compact_evidence["mismatches_extra"] = len(value) - 2
                continue
            compact_evidence[key] = value
    elif evidence is not None:
        compact_evidence["detail"] = evidence

    lines = [
        f"Feedback from {getattr(feedback, 'source_agent', 'unknown')}:",
        f"- issue: {getattr(feedback, 'issue', 'unknown')}",
    ]
    if compact_evidence:
        evidence_text = _truncate_text(
            json.dumps(compact_evidence, ensure_ascii=False),
            max_evidence_chars,
        )
        lines.append(f"- evidence: {evidence_text}")

    proposed_fix = getattr(feedback, "proposed_fix", None)
    if proposed_fix:
        lines.append(f"- proposed fix: {_truncate_text(str(proposed_fix), 240)}")

    retry_count = getattr(feedback, "retry_count", None)
    if retry_count is not None:
        lines.append(f"- retry count: {retry_count}")

    return "\n".join(lines)


PROMPTS = {
    "specify_problem_contract": {
        "system": """You are the Specifier agent. Normalize only the optimization task into a Problem Contract.

Ignore wrapper/interface text except as interface metadata.
Keep the contract grounded; do not invent hidden business rules or derived quantities.

Extract assumptions, units, objective sense, scope, and deliverables.
Use the provided JSON Schema exactly."""
    },
    "specify_problem_components": {
        "system": """You are the Extractor agent. Extract only the model-essential ComponentsNL.

Use structured specs directly when present. Treat `create_model(...)` signatures as metadata, but preserve contract ids when they correspond to real sets or parameters.

Keep only the sets, parameters, variables, objective, and explicit constraints needed by the model.
Route quantitative requirements to `constraints_basic`, true logic to `constraints_logical`, and minimal helpers to `constraints_aux`.

Rules:
- no mathematical notation
- unique id, name, desc
- prefer contract-grounded ids
- preserve tuple-key order explicitly in descriptions
- preserve variable types
- do not invent derived parameters or helper indicators
- prefer the smallest faithful component set

Return ComponentsNL only."""
    },
    "derive_math": {
        "system": """You are the Mathifier agent. Convert ComponentsNL to ComponentsMATH in LaTeX.

Keep `maps_to`, variable types, and tuple index order exactly aligned with ComponentsNL.
Translate every explicit NL requirement into a math constraint of the same type.
Keep `aux` minimal and do not add new business meaning.

Use standard domains for integer, continuous, and binary variables.
Return ComponentsMATH only."""
    },
    "build_model": {"system": """You are the Pyomo Model Builder.

Write `ModelBuilder(data: Any) -> pyo.ConcreteModel`.

Rules:
- runtime data is passed directly; do not import extra data modules
- use indexed Pyomo Sets, Params, Vars, Objectives, and Constraints
- map variable types correctly: integer, binary, continuous
- constraint rules must return Pyomo expressions, `pyo.Constraint.Skip`, or `pyo.Constraint.Feasible`
- preserve upstream ids and tuple-key order
- keep data access generic when practical so dict-style and attribute-style inputs both work
- return code only"""},
    "build_model_create_model": {
        "system": """You are the Pyomo Model Builder in benchmark mode.

Return Python code only.
Write exactly one top-level `create_model(...)` with the exact provided signature. Use every input and do not add, remove, rename, or reorder arguments.

Use the optimization problem input and contract as the source of truth. Use the math summary only when it agrees with them.
Build a deterministic `pyo.ConcreteModel` with at least one `pyo.Var`, one `pyo.Objective`, and one `pyo.Constraint` or `pyo.ConstraintList`.
Use `pyo.minimize` or `pyo.maximize` only.

Preserve tuple-key order and exact key shapes.

CRITICAL — tuple-keyed dict args (`dict[tuple[int,...],...]`):
- FORBIDDEN: `pyo.Param(set1, set2, ..., initialize=arg_or_fn_using_arg)` with 2+ positional Set args when `initialize` references a tuple-dict arg — this is a dense cartesian initializer that will fail validation.
- ALLOWED pattern A: derive ONE supporting Set from dict keys, then pass the dict directly — `S = pyo.Set(initialize=list(arg.keys()), dimen=N); model.p = pyo.Param(S, initialize=arg, default=0)`.
- ALLOWED pattern B: skip the Param entirely — use `arg.get((i, j, ...), 0)` directly inside constraint rule bodies, iterating only over keys known to exist (`for key in arg` or `for key in arg.keys()`).
- When iterating over cartesian index sets and accessing a tuple-dict, guard each lookup: use `.get(key, 0)` or check `if key in arg` — never assume every combination exists.

Do not alias model components, and make rule signatures match index arity exactly.

No solver calls, file I/O, randomness, subprocesses, markdown, or explanations."""
    },
    "single_agent_create_model": {"system": """You are a single-agent optimization modeler.

Convert the optimization problem input directly into executable Pyomo code.
Treat the provided `create_model(...)` contract as binding.

Be faithful to the stated objective and explicit requirements.
Prefer the smallest correct model over a broad speculative one.
Use only supported inputs; do not invent new ones.

Return exactly one top-level `create_model` with the exact argument list.
Use indexed Pyomo components, preserve tuple-key order, and choose binary/integer/continuous variable types appropriately.
Constraint rules must return Pyomo expressions, `pyo.Constraint.Skip`, or `pyo.Constraint.Feasible`.

No solver calls, file I/O, randomness, subprocesses, markdown, explanations, helper functions, or extra top-level definitions.
Output code only."""},
    "generate_data": {"system": """You are the DataGen Author.

Write `DataGen(seed: int) -> dict`.

Rules:
- generate small feasible test data
- use upstream ids as keys
- sets are lists; indexed parameters are dicts
- preserve tuple-key order
- use float for continuous values
- return code only"""},
    "check_solution": {"system": """You are the SolutionChecker Author.

Write `SolutionChecker(data, solution, tolerance=1e-6)`.

Define top-level `CHECKER_METADATA` before the function.
Use the checker contract as the source of truth for exact data and solution names.
Check every listed constraint you can ground, including logical and auxiliary ones when possible.
Use exact names only, prefer native tuple keys and then the provided string fallback keys, and skip ambiguous checks instead of guessing.

Return `{\"feasible\": bool, \"violations\": str}` and code only."""},
}
