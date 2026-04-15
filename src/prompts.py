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
            json.dumps(compact_evidence, ensure_ascii=True),
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
        "system": """Specifier. Normalize the optimization task into a Problem Contract.
Treat `create_model(...)` signatures as interface metadata only; do not invent hidden rules or derived quantities.
Extract assumptions, units, objective sense, scope, deliverables. Follow the JSON schema exactly."""
    },
    "specify_problem_components": {
        "system": """Extractor. Extract the minimal ComponentsNL (sets, parameters, variables, objective, explicit constraints).
Preserve ids from the `create_model(...)` signature when they correspond to real sets/params. Preserve tuple-key order.
Route quantitative rules to `constraints_basic`, logic to `constraints_logical`, minimal helpers to `constraints_aux`.
Rules: no LaTeX; unique id/name/desc; preserve variable types; do not invent derived params.

Signature role mapping (do this before extracting):
- Each list[int] arg is a 1-D Set — map it to exactly one NL entity; different entities must map to different Sets.
- dict[int, list[int]] with desc "B per A" has outer key=Set_A, inner values=Set_B (different Sets).
- dict[tuple[int,int], ...] tuple order is (A,B) from the NL phrasing (e.g. "from A to B").
- Decision vars are indexed by the Sets implied by the NL action (e.g. "assign module to cabinet" -> x[module, cabinet]).

Return ComponentsNL only."""
    },
    "derive_math": {
        "system": """Mathifier. Convert ComponentsNL to ComponentsMATH (LaTeX).
Keep `maps_to`, variable types, tuple index order aligned with ComponentsNL.
Translate every explicit NL requirement into a math constraint of the same type. Keep `aux` minimal. Use standard domains.
Return ComponentsMATH only."""
    },
    "build_model": {"system": """Pyomo Model Builder. Write `ModelBuilder(data: Any) -> pyo.ConcreteModel` using indexed Sets, Params, Vars, Objective, Constraints. Preserve upstream ids and tuple-key order. Constraint rules return Pyomo expressions, `pyo.Constraint.Skip`, or `pyo.Constraint.Feasible`. Code only."""},
    "build_model_create_model": {
        "system": """Pyomo Model Builder (benchmark). Return Python code only: one top-level `create_model(...)` with the exact provided signature. Use every argument; do not rename or reorder.

Problem input and signature are authoritative; math summary is advisory. Build a `pyo.ConcreteModel` with >=1 Var, >=1 Objective (`pyo.minimize`/`pyo.maximize`), >=1 Constraint.

Parameter role mapping (do first):
- Identify which NL entity each list[int] Set represents. Different NL entities must map to different Sets.
- dict[int, list[int]] "B per A" — outer key is Set_A, inner values are Set_B (different Sets).
- dict[tuple[int,int], ...] — tuple order comes from the NL ("from X to Y" -> (X,Y)); re-read the sentence.
- dict[int, int] "x per A" — keyed by Set_A (which may differ from other Sets in the signature).
- Decision vars indexed by Sets implied by NL action ("assign M to C" -> x[M, C]); don't reverse.

Defensive data access:
- Use `.get(key, 0)` on dict params; some dicts are sparse over their nominal Set (e.g. depot excluded).
- For `dict[tuple,...]`: build `S=pyo.Set(initialize=list(d.keys()), dimen=N); pyo.Param(S, initialize=d, default=0)`, OR skip the Param and use `d.get((i,j), 0)` inside rules. Never `pyo.Param(setA, setB, initialize=d)` (dense cartesian).
- `list[tuple[int,...]]` is an edge set: `pyo.Set(initialize=E, dimen=2)`, index vars `model.x[E]`, not V×V.
- For symmetric/undirected tuple dicts: `d.get((i,j), d.get((j,i), 0))`.
- Constraint rule arity must match indexed-Set arity. `pyo.Constraint(model.A, model.B, rule=r)` -> `def r(m, a, b)`.
- Rules return Pyomo expressions, `pyo.Constraint.Skip`, or `pyo.Constraint.Feasible` — never Python True/False.
- Match variable domain (binary/integer/continuous) to NL.

Faithful minimal model. Forbidden: solver calls, I/O, randomness, subprocesses, markdown, commentary."""
    },
    "single_agent_create_model": {"system": """Single-agent optimization modeler. Convert the optimization problem directly into executable Pyomo code.
Treat the `create_model(...)` contract as binding. Use every argument; do not rename.

BEFORE writing code, map each signature parameter to an NL entity or quantity:
- list[int] args are 1-D Sets; identify the NL entity each represents (donor / hub / group / module / cabinet ...).
- dict[int,int] args are scalar-keyed params; identify which Set their key comes from.
- dict[int, list[int]] args map one Set to subsets of another - outer key Set and inner value Set are USUALLY DIFFERENT 1-D Sets.
- dict[tuple[int,int], int] args carry (leftSet, rightSet) tuple order - re-read the NL to fix the order.
- Decision variables should be indexed to match the NL action ("assign module to cabinet" -> x[module, cabinet]).

Faithful, minimal model. Indexed Pyomo components, tuple-key order preserved, correct variable domains.
Constraint rules return Pyomo expressions, `pyo.Constraint.Skip`, or `pyo.Constraint.Feasible`.
Use `.get(key, 0)` on dict params; never assume dense cartesian support.
Code only - no solver calls, I/O, randomness, subprocesses, markdown, or helpers."""},
    "build_model_critique": {
        "system": """Pyomo create_model critic. Given the NL problem, signature, and proposed code, return a corrected create_model (or return the code unchanged if correct).

Check only these high-value issues in order; fix in place:
1. Set role reversal: re-read the NL and confirm each list[int] Set maps to the right NL entity. For `dict[int,list[int]]` "B per A" the outer Set is A and inner values are B (different Sets). For `dict[tuple,...]` "from X to Y" the tuple order is (X,Y). Decision vars follow the NL action ("assign M to C" -> x[M,C]).
2. Missing explicit NL requirements ("must", "exactly", "at most", "at least", "cannot") — add any missing constraint.
3. Dense-cartesian Param on a tuple dict: forbid `pyo.Param(setA, setB, initialize=d)` when `d` is `dict[tuple,...]`. Use a keys-Set with `dimen=N`, or skip the Param and use `d.get((i,j),0)` inline.
4. Missing `.get(k, 0)` on sparse scalar-keyed dict params (a depot/root may be absent).
5. Wrong variable domain (binary vs integer vs continuous) vs NL.
6. Rules returning Python True/False (must return Pyomo expressions, Constraint.Skip, or Constraint.Feasible).

Preserve the exact signature; use every argument. Return Python code only."""
    },
    "generate_data": {"system": """DataGen Author. Write `DataGen(seed: int) -> dict` returning small feasible test data using upstream ids. Sets are lists; indexed parameters are dicts preserving tuple-key order. Use float for continuous values. Code only."""},
    "check_solution": {"system": """SolutionChecker Author. Define top-level `CHECKER_METADATA` then `SolutionChecker(data, solution, tolerance=1e-6)`.
Use the checker contract's exact names. Check every grounded constraint; skip (with reason) when uncertain.
Prefer native tuple/index keys with `str(index)` fallback.
Return `{"feasible": bool, "violations": str}`. Code only."""},
}
