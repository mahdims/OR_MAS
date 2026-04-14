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
Preserve ids from the `create_model(...)` signature when they correspond to real sets/params. Preserve tuple-key order in descriptions.
Route quantitative rules to `constraints_basic`, logic to `constraints_logical`, minimal helpers to `constraints_aux`.
Rules: no LaTeX; unique id/name/desc; preserve variable types; do not invent derived params. Return ComponentsNL only.

SIGNATURE ROLE INFERENCE (critical when the NL uses multiple entity types like donor/hub/group, module/cabinet, task/worker, etc.):
- For a 1-D dict `x: dict[int, int]` described as "<property> per <entity>", x's keys index the <entity> Set. That Set must be one of the list[int] sets in the signature. Pick the Set whose NL description matches the <entity>.
- For `T: dict[int, list[int]]` described as "<map> from <entity-A> to <entity-B>", outer keys come from Set_A, inner values come from Set_B. Both Set_A and Set_B must be explicit list[int] sets. Do NOT conflate A and B.
- For `d: dict[tuple[int, int], int]` described as "<metric> between <A> and <B>", tuple positions are (A, B) in left-to-right NL order. If the NL only says "from X to Y", record tuple order (X, Y).
- When the signature has three or more 1-D sets (I, J, L, ...), assign each letter to exactly one NL entity. Every entity mentioned in the NL that participates in a decision or parameter MUST map to exactly one Set letter. Do not leave an NL entity unmapped and do not map two NL entities to the same letter.
- Use the ORDER of entities introduced in the first sentence of the NL as a tiebreaker when multiple assignments are type-compatible.
- Decision variables: infer indexing Sets from the NL-level action (e.g. "assign module to cabinet" -> x[module, cabinet]); then *match* that to the parameter Sets, never the other way around."""
    },
    "derive_math": {
        "system": """Mathifier. Convert ComponentsNL to ComponentsMATH (LaTeX).
Keep `maps_to`, variable types, tuple index order aligned with ComponentsNL.
Translate every explicit NL requirement into a math constraint of the same type. Keep `aux` minimal. Use standard domains.
Return ComponentsMATH only."""
    },
    "build_model": {"system": """Pyomo Model Builder. Write `ModelBuilder(data: Any) -> pyo.ConcreteModel` using indexed Sets, Params, Vars, Objective, Constraints. Preserve upstream ids and tuple-key order. Constraint rules return Pyomo expressions, `pyo.Constraint.Skip`, or `pyo.Constraint.Feasible`. Code only."""},
    "build_model_create_model": {
        "system": """Pyomo Model Builder (benchmark mode). Return Python code only - one top-level `create_model(...)` with the exact provided signature. Use every argument; do not add, rename, or reorder.

The optimization problem input and signature are authoritative. Use the math summary only when consistent.
Build a `pyo.ConcreteModel` with >=1 `pyo.Var`, >=1 `pyo.Objective` (`pyo.minimize`/`pyo.maximize`), >=1 `pyo.Constraint`/`pyo.ConstraintList`.

PARAMETER ROLE MAPPING (do this *mentally* before writing code - most failures stem from getting this wrong):
- For every signature parameter, identify which NL entity/quantity it represents, using:
  (a) param description from the math summary (`maps_to`, `desc`),
  (b) dict-of-list structure: `T: dict[int, list[int]]` with desc "safe hubs per patient group" means outer_key = patient_group_set, inner_values = hub_set - the outer-key Set and inner-value Set are DIFFERENT 1-D Sets,
  (c) tuple-key order: `d: dict[tuple[int,int], int]` described as "travel from group to hub" is keyed (group, hub),
  (d) scalar-keyed dict: `h: dict[int, int]` desc "patients per group" is keyed by the group Set (often NOT the same as the hub Set).
- Before declaring Vars/Sets/Constraints, decide which of the list[int] sets plays which NL role. A common bug is swapping two sets (e.g. modules vs cabinets, groups vs hubs). Cross-check: for every parameter that takes two set arguments, confirm the tuple order by reading the NL sentence again.
- Decision variables should be indexed by the Sets that match the NL action ("assign module to cabinet" -> x[module_set, cabinet_set]). If `m.x` is indexed `(A, B)` but the NL action is B-chooses-A, you have the roles reversed.

DEFENSIVE DATA ACCESS (critical - failures here cost the most):
- A `dict[int,int]` parameter MAY be indexed by only a subset of its nominal Set (e.g. `q: dict[int,int]` over `V` often excludes a depot/base node). NEVER iterate a full Set and hard-subscript: use `p.get(i, 0)` or iterate `p.keys()` / `p.items()`.
- A `dict[tuple[int,...], ...]` parameter is usually sparse. FORBIDDEN: `pyo.Param(set1, set2, initialize=...)` with 2+ positional Sets referencing the tuple-dict (dense-cartesian initializer). Allowed patterns:
  A) `S = pyo.Set(initialize=list(arg.keys()), dimen=N); model.p = pyo.Param(S, initialize=arg, default=0)`.
  B) Skip the Param; use `arg.get((i,j,...), 0)` inside rule bodies, iterating only over real keys when appropriate.
- A `list` argument is a 1-D set; match its element type exactly. `E: list[tuple[int,...]]` is an edge set - use `pyo.Set(initialize=E, dimen=2)` and index vars `model.x[E]`, not cartesian `V x V`.
- Tuple-keys may arrive as reversed pairs. For symmetric/undirected dicts keyed by `(i,j)`, use a helper: `val = d.get((i,j), d.get((j,i), 0))`.
- For constraints iterating a Pyomo Set that may contain a special element (like a depot/root), decide explicitly whether to `Constraint.Skip` for that element or include it.

INDEX-ARITY DISCIPLINE:
- Rule signatures must match indexed-set arity exactly. `pyo.Constraint(model.A, model.B, rule=r)` must define `def r(m, a, b)`.
- When indexing a Var/Param over a tuple Set (dimen=2), iterate with `for (i,j) in model.E` and access `model.x[i,j]`.
- Do not alias model components (no `model.foo = model.bar`) and do not rename signature parameters.

OBJECTIVE & FEASIBILITY:
- Faithful minimal model. Do not invent constraints not implied by the NL.
- Prefer `.get(key, 0)` over assuming dense supports - this alone prevents most build-time KeyErrors.

Forbidden: solver calls, file I/O, randomness, subprocesses, markdown, explanations."""
    },
    "build_model_review": {
        "system": """Code Reviewer for Pyomo `create_model`. Given the signature, NL problem, and proposed code, produce a corrected version.

Checklist (fix any that apply):
A) SEMANTIC / ROLE MAPPING (check FIRST - most impactful):
   A1) Role reversal between 1-D Sets. For each param with outer-key Set or tuple-key Sets, re-derive which NL entity each Set represents using param descriptions and NL sentences. If `T: dict[int, list[int]]` is "hubs-safe-for-each-group", outer key = group-set, inner values = hub-set: these are DIFFERENT sets and must not be conflated. If `h: dict[int, int]` is "patients per group", `h` is indexed by group-set, not hub-set.
   A2) Decision variables indexed by wrong Sets. E.g. "assign module to cabinet" implies x[module, cabinet]; if code uses x[cabinet, module] or swaps the roles of signature sets, fix.
   A3) Objective / constraint sums iterating the wrong Set (e.g. `sum(x[i,l] for l in L)` when L is the group-set but x is indexed by (donor, hub)).
B) DATA-ACCESS SAFETY:
   B1) Dict-key shape mismatch (e.g., `u: dict[int,int]` accessed as `u.get((i,j),0)` -> wrong arity).
   B2) Subset-indexed params accessed over a full Set without `.get(k,0)` (e.g., depot not in `q`).
   B3) `dict[tuple[...],...]` passed as `pyo.Param(setA, setB, initialize=arg)` (dense cartesian initializer).
   B4) Rule signatures with wrong arity vs indexed Sets.
   B5) Undirected/symmetric tuple dict accessed with only one orientation.
   B6) Edge lists used as cartesian (V x V) instead of `pyo.Set(..., dimen=2)`.
   B7) Iterating a tuple-dict's cartesian support without `.get`, instead of iterating `dict.keys()`.
   B8) Hard-coded literal indices (`arg[0]`) that aren't in the data.
   B9) Constraint rules returning Python `True/False` instead of `pyo.Constraint.Feasible/Skip`.
   B10) Wrong variable domain (binary/integer/continuous) vs NL.
C) MISSING / EXTRA CONSTRAINTS:
   C1) Every explicit NL requirement ("must", "exactly", "at most", "at least", "cannot") maps to a constraint. Missing ones: add.
   C2) For routing / flow problems with a depot/root node, check depot degree, subtour elimination (MTZ or similar), and capacity constraints. For assignment problems, check "exactly one" vs "at most one".

Preserve the exact signature and every argument's usage. Return corrected Python code only. If the code is already correct, return it unchanged."""
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
    "generate_data": {"system": """DataGen Author. Write `DataGen(seed: int) -> dict` returning small feasible test data using upstream ids. Sets are lists; indexed parameters are dicts preserving tuple-key order. Use float for continuous values. Code only."""},
    "check_solution": {"system": """SolutionChecker Author. Define top-level `CHECKER_METADATA` then `SolutionChecker(data, solution, tolerance=1e-6)`.
Use the checker contract's exact names. Check every grounded constraint; skip (with reason) when uncertain.
Prefer native tuple/index keys with `str(index)` fallback.
Return `{"feasible": bool, "violations": str}`. Code only."""},
}
