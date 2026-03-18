# modelpack/agents/agent4_pyomo.py
import ast
import re
from typing import List, Optional, Set, Tuple

import structlog

from ..schemas import ModelPack, CodeBlob
from ..llm import llm_client
from ..prompts import PROMPTS, compact_feedback_context, runtime_data_note

logger = structlog.get_logger(__name__)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _attribute_root_name(node: ast.Attribute) -> Optional[str]:
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        current = current.value
    if isinstance(current, ast.Name):
        return current.id
    return None


def _function_arg_names(fn_node: ast.FunctionDef) -> List[str]:
    names: List[str] = []
    names.extend(arg.arg for arg in fn_node.args.posonlyargs)
    names.extend(arg.arg for arg in fn_node.args.args)
    names.extend(arg.arg for arg in fn_node.args.kwonlyargs)
    return names


def _get_model_aliases(fn_node: ast.FunctionDef) -> Set[str]:
    aliases: Set[str] = set()
    for node in ast.walk(fn_node):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if _call_name(node.value.func) not in {"pyo.ConcreteModel", "pyomo.environ.ConcreteModel"}:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                aliases.add(target.id)
    return aliases


def _set_initialize_expr(call_node: ast.Call) -> Optional[ast.AST]:
    for keyword in call_node.keywords:
        if keyword.arg == "initialize":
            return keyword.value
    return None


def _collect_forbidden_call_diagnostics(fn_node: ast.FunctionDef) -> List[str]:
    forbidden_exact = {
        "solve",
        "open",
        "os.system",
        "os.popen",
        "subprocess.run",
        "subprocess.Popen",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "requests.get",
        "requests.post",
        "requests.put",
        "requests.patch",
        "requests.delete",
        "pyo.SolverFactory",
        "pyomo.environ.SolverFactory",
        "SolverFactory",
        "time.time",
        "time.sleep",
        "random.random",
        "random.randint",
        "random.randrange",
        "random.choice",
        "random.uniform",
    }
    forbidden_roots = {"subprocess", "requests", "urllib", "socket", "httpx"}

    diagnostics: List[str] = []
    for node in ast.walk(fn_node):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if not name:
            continue
        root = name.split(".", maxsplit=1)[0]
        if name in forbidden_exact:
            diagnostics.append(f"forbidden_call:{name}")
            continue
        if name.endswith(".solve"):
            diagnostics.append(f"forbidden_call:{name}")
            continue
        if root in forbidden_roots:
            diagnostics.append(f"forbidden_call:{name}")
    return diagnostics


def _collect_set_init_diagnostics(
    fn_node: ast.FunctionDef,
    model_aliases: Set[str],
) -> List[str]:
    if not model_aliases:
        return []

    diagnostics: List[str] = []
    for node in ast.walk(fn_node):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func) not in {"pyo.Set", "pyomo.environ.Set"}:
            continue
        initialize_expr = _set_initialize_expr(node)
        if initialize_expr is None:
            continue
        for subnode in ast.walk(initialize_expr):
            if not isinstance(subnode, ast.Attribute):
                continue
            root_name = _attribute_root_name(subnode)
            if root_name not in model_aliases:
                continue
            diagnostics.append("set_initialize_references_model_component")
            if isinstance(subnode, ast.Attribute) and subnode.attr == "value":
                diagnostics.append("set_initialize_uses_model_component_value")
            break
    return diagnostics


def _validate_create_model_entrypoint(source: str) -> Tuple[bool, List[str]]:
    """Validate benchmark-mode code quality for create_model entrypoint."""
    diagnostics: List[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, [f"invalid_python:{exc}"]

    fn_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    async_defs = [node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)]
    class_defs = [node for node in tree.body if isinstance(node, ast.ClassDef)]

    if async_defs:
        diagnostics.append("top_level_async_functions_not_allowed")
    if class_defs:
        diagnostics.append("top_level_classes_not_allowed")
    if len(fn_defs) != 1:
        diagnostics.append("must_define_exactly_one_top_level_function")
        return False, diagnostics
    if fn_defs[0].name != "create_model":
        diagnostics.append("top_level_function_must_be_create_model")
        return False, diagnostics

    fn_node = fn_defs[0]
    arg_names = _function_arg_names(fn_node)
    if fn_node.args.vararg is not None:
        diagnostics.append("create_model_varargs_not_allowed")
    if fn_node.args.kwarg is not None:
        diagnostics.append("create_model_kwargs_not_allowed")
    if not arg_names:
        diagnostics.append("create_model_args_must_be_non_empty")

    used_arg_names: Set[str] = set()
    arg_name_set = set(arg_names)
    for node in ast.walk(fn_node):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in arg_name_set
        ):
            used_arg_names.add(node.id)
    for arg_name in arg_names:
        if arg_name not in used_arg_names:
            diagnostics.append(f"unused_create_model_arg:{arg_name}")

    objective_count = 0
    constraint_count = 0
    for node in ast.walk(fn_node):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name in {"pyo.Objective", "pyomo.environ.Objective"}:
            objective_count += 1
        if name in {"pyo.Constraint", "pyomo.environ.Constraint"}:
            constraint_count += 1
        if name in {"pyo.ConstraintList", "pyomo.environ.ConstraintList"}:
            constraint_count += 1
    if objective_count < 1:
        diagnostics.append("missing_pyo_objective_component")
    if constraint_count < 1:
        diagnostics.append("missing_pyo_constraint_component")

    model_aliases = _get_model_aliases(fn_node)
    diagnostics.extend(_collect_forbidden_call_diagnostics(fn_node))
    diagnostics.extend(_collect_set_init_diagnostics(fn_node, model_aliases))

    deduped = sorted(set(diagnostics))
    return len(deduped) == 0, deduped


def _extract_contract_block(prompt_text: str) -> str:
    match = re.search(
        r"DataGenerator contract \(source of truth\):\n(.*?)\n\nRequired create_model signature:",
        str(prompt_text or ""),
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match else ""


async def a4_pyomo(state: ModelPack) -> ModelPack:
    """A4 - Pyomo Builder: Generate Pyomo model code."""

    logger.info("a4_pyomo_start", model_id=state.id)

    target_interface = str(state.context.get("target_interface") or "").strip()
    benchmark_mode = target_interface == "create_model"
    if not state.components_math:
        logger.error("a4_missing_prerequisites")
        return state

    try:
        generation_mode = str(state.context.get("generation_mode") or "repair2").strip().lower()
        if generation_mode not in {"prompt_only", "repair2"}:
            generation_mode = "repair2"

        # Check for feedback
        feedback_note = ""
        feedback = state.tests.get("last_feedback")
        if feedback and feedback.target_agent == "A4":
            feedback_note = compact_feedback_context(feedback)
        feedback_context = f"Targeted feedback:\n{feedback_note}\n" if feedback_note else ""

        if benchmark_mode:
            nl_problem = str(state.context.get("nl_problem") or "")
            problem_spec = re.split(
                r"\nDataGenerator contract \(source of truth\):",
                nl_problem,
                maxsplit=1,
            )[0].strip()
            contract_block = _extract_contract_block(nl_problem)
            math_spec_json = state.components_math.model_dump_json(indent=2)

            # Extract exact create_model signature from the DataGenerator contract block in nl_problem
            sig_match = re.search(
                r"(def create_model\([^)]*\)\s*->\s*pyo\.ConcreteModel:)", nl_problem
            )
            signature_line = (
                sig_match.group(1) if sig_match else "def create_model(...) -> pyo.ConcreteModel:"
            )
            trace_input = {
                "agent": "A4_pyomo",
                "mode": "benchmark_create_model",
                "upstream_artifacts": [
                    {
                        "label": "problem_specification",
                        "source": "state.context.nl_problem",
                        "value": problem_spec,
                    },
                    {
                        "label": "data_generator_contract",
                        "source": "state.context.nl_problem",
                        "value": contract_block,
                    },
                    {
                        "label": "required_interface",
                        "source": "state.context.nl_problem",
                        "value": signature_line,
                    },
                    {
                        "label": "components_math",
                        "source": "state.components_math",
                        "value": math_spec_json,
                    },
                ],
            }
            if feedback_note:
                trace_input["upstream_artifacts"].append(
                    {
                        "label": "targeted_feedback",
                        "source": "state.tests.last_feedback",
                        "value": feedback_note,
                    }
                )
            user_prompt_sections = [
                "Structured optimization specification:",
                problem_spec or "Not available",
                "Authoritative DataGenerator contract:",
                contract_block or "Not available",
                "Required interface:",
                signature_line,
                "Mathematical specification (LaTeX):",
                math_spec_json,
            ]
            if feedback_note:
                user_prompt_sections.extend(["Targeted feedback:", feedback_note])
            user_prompt_sections.extend(
                [
                    "Task:",
                    (
                        "Return only the exact create_model implementation. "
                        "Use only the contract inputs above, preserve tuple-key order, "
                        "and do not rename parameters."
                    ),
                ]
            )
            user_prompt = "\n".join(user_prompt_sections)
            system_prompt = PROMPTS["A4_pyomo_create_model"]["system"]
        else:
            nl_problem = str(state.context.get("nl_problem") or "")
            nl_components_json = (
                state.components_nl.model_dump_json(indent=2)
                if state.components_nl is not None
                else "Not available"
            )
            trace_input = {
                "agent": "A4_pyomo",
                "mode": "default_model_builder",
                "upstream_artifacts": [
                    {
                        "label": "problem_input",
                        "source": "state.context.nl_problem",
                        "value": nl_problem,
                    },
                    {
                        "label": "components_nl",
                        "source": "state.components_nl",
                        "value": nl_components_json,
                    },
                    {
                        "label": "components_math",
                        "source": "state.components_math",
                        "value": state.components_math.model_dump_json(indent=2),
                    },
                ],
            }
            if feedback_note:
                trace_input["upstream_artifacts"].append(
                    {
                        "label": "targeted_feedback",
                        "source": "state.tests.last_feedback",
                        "value": feedback_note,
                    }
                )
            user_prompt = f"""Problem:
{nl_problem or 'Not available'}

NL:
{nl_components_json}

Math:
{state.components_math.model_dump_json(indent=2)}

{runtime_data_note()}

{feedback_context}

Task:
Return `ModelBuilder(data: Any) -> pyo.ConcreteModel`."""
            system_prompt = PROMPTS["A4_pyomo"]["system"]

        if benchmark_mode:
            code = llm_client.code_generation_call(
                sys_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                validate=True,
                trace_input=trace_input,
            )
            valid, diagnostics = _validate_create_model_entrypoint(code)
            if not valid and generation_mode == "repair2":
                repair_iterations = state.tests.setdefault("repair_iterations", {})
                repair_iterations["A4_validation"] = (
                    int(repair_iterations.get("A4_validation") or 0) + 1
                )
                diagnostic_lines = "\n".join(f"- {item}" for item in diagnostics)
                repair_sections = [
                    "Repair the existing create_model implementation.",
                    "Authoritative DataGenerator contract:",
                    contract_block or "Not available",
                    "Required interface:",
                    signature_line,
                    "Mathematical specification (LaTeX):",
                    math_spec_json,
                ]
                if feedback_note:
                    repair_sections.extend(["Targeted feedback:", feedback_note])
                repair_sections.extend(
                    [
                        "Validation diagnostics from previous attempt:",
                        diagnostic_lines,
                        "Previous code to repair:",
                        f"```python\n{code}\n```",
                        "Return corrected code only.",
                    ]
                )
                repair_prompt = "\n".join(repair_sections)
                repair_trace_input = {
                    "agent": "A4_pyomo",
                    "mode": "benchmark_create_model_repair",
                    "upstream_artifacts": trace_input["upstream_artifacts"]
                    + [
                        {
                            "label": "validation_diagnostics",
                            "source": "_validate_create_model_entrypoint(previous_code)",
                            "value": diagnostics,
                        },
                        {
                            "label": "previous_code",
                            "source": "previous_llm_output",
                            "value": code,
                        },
                    ],
                }
                repaired_code = llm_client.code_generation_call(
                    sys_prompt=system_prompt,
                    user_prompt=repair_prompt,
                    temperature=0.0,
                    validate=True,
                    trace_input=repair_trace_input,
                )
                repaired_valid, repaired_diagnostics = _validate_create_model_entrypoint(
                    repaired_code
                )
                if not repaired_valid:
                    joined = ", ".join(repaired_diagnostics)
                    raise ValueError(
                        "benchmark_create_model_validation_failed_after_repair: " f"{joined}"
                    )
                code = repaired_code
            elif not valid:
                joined = ", ".join(diagnostics)
                raise ValueError(f"benchmark_create_model_validation_failed: {joined}")
        else:
            code = llm_client.code_generation_call(
                sys_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                validate=True,
                trace_input=trace_input,
            )

        if benchmark_mode:
            valid, diagnostics = _validate_create_model_entrypoint(code)
            if not valid:
                joined = ", ".join(diagnostics)
                raise ValueError(f"benchmark_create_model_validation_failed: {joined}")

        state.code.model_builder = CodeBlob(
            language="python",
            filename="create_model.py" if benchmark_mode else "model_builder.py",
            source=code,
        )

        logger.info(
            "a4_pyomo_success",
            code_length=len(code),
            target_interface=target_interface or "default",
            generation_mode=generation_mode if benchmark_mode else "default",
        )

    except Exception as e:
        logger.error("a4_pyomo_error", error=str(e))

    return state
