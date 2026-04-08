# modelpack/agents/build_model.py
import ast
import builtins
import re
import symtable
from typing import Dict, List, Optional, Set, Tuple

import structlog

from ..schemas import ModelPack, CodeBlob
from ..llm import llm_client
from ..prompts import (
    PROMPTS,
    compact_feedback_context,
    llm_problem_text,
    problem_input_note,
    runtime_data_note,
)
from .utils import normalize_generation_mode

logger = structlog.get_logger(__name__)
_DICT_LIKE_ANNOTATIONS = {"dict", "Dict", "Mapping", "MutableMapping"}


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


def _parse_signature_contract_fn(signature_line: str) -> Optional[ast.FunctionDef]:
    signature = signature_line.strip()
    if not signature.startswith("def create_model("):
        return None
    try:
        tree = ast.parse(f"{signature}\n    pass\n")
    except SyntaxError:
        return None
    fn_node = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "create_model"),
        None,
    )
    return fn_node


def _apply_required_signature_contract(source: str, signature_line: Optional[str]) -> str:
    if not signature_line:
        return source
    contract_fn = _parse_signature_contract_fn(signature_line)
    if contract_fn is None:
        return source
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    fn_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "create_model"
        ),
        None,
    )
    if fn_node is None:
        return source

    source_args = list(fn_node.args.posonlyargs) + list(fn_node.args.args) + list(fn_node.args.kwonlyargs)
    contract_args = (
        list(contract_fn.args.posonlyargs)
        + list(contract_fn.args.args)
        + list(contract_fn.args.kwonlyargs)
    )
    if len(source_args) != len(contract_args):
        return source
    if [arg.arg for arg in source_args] != [arg.arg for arg in contract_args]:
        return source

    changed = False
    for source_arg, contract_arg in zip(source_args, contract_args):
        if source_arg.annotation is None and contract_arg.annotation is not None:
            source_arg.annotation = contract_arg.annotation
            changed = True
    if fn_node.returns is None and contract_fn.returns is not None:
        fn_node.returns = contract_fn.returns
        changed = True
    if not changed:
        return source
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _annotation_base_name(node: Optional[ast.AST]) -> str:
    if node is None:
        return ""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _annotation_base_name(node.value)
    return ""


def _dict_arg_names(fn_node: ast.FunctionDef) -> Set[str]:
    names: Set[str] = set()
    for arg in (
        list(fn_node.args.posonlyargs)
        + list(fn_node.args.args)
        + list(fn_node.args.kwonlyargs)
    ):
        if _annotation_base_name(arg.annotation) in _DICT_LIKE_ANNOTATIONS:
            names.add(arg.arg)
    return names


def _subscript_args(node: Optional[ast.AST]) -> List[ast.AST]:
    if not isinstance(node, ast.Subscript):
        return []
    if isinstance(node.slice, ast.Tuple):
        return list(node.slice.elts)
    return [node.slice]


def _dict_annotation_kind(node: Optional[ast.AST]) -> Optional[str]:
    if _annotation_base_name(node) not in _DICT_LIKE_ANNOTATIONS:
        return None

    args = _subscript_args(node)
    key_annotation = args[0] if args else None
    value_annotation = args[1] if len(args) > 1 else None
    if _annotation_base_name(key_annotation) in {"tuple", "Tuple"}:
        return "tuple_dict"
    if _annotation_base_name(value_annotation) in _DICT_LIKE_ANNOTATIONS:
        return "nested_dict"
    return "dict"


def _dict_arg_kinds(fn_node: ast.FunctionDef) -> Dict[str, str]:
    kinds: Dict[str, str] = {}
    for arg in (
        list(fn_node.args.posonlyargs)
        + list(fn_node.args.args)
        + list(fn_node.args.kwonlyargs)
    ):
        kind = _dict_annotation_kind(arg.annotation)
        if kind:
            kinds[arg.arg] = kind
    return kinds


def _bool_dict_arg_names(fn_node: ast.FunctionDef) -> Set[str]:
    names: Set[str] = set()
    for arg in (
        list(fn_node.args.posonlyargs)
        + list(fn_node.args.args)
        + list(fn_node.args.kwonlyargs)
    ):
        if _annotation_base_name(arg.annotation) not in _DICT_LIKE_ANNOTATIONS:
            continue
        args = _subscript_args(arg.annotation)
        value_annotation = args[1] if len(args) > 1 else None
        if _annotation_base_name(value_annotation) == "bool":
            names.add(arg.arg)
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


def _iter_executable_nodes(fn_node: ast.FunctionDef):
    for stmt in fn_node.body:
        yield from ast.walk(stmt)


def _parent_map(node: ast.AST) -> Dict[ast.AST, ast.AST]:
    parents: Dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(node):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _integer_literal_index_repr(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return repr(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    ):
        sign = "-" if isinstance(node.op, ast.USub) else "+"
        return f"{sign}{node.operand.value!r}"
    if isinstance(node, ast.Tuple):
        item_reprs: List[str] = []
        for elt in node.elts:
            literal = _integer_literal_index_repr(elt)
            if literal is None:
                return None
            item_reprs.append(literal)
        return f"({', '.join(item_reprs)})"
    return None


def _loaded_arg_names(node: ast.AST, arg_names: Set[str]) -> Set[str]:
    names: Set[str] = set()
    for subnode in ast.walk(node):
        if (
            isinstance(subnode, ast.Name)
            and isinstance(subnode.ctx, ast.Load)
            and subnode.id in arg_names
        ):
            names.add(subnode.id)
    return names


def _name_tuple_slice(node: ast.AST) -> Optional[Tuple[str, ...]]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Tuple):
        names: List[str] = []
        for elt in node.elts:
            if not isinstance(elt, ast.Name):
                return None
            names.append(elt.id)
        return tuple(names)
    return None


def _is_zero_numeric_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value == 0
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        return node.operand.value == 0
    return False


def _is_name_attr_call(node: ast.AST, arg_name: str, attrs: Set[str]) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Attribute) or node.func.attr not in attrs:
        return False
    return isinstance(node.func.value, ast.Name) and node.func.value.id == arg_name


def _is_subscript_attr_call(node: ast.AST, arg_name: str, attrs: Set[str]) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Attribute) or node.func.attr not in attrs:
        return False
    base = node.func.value
    return (
        isinstance(base, ast.Subscript)
        and isinstance(base.value, ast.Name)
        and base.value.id == arg_name
    )


def _integer_constant_value(node: Optional[ast.AST]) -> Optional[int]:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    ):
        value = int(node.operand.value)
        return -value if isinstance(node.op, ast.USub) else value
    return None


def _uses_synthetic_ranges(fn_node: ast.FunctionDef) -> bool:
    for node in _iter_executable_nodes(fn_node):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func) in {"range", "pyo.RangeSet", "pyomo.environ.RangeSet"}:
            return True
    return False


def _has_tuple_support_derivation(fn_node: ast.FunctionDef, arg_name: str) -> bool:
    for node in _iter_executable_nodes(fn_node):
        if _is_name_attr_call(node, arg_name, {"keys", "items", "get"}):
            return True
        if isinstance(node, ast.For):
            iter_expr = node.iter
            if isinstance(iter_expr, ast.Name) and iter_expr.id == arg_name:
                return True
            if _is_name_attr_call(iter_expr, arg_name, {"keys", "items"}):
                return True
        if isinstance(node, ast.comprehension):
            iter_expr = node.iter
            if isinstance(iter_expr, ast.Name) and iter_expr.id == arg_name:
                return True
            if _is_name_attr_call(iter_expr, arg_name, {"keys", "items"}):
                return True
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.In):
            if (
                isinstance(node.comparators[0], ast.Name)
                and node.comparators[0].id == arg_name
                and isinstance(node.left, ast.Tuple)
            ):
                return True
    return False


def _has_nested_support_derivation(fn_node: ast.FunctionDef, arg_name: str) -> bool:
    for node in _iter_executable_nodes(fn_node):
        if isinstance(node, ast.For):
            iter_expr = node.iter
            if (
                isinstance(iter_expr, ast.Subscript)
                and isinstance(iter_expr.value, ast.Name)
                and iter_expr.value.id == arg_name
            ):
                return True
            if _is_subscript_attr_call(iter_expr, arg_name, {"keys", "items", "values"}):
                return True
        if isinstance(node, ast.comprehension):
            iter_expr = node.iter
            if (
                isinstance(iter_expr, ast.Subscript)
                and isinstance(iter_expr.value, ast.Name)
                and iter_expr.value.id == arg_name
            ):
                return True
            if _is_subscript_attr_call(iter_expr, arg_name, {"keys", "items", "values"}):
                return True
    return False


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
    for node in _iter_executable_nodes(fn_node):
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
    for node in _iter_executable_nodes(fn_node):
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


def _collect_dict_literal_subscript_diagnostics(fn_node: ast.FunctionDef) -> List[str]:
    dict_args = _dict_arg_names(fn_node)
    if not dict_args:
        return []

    diagnostics: List[str] = []
    for node in _iter_executable_nodes(fn_node):
        if not isinstance(node, ast.Subscript):
            continue
        if not isinstance(node.value, ast.Name):
            continue
        arg_name = node.value.id
        if arg_name not in dict_args:
            continue
        literal_index = _integer_literal_index_repr(node.slice)
        if literal_index is None:
            continue
        diagnostics.append(f"dict_arg_literal_subscript:{arg_name}:{literal_index}")
    return diagnostics


def _collect_tuple_dict_support_diagnostics(fn_node: ast.FunctionDef) -> List[str]:
    arg_kinds = _dict_arg_kinds(fn_node)
    tuple_args = {name for name, kind in arg_kinds.items() if kind == "tuple_dict"}
    if not tuple_args:
        return []

    diagnostics: List[str] = []
    for arg_name in tuple_args:
        if _has_tuple_support_derivation(fn_node, arg_name):
            continue
        for node in _iter_executable_nodes(fn_node):
            if not isinstance(node, ast.Subscript):
                continue
            if not isinstance(node.value, ast.Name) or node.value.id != arg_name:
                continue
            index_names = _name_tuple_slice(node.slice)
            if index_names is None or len(index_names) < 2:
                continue
            diagnostics.append(f"tuple_dict_cartesian_access_without_support:{arg_name}")
            break
    return diagnostics


def _collect_tuple_dict_dense_param_initializer_diagnostics(fn_node: ast.FunctionDef) -> List[str]:
    arg_kinds = _dict_arg_kinds(fn_node)
    tuple_args = {name for name, kind in arg_kinds.items() if kind == "tuple_dict"}
    if not tuple_args:
        return []

    diagnostics: List[str] = []
    for node in _iter_executable_nodes(fn_node):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func) not in {"pyo.Param", "pyomo.environ.Param"}:
            continue
        initialize_expr = _set_initialize_expr(node)
        if initialize_expr is None:
            continue
        referenced_args = _loaded_arg_names(initialize_expr, tuple_args)
        if not referenced_args:
            continue
        if len(node.args) < 2:
            continue
        for arg_name in sorted(referenced_args):
            diagnostics.append(f"tuple_dict_dense_param_initializer:{arg_name}")
    return diagnostics


def _collect_nested_dict_support_diagnostics(fn_node: ast.FunctionDef) -> List[str]:
    arg_kinds = _dict_arg_kinds(fn_node)
    nested_args = {name for name, kind in arg_kinds.items() if kind == "nested_dict"}
    if not nested_args:
        return []

    diagnostics: List[str] = []
    for arg_name in nested_args:
        if _has_nested_support_derivation(fn_node, arg_name):
            continue
        for node in _iter_executable_nodes(fn_node):
            if not isinstance(node, ast.Subscript):
                continue
            if not isinstance(node.value, ast.Subscript):
                continue
            outer = node.value
            if not isinstance(outer.value, ast.Name) or outer.value.id != arg_name:
                continue
            outer_index = _name_tuple_slice(outer.slice)
            inner_index = _name_tuple_slice(node.slice)
            if outer_index is None or inner_index is None:
                continue
            diagnostics.append(f"nested_dict_cartesian_access_without_support:{arg_name}")
            break
    return diagnostics


def _collect_nested_dict_literal_inner_subscript_diagnostics(
    fn_node: ast.FunctionDef,
) -> List[str]:
    if not _uses_synthetic_ranges(fn_node):
        return []

    arg_kinds = _dict_arg_kinds(fn_node)
    nested_args = {name for name, kind in arg_kinds.items() if kind == "nested_dict"}
    if not nested_args:
        return []

    diagnostics: List[str] = []
    for arg_name in nested_args:
        if _has_nested_support_derivation(fn_node, arg_name):
            continue
        for node in _iter_executable_nodes(fn_node):
            if not isinstance(node, ast.Subscript):
                continue
            if not isinstance(node.value, ast.Subscript):
                continue
            outer = node.value
            if not isinstance(outer.value, ast.Name) or outer.value.id != arg_name:
                continue
            if _name_tuple_slice(outer.slice) is None:
                continue
            literal_index = _integer_literal_index_repr(node.slice)
            if literal_index is None:
                continue
            diagnostics.append(
                f"nested_dict_literal_inner_subscript_without_support:{arg_name}:{literal_index}"
            )
            break
    return diagnostics


def _collect_no_effect_arg_diagnostics(
    fn_node: ast.FunctionDef,
    arg_names: Set[str],
) -> List[str]:
    diagnostics: List[str] = []
    for node in _iter_executable_nodes(fn_node):
        if isinstance(node, ast.Assign):
            value_arg_names = _loaded_arg_names(node.value, arg_names)
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith("_unused"):
                    for arg_name in value_arg_names:
                        diagnostics.append(f"no_effect_unused_alias_arg:{arg_name}")
        elif isinstance(node, ast.AnnAssign):
            value_arg_names = (
                _loaded_arg_names(node.value, arg_names) if node.value is not None else set()
            )
            if isinstance(node.target, ast.Name) and node.target.id.startswith("_unused"):
                for arg_name in value_arg_names:
                    diagnostics.append(f"no_effect_unused_alias_arg:{arg_name}")

        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            zero_side = None
            if _is_zero_numeric_literal(node.left):
                zero_side = node.right
            elif _is_zero_numeric_literal(node.right):
                zero_side = node.left
            if zero_side is not None:
                for arg_name in _loaded_arg_names(zero_side, arg_names):
                    diagnostics.append(f"no_effect_zero_multiplier_arg:{arg_name}")

        if (
            isinstance(node, ast.If)
            and node.body
            and all(isinstance(stmt, ast.Pass) for stmt in node.body)
            and not node.orelse
        ):
            for arg_name in _loaded_arg_names(node.test, arg_names):
                diagnostics.append(f"no_effect_branch_arg:{arg_name}")
    return diagnostics


def _collect_dummy_component_diagnostics(
    fn_node: ast.FunctionDef,
    model_aliases: Set[str],
) -> List[str]:
    if not model_aliases:
        return []

    diagnostics: List[str] = []
    for node in _iter_executable_nodes(fn_node):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            if (
                isinstance(node, ast.Call)
                and _call_name(node.func) == "setattr"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in model_aliases
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
                and node.args[1].value.startswith("dummy_")
            ):
                diagnostics.append(f"dummy_component_name:{node.args[1].value}")
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if not isinstance(target, ast.Attribute):
                continue
            root_name = _attribute_root_name(target)
            if root_name not in model_aliases:
                continue
            if target.attr.startswith("dummy_"):
                diagnostics.append(f"dummy_component_name:{target.attr}")
    return diagnostics


def _collect_model_component_alias_diagnostics(
    fn_node: ast.FunctionDef,
    model_aliases: Set[str],
) -> List[str]:
    if not model_aliases:
        return []

    diagnostics: List[str] = []
    for node in _iter_executable_nodes(fn_node):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Attribute):
            continue
        value_root = _attribute_root_name(value)
        if value_root not in model_aliases:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if not isinstance(target, ast.Attribute):
                continue
            target_root = _attribute_root_name(target)
            if target_root not in model_aliases:
                continue
            if target.attr == value.attr:
                continue
            diagnostics.append(
                f"model_component_alias_assignment:{target.attr}:{value.attr}"
            )
    return diagnostics


def _collect_model_raw_arg_aliases(
    fn_node: ast.FunctionDef,
    model_aliases: Set[str],
    arg_names: Set[str],
) -> Dict[str, str]:
    if not model_aliases or not arg_names:
        return {}

    aliases: Dict[str, str] = {}
    for node in _iter_executable_nodes(fn_node):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Name) or value.id not in arg_names:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if not isinstance(target, ast.Attribute):
                continue
            if _attribute_root_name(target) not in model_aliases:
                continue
            aliases[target.attr] = value.id
    return aliases


def _infer_model_set_dimensions(
    fn_node: ast.FunctionDef,
    model_aliases: Set[str],
) -> Dict[str, int]:
    dimensions: Dict[str, int] = {}
    if not model_aliases:
        return dimensions

    for node in _iter_executable_nodes(fn_node):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Call):
            continue
        call_name = _call_name(value.func)
        if call_name not in {
            "pyo.Set",
            "pyomo.environ.Set",
            "pyo.RangeSet",
            "pyomo.environ.RangeSet",
        }:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        dim = 1
        if call_name in {"pyo.Set", "pyomo.environ.Set"}:
            for keyword in value.keywords:
                if keyword.arg == "dimen":
                    literal_dim = _integer_constant_value(keyword.value)
                    if literal_dim is not None and literal_dim > 0:
                        dim = literal_dim
                    break
        for target in targets:
            if not isinstance(target, ast.Attribute):
                continue
            if _attribute_root_name(target) not in model_aliases:
                continue
            dimensions[target.attr] = dim
    return dimensions


def _collect_constraint_rule_arity_diagnostics(
    fn_node: ast.FunctionDef,
    model_aliases: Set[str],
) -> List[str]:
    if not model_aliases:
        return []

    set_dimensions = _infer_model_set_dimensions(fn_node, model_aliases)
    local_rule_arities = {
        node.name: len(_function_arg_names(node))
        for node in ast.walk(fn_node)
        if isinstance(node, ast.FunctionDef) and node is not fn_node
    }
    diagnostics: List[str] = []

    for node in _iter_executable_nodes(fn_node):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func) not in {"pyo.Constraint", "pyomo.environ.Constraint"}:
            continue

        rule_name: Optional[str] = None
        for keyword in node.keywords:
            if keyword.arg == "rule" and isinstance(keyword.value, ast.Name):
                rule_name = keyword.value.id
                break
        if not rule_name:
            continue
        actual_arity = local_rule_arities.get(rule_name)
        if actual_arity is None:
            continue

        expected_index_arity = 0
        unknown_dimension = False
        for arg in node.args:
            if not isinstance(arg, ast.Attribute):
                unknown_dimension = True
                break
            if _attribute_root_name(arg) not in model_aliases:
                unknown_dimension = True
                break
            dim = set_dimensions.get(arg.attr)
            if dim is None:
                unknown_dimension = True
                break
            expected_index_arity += dim
        if unknown_dimension:
            continue

        expected_arity = 1 + expected_index_arity
        if actual_arity != expected_arity:
            diagnostics.append(
                f"constraint_rule_arity_mismatch:{rule_name}:expected{expected_arity}:got{actual_arity}"
            )
    return diagnostics


def _is_raw_bool_dict_access(node: ast.AST, raw_bool_attrs: Set[str]) -> bool:
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
        return node.value.attr in raw_bool_attrs
    if isinstance(node, ast.Call):
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Attribute)
        ):
            return node.func.value.attr in raw_bool_attrs
    return False


def _collect_bool_dict_compare_diagnostics(
    fn_node: ast.FunctionDef,
    model_aliases: Set[str],
) -> List[str]:
    raw_bool_aliases = _collect_model_raw_arg_aliases(
        fn_node, model_aliases, _bool_dict_arg_names(fn_node)
    )
    if not raw_bool_aliases:
        return []

    diagnostics: List[str] = []
    raw_bool_attrs = set(raw_bool_aliases)
    for node in _iter_executable_nodes(fn_node):
        if not isinstance(node, ast.Compare):
            continue
        for expr in [node.left, *node.comparators]:
            if not _is_raw_bool_dict_access(expr, raw_bool_attrs):
                continue
            attr_name: Optional[str] = None
            if isinstance(expr, ast.Subscript) and isinstance(expr.value, ast.Attribute):
                attr_name = expr.value.attr
            elif (
                isinstance(expr, ast.Call)
                and isinstance(expr.func, ast.Attribute)
                and isinstance(expr.func.value, ast.Attribute)
            ):
                attr_name = expr.func.value.attr
            if attr_name is not None:
                diagnostics.append(
                    f"bool_dict_compare_without_cast:{raw_bool_aliases[attr_name]}"
                )
            break
    return diagnostics


def _has_pyomo_pyo_import(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pyomo.environ" and alias.asname == "pyo":
                    return True
        if isinstance(node, ast.ImportFrom):
            if node.module != "pyomo":
                continue
            for alias in node.names:
                if alias.name == "environ" and alias.asname == "pyo":
                    return True
    return False


def _uses_pyo_alias(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "pyo":
            return True
    return False


def _collect_undefined_name_diagnostics(source: str) -> List[str]:
    try:
        module_table = symtable.symtable(source, "create_model.py", "exec")
    except SyntaxError:
        return []

    create_model_table = next(
        (
            child
            for child in module_table.get_children()
            if child.get_name() == "create_model" and child.get_type() == "function"
        ),
        None,
    )
    if create_model_table is None:
        return []

    allowed_names = {symbol.get_name() for symbol in module_table.get_symbols()}
    allowed_names.update(dir(builtins))

    diagnostics: Set[str] = set()

    def visit(table: symtable.SymbolTable) -> None:
        for symbol in table.get_symbols():
            if not (symbol.is_global() and symbol.is_referenced()):
                continue
            if symbol.get_name() in allowed_names:
                continue
            diagnostics.add(f"undefined_name:{symbol.get_name()}")
        for child in table.get_children():
            visit(child)

    visit(create_model_table)
    return sorted(diagnostics)


def _name_used_only_as_attribute_root(fn_node: ast.FunctionDef, name: str) -> bool:
    parents = _parent_map(fn_node)
    seen = False
    for node in ast.walk(fn_node):
        if not (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == name
        ):
            continue
        seen = True
        parent = parents.get(node)
        if not (isinstance(parent, ast.Attribute) and parent.value is node):
            return False
    return seen


def _name_is_loaded(fn_node: ast.FunctionDef, name: str) -> bool:
    for node in ast.walk(fn_node):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == name
        ):
            return True
    return False


class _CreateModelAutoFixer(ast.NodeTransformer):
    def __init__(self, rename_map: Dict[str, str], raw_bool_attrs: Set[str]):
        self.rename_map = rename_map
        self.raw_bool_attrs = raw_bool_attrs

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self.rename_map:
            return ast.copy_location(
                ast.Name(id=self.rename_map[node.id], ctx=node.ctx),
                node,
            )
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        node = self.generic_visit(node)
        if _is_raw_bool_dict_access(node.left, self.raw_bool_attrs):
            node.left = ast.copy_location(
                ast.Call(func=ast.Name(id="int", ctx=ast.Load()), args=[node.left], keywords=[]),
                node.left,
            )
        node.comparators = [
            ast.copy_location(
                ast.Call(func=ast.Name(id="int", ctx=ast.Load()), args=[expr], keywords=[]),
                expr,
            )
            if _is_raw_bool_dict_access(expr, self.raw_bool_attrs)
            else expr
            for expr in node.comparators
        ]
        return node


def _apply_create_model_autofixes(
    source: str, required_signature: Optional[str] = None
) -> str:
    source = _apply_required_signature_contract(source, required_signature)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    fn_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "create_model"
        ),
        None,
    )
    if fn_node is None:
        return source

    model_aliases = sorted(_get_model_aliases(fn_node))
    rename_map: Dict[str, str] = {}
    if len(model_aliases) == 1:
        model_alias = model_aliases[0]
        for diagnostic in _collect_undefined_name_diagnostics(source):
            _, undefined_name = diagnostic.split(":", maxsplit=1)
            if _name_used_only_as_attribute_root(fn_node, undefined_name):
                rename_map[undefined_name] = model_alias

    raw_bool_attr_names = set(
        _collect_model_raw_arg_aliases(fn_node, set(model_aliases), _bool_dict_arg_names(fn_node))
    )
    removed_kwargs = False
    if fn_node.args.kwarg is not None and not _name_is_loaded(fn_node, fn_node.args.kwarg.arg):
        fn_node.args.kwarg = None
        removed_kwargs = True

    needs_import = _uses_pyo_alias(tree) and not _has_pyomo_pyo_import(tree)
    if not rename_map and not raw_bool_attr_names and not needs_import and not removed_kwargs:
        return source

    transformed = tree
    if rename_map or raw_bool_attr_names:
        transformed = _CreateModelAutoFixer(rename_map, raw_bool_attr_names).visit(transformed)
        ast.fix_missing_locations(transformed)
    if needs_import:
        transformed.body.insert(
            0,
            ast.Import(names=[ast.alias(name="pyomo.environ", asname="pyo")]),
        )
        ast.fix_missing_locations(transformed)
    return ast.unparse(transformed)


def _diagnostic_repair_hints(diagnostics: List[str]) -> List[str]:
    hints: List[str] = []
    for diagnostic in diagnostics:
        if diagnostic == "missing_pyomo_import_alias_pyo":
            hints.append("Add `import pyomo.environ as pyo`.")
        elif diagnostic.startswith("tuple_dict_dense_param_initializer:"):
            _, arg_name = diagnostic.split(":", maxsplit=1)
            hints.append(
                f"`{arg_name}` is a sparse tuple-keyed dict. "
                f"NEVER use `pyo.Param(set1, set2, ..., initialize=...)` with multiple positional Set args for it — that is the forbidden dense pattern. "
                f"Fix with pattern A: `S = pyo.Set(initialize=list({arg_name}.keys()), dimen=N)` then `pyo.Param(S, initialize={arg_name}, default=0)`. "
                f"Or pattern B: remove the pyo.Param entirely and use `{arg_name}.get((i, j, ...), 0)` directly inside constraint rule bodies."
            )
        elif diagnostic == "set_initialize_references_model_component":
            hints.append(
                "Do not initialize a `pyo.Set` from `model.<component>` values. Build the iterable from plain Python data before creating the set."
            )
        elif diagnostic.startswith("constraint_rule_arity_mismatch:"):
            _, rule_name, expected, actual = diagnostic.split(":", maxsplit=3)
            hints.append(
                f"`{rule_name}` has the wrong rule signature. Match the number of rule indices to the indexed set dimensions ({expected}, {actual})."
            )
        elif diagnostic.startswith("undefined_name:"):
            _, name = diagnostic.split(":", maxsplit=1)
            hints.append(
                f"`{name}` is undefined. Use the actual model alias or function argument consistently."
            )
        elif diagnostic.startswith("bool_dict_compare_without_cast:"):
            _, arg_name = diagnostic.split(":", maxsplit=1)
            hints.append(
                f"`{arg_name}` has boolean values. Cast lookup results to `int(...)` or branch on them before using them in Pyomo inequalities."
            )
        elif diagnostic == "create_model_kwargs_not_allowed":
            hints.append("Do not add `**kwargs`; keep the exact required `create_model(...)` signature.")
        elif diagnostic.startswith("tuple_dict_cartesian_access_without_support:"):
            _, arg_name = diagnostic.split(":", maxsplit=1)
            hints.append(
                f"`{arg_name}` is a sparse tuple-keyed dict. Never use `{arg_name}[i, j, ...]` (direct subscript) when iterating cartesian index sets — it will KeyError on missing keys. "
                f"Replace every `{arg_name}[(i, j, ...)]` with `{arg_name}.get((i, j, ...), 0)` OR iterate `for key in {arg_name}: ...` / `for key in {arg_name}.keys(): ...` to stay within the dict's actual support."
            )
        elif diagnostic.startswith("unused_create_model_arg:"):
            _, arg_name = diagnostic.split(":", maxsplit=1)
            hints.append(
                f"`{arg_name}` is a required argument that is never used in any constraint or objective expression. "
                f"Use it meaningfully — not as a zero-multiplier or dead alias."
            )
        elif diagnostic.startswith("no_effect_zero_multiplier_arg:"):
            _, arg_name = diagnostic.split(":", maxsplit=1)
            hints.append(
                f"`{arg_name}` is multiplied by zero, which has no effect. Use it in an actual constraint or objective expression."
            )
    return sorted(set(hints))


def _validate_create_model_entrypoint(
    source: str, required_signature: Optional[str] = None
) -> Tuple[bool, List[str]]:
    """Validate benchmark-mode code quality for create_model entrypoint."""
    source = _apply_required_signature_contract(source, required_signature)
    diagnostics: List[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, [f"invalid_python:{exc}"]

    fn_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    async_defs = [node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)]
    class_defs = [node for node in tree.body if isinstance(node, ast.ClassDef)]

    if _uses_pyo_alias(tree) and not _has_pyomo_pyo_import(tree):
        diagnostics.append("missing_pyomo_import_alias_pyo")

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
    diagnostics.extend(_collect_dict_literal_subscript_diagnostics(fn_node))
    diagnostics.extend(_collect_tuple_dict_support_diagnostics(fn_node))
    diagnostics.extend(_collect_tuple_dict_dense_param_initializer_diagnostics(fn_node))
    diagnostics.extend(_collect_nested_dict_support_diagnostics(fn_node))
    diagnostics.extend(_collect_nested_dict_literal_inner_subscript_diagnostics(fn_node))
    diagnostics.extend(_collect_no_effect_arg_diagnostics(fn_node, arg_name_set))
    diagnostics.extend(_collect_dummy_component_diagnostics(fn_node, model_aliases))
    diagnostics.extend(_collect_model_component_alias_diagnostics(fn_node, model_aliases))
    diagnostics.extend(_collect_constraint_rule_arity_diagnostics(fn_node, model_aliases))
    diagnostics.extend(_collect_bool_dict_compare_diagnostics(fn_node, model_aliases))
    diagnostics.extend(_collect_undefined_name_diagnostics(source))

    deduped = sorted(set(diagnostics))
    return len(deduped) == 0, deduped


async def build_model(state: ModelPack) -> ModelPack:
    """Generate Pyomo model code."""

    logger.info("build_model_start", model_id=state.id)

    target_interface = str(state.context.get("target_interface") or "").strip()
    benchmark_mode = target_interface == "create_model"
    if not state.components_math:
        logger.error("build_model_missing_prerequisites")
        return state

    try:
        generation_mode = normalize_generation_mode(state.context.get("generation_mode") or "")
        state.tests["build_model_error"] = None
        state.code.model_builder = None

        # Check for feedback
        feedback_note = ""
        feedback = state.tests.get("last_feedback")
        if feedback and feedback.target_agent == "build_model":
            feedback_note = compact_feedback_context(feedback)
        feedback_context = f"Targeted feedback:\n{feedback_note}\n" if feedback_note else ""

        if benchmark_mode:
            nl_problem = llm_problem_text(
                state.context.get("nl_problem") or "",
                preserve_data_generator_contract=True,
            )
            problem_spec = re.split(
                r"\nRequired create_model signature:",
                nl_problem,
                maxsplit=1,
            )[0].strip()
            math_spec_json = state.components_math.model_dump_json(indent=2)

            sig_match = re.search(
                r"(def create_model\([^)]*\)\s*->\s*pyo\.ConcreteModel:)", nl_problem
            )
            signature_line = (
                sig_match.group(1) if sig_match else "def create_model(...) -> pyo.ConcreteModel:"
            )
            problem_input_mode = problem_input_note(problem_spec)
            trace_input = {
                "agent": "build_model",
                "mode": "benchmark_create_model",
                "upstream_artifacts": [
                    {
                        "label": "problem_input",
                        "source": "llm_problem_text(state.context.nl_problem)",
                        "value": problem_spec,
                    },
                    {
                        "label": "required_interface",
                        "source": "llm_problem_text(state.context.nl_problem)",
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
                "Optimization problem input:",
                problem_spec or "Not available",
                problem_input_mode,
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
                        "Treat the optimization problem input and interface contract as authoritative, "
                        "use the math summary only when consistent with them, preserve tuple-key order, "
                        "and do not rename parameters."
                    ),
                ]
            )
            user_prompt = "\n".join(user_prompt_sections)
            system_prompt = PROMPTS["build_model_create_model"]["system"]
        else:
            nl_problem = llm_problem_text(state.context.get("nl_problem") or "")
            nl_components_json = (
                state.components_nl.model_dump_json(indent=2)
                if state.components_nl is not None
                else "Not available"
            )
            trace_input = {
                "agent": "build_model",
                "mode": "default_model_builder",
                "upstream_artifacts": [
                    {
                        "label": "problem_input",
                        "source": "llm_problem_text(state.context.nl_problem)",
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

{feedback_context}"""
            system_prompt = PROMPTS["build_model"]["system"]

        if benchmark_mode:
            code = llm_client.code_generation_call(
                sys_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                validate=True,
                trace_input=trace_input,
            )
            code = _apply_create_model_autofixes(code, required_signature=signature_line)
            valid, diagnostics = _validate_create_model_entrypoint(
                code, required_signature=signature_line
            )
            if not valid and generation_mode == "repair_once":
                repair_iterations = state.tests.setdefault("repair_iterations", {})
                repair_iterations["build_model_validation"] = (
                    int(repair_iterations.get("build_model_validation") or 0) + 1
                )
                diagnostic_lines = "\n".join(f"- {item}" for item in diagnostics)
                repair_hints = _diagnostic_repair_hints(diagnostics)
                repair_sections = [
                    "Repair your previous create_model implementation.",
                    "Keep the same required interface and upstream contract from the earlier messages.",
                    "Validation diagnostics from the previous attempt:",
                    diagnostic_lines,
                ]
                if repair_hints:
                    repair_sections.extend(
                        [
                            "Concrete fixes required:",
                            "\n".join(f"- {item}" for item in repair_hints),
                        ]
                    )
                if feedback_note:
                    repair_sections.extend(
                        [
                            "Also address the targeted feedback already provided earlier in this conversation.",
                        ]
                    )
                repair_sections.append("Return corrected code only.")
                repair_prompt = "\n".join(repair_sections)
                repair_trace_input = {
                    "agent": "build_model",
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
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": code},
                        {"role": "user", "content": repair_prompt},
                    ],
                    temperature=0.0,
                    validate=True,
                    trace_input=repair_trace_input,
                )
                repaired_code = _apply_create_model_autofixes(
                    repaired_code, required_signature=signature_line
                )
                repaired_valid, repaired_diagnostics = _validate_create_model_entrypoint(
                    repaired_code, required_signature=signature_line
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
            code = _apply_create_model_autofixes(code, required_signature=signature_line)
            valid, diagnostics = _validate_create_model_entrypoint(
                code, required_signature=signature_line
            )
            if not valid:
                joined = ", ".join(diagnostics)
                raise ValueError(f"benchmark_create_model_validation_failed: {joined}")

        state.code.model_builder = CodeBlob(
            language="python",
            filename="create_model.py" if benchmark_mode else "model_builder.py",
            source=code,
        )
        state.tests["build_model_error"] = None
        if feedback and feedback.target_agent == "build_model":
            state.tests["last_feedback"] = None

        logger.info(
            "build_model_success",
            code_length=len(code),
            target_interface=target_interface or "default",
            generation_mode=generation_mode if benchmark_mode else "default",
        )

    except Exception as e:
        state.tests["build_model_error"] = str(e)
        logger.error("build_model_error", error=str(e))

    return state
