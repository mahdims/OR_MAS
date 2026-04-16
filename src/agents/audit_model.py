# modelpack/agents/audit_model.py
"""Audit the generated create_model against the structured NL constraint list.

Targets the checker_reject and over-constrained candidate_solve failure modes by
feeding the LLM a structured list of NL constraints (id + desc) and asking it
to add any that are missing from the code — without inventing new ones.
"""

import json
import re
from typing import List, Optional

import structlog

from ..llm import llm_client
from ..prompts import PROMPTS, llm_problem_text
from ..schemas import CodeBlob, ModelPack
from .build_model import (
    _apply_create_model_autofixes,
    _validate_create_model_entrypoint,
)

logger = structlog.get_logger(__name__)


def _nl_constraint_items(components_nl) -> List[dict]:
    items: List[dict] = []
    for bucket in ("constraints_basic", "constraints_logical"):
        for item in getattr(components_nl, bucket, []) or []:
            items.append(
                {
                    "id": getattr(item, "id", ""),
                    "name": getattr(item, "name", ""),
                    "desc": getattr(item, "desc", ""),
                    "kind": bucket,
                }
            )
    return items


def _extract_signature_line(problem_text: str) -> str:
    match = re.search(r"(def create_model\([^)]*\)\s*->\s*pyo\.ConcreteModel:)", problem_text)
    return match.group(1) if match else "def create_model(...) -> pyo.ConcreteModel:"


async def audit_model(state: ModelPack) -> ModelPack:
    """Audit create_model against the structured NL constraint list."""

    logger.info("audit_model_start", model_id=state.id)

    if str(state.context.get("target_interface") or "").strip() != "create_model":
        return state
    model_builder = state.code.model_builder
    code = str(getattr(model_builder, "source", "") or "").strip()
    if not code or state.components_nl is None:
        return state

    nl_constraints = _nl_constraint_items(state.components_nl)
    if not nl_constraints:
        return state

    nl_problem = llm_problem_text(
        state.context.get("nl_problem") or "",
        preserve_data_generator_contract=True,
    )
    problem_spec = re.split(r"\nRequired create_model signature:", nl_problem, maxsplit=1)[0].strip()
    signature_line = _extract_signature_line(nl_problem)

    user_prompt = "\n".join(
        [
            "Optimization problem input:",
            problem_spec or "Not available",
            "Required interface:",
            signature_line,
            "Structured NL constraints (id / name / desc / kind):",
            json.dumps(nl_constraints, ensure_ascii=False, indent=2),
            "Current create_model:",
            "```python",
            code,
            "```",
            "Return the corrected create_model. Code only.",
        ]
    )

    trace_input = {
        "agent": "audit_model",
        "upstream_artifacts": [
            {"label": "problem_input", "source": "problem_spec", "value": problem_spec},
            {"label": "required_interface", "source": "signature_line", "value": signature_line},
            {"label": "nl_constraints", "source": "components_nl", "value": nl_constraints},
            {"label": "previous_code", "source": "state.code.model_builder", "value": code},
        ],
    }

    try:
        audited = llm_client.code_generation_call(
            sys_prompt=PROMPTS["audit_model"]["system"],
            user_prompt=user_prompt,
            temperature=0.0,
            validate=True,
            trace_input=trace_input,
        )
    except Exception as exc:
        logger.warning("audit_model_llm_failed", error=str(exc))
        return state

    audited = _apply_create_model_autofixes(audited, required_signature=signature_line)
    valid, _ = _validate_create_model_entrypoint(audited, required_signature=signature_line)
    if not (valid and audited.strip()):
        logger.info("audit_model_rejected_invalid_output")
        return state

    if audited.strip() == code.strip():
        logger.info("audit_model_no_change")
        return state

    state.code.model_builder = CodeBlob(
        language="python",
        filename=model_builder.filename if model_builder else "create_model.py",
        source=audited,
    )
    repair_iterations = state.tests.setdefault("repair_iterations", {})
    repair_iterations["audit_model"] = int(repair_iterations.get("audit_model") or 0) + 1
    logger.info("audit_model_applied")
    return state
