# modelpack/schemas.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Optional, Any, Dict, List, Tuple
from datetime import datetime
import uuid

# ---- NL Components ----
class NLItem(BaseModel):
    id: str
    name: str
    desc: str
    var_type: Optional[Literal["integer", "continuous", "binary"]] = None  # For variables

class ComponentsNL(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sets: List[NLItem] = Field(default_factory=list)
    parameters: List[NLItem] = Field(default_factory=list)
    variables: List[NLItem] = Field(default_factory=list)
    constraints_basic: List[NLItem] = Field(default_factory=list)
    constraints_logical: List[NLItem] = Field(default_factory=list)
    constraints_aux: List[NLItem] = Field(default_factory=list)
    objective: Optional[NLItem] = None

class NLMetaEdit(BaseModel):
    op: Literal["keep", "drop", "modify", "add"]
    target_id: Optional[str] = None
    rationale: str

class ComponentsNLMETA(BaseModel):
    edits: List[NLMetaEdit] = Field(default_factory=list)

# ---- Math Components ----
class SymbolDef(BaseModel):
    sym: str
    maps_to: str
    domain: Optional[str] = None
    bounds: Optional[str] = None
    desc: Optional[str] = None
    var_type: Optional[Literal["integer", "continuous", "binary"]] = None  # For variables

class MathConstraint(BaseModel):
    maps_to: str
    latex: str
    type: Literal["basic", "logical", "aux"]

class ComponentsMATH(BaseModel):
    model_config = ConfigDict(extra="forbid")
    indices: List[SymbolDef] = Field(default_factory=list)
    parameters: List[SymbolDef] = Field(default_factory=list)
    variables: List[SymbolDef] = Field(default_factory=list)
    constraints: List[MathConstraint] = Field(default_factory=list)
    objective: Optional[SymbolDef] = None
    sense: Optional[Literal["min", "max"]] = None

# ---- Data Extraction ----
class ExtractedData(BaseModel):
    """Concrete data values extracted from NL problem."""
    sets: Dict[str, List[Any]] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    derived_values: Dict[str, Any] = Field(default_factory=dict)

# ---- Code Artifacts ----
class CodeBlob(BaseModel):
    language: Literal["python"] = "python"
    filename: str
    source: str

class CodePack(BaseModel):
    model_builder: Optional[CodeBlob] = None
    data_schema: Optional[CodeBlob] = None
    datagen: Optional[CodeBlob] = None
    solution_checker: Optional[CodeBlob] = None

# ---- Test Results ----
class TestInstance(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: str
    data_dict: Dict[str, Any]
    solution_dict: Optional[Dict[str, Any]] = None
    solver_status: Optional[str] = None
    feasible: Optional[bool] = None
    objective_value: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class Feedback(BaseModel):
    source_agent: str
    target_agent: str
    issue: Literal[
        "data_infeasible", "checker_false_negative", "code_build_error",
        "domain_mismatch", "schema_violation", "math_inconsistency",
        "pyomo_build_error", "type_mismatch"
    ]
    evidence: Dict[str, Any]
    proposed_fix: Optional[str] = None
    retry_count: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)

# ---- Main State ----
class ModelPack(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    context: Dict[str, Any] = Field(default_factory=lambda: {
        "nl_problem": "",
        "assumptions": [],
        "units": [],
        "objective_sense": None
    })
    components_nl: Optional[ComponentsNL] = None
    components_nl_meta: Optional[ComponentsNLMETA] = None
    components_math: Optional[ComponentsMATH] = None
    extracted_data: Optional[ExtractedData] = None  # NEW
    code: CodePack = Field(default_factory=CodePack)
    tests: Dict[str, Any] = Field(default_factory=lambda: {
        "instances": [],
        "logs": [],
        "last_feedback": None,
        "retry_counts": {}  # Track retries per agent
    })
    status: str = "initialized"

# ---- Context for A0 ----
class ContextContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    assumptions: List[str] = Field(default_factory=list)
    units: List[str] = Field(default_factory=list)
    objective_sense: Literal["min", "max"]
    scope: str
    deliverables: List[str] = Field(default_factory=list)
