# modelpack/llm.py
import ast
import inspect
import os
import time
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type, TypeVar

import google.generativeai as genai
import instructor
import structlog
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()
logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)
_ACTIVE_LLM_TRACE: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar(
    "ACTIVE_LLM_TRACE",
    default=None,
)


def _env_retry_attempts(default: int = 3) -> int:
    raw_value = os.getenv("OPENAI_CLIENT_MAX_ATTEMPTS")
    if not raw_value:
        return default
    try:
        parsed_value = int(raw_value)
    except ValueError:
        return default
    return parsed_value if parsed_value > 0 else default


def _env_optional_positive_int(name: str) -> Optional[int]:
    raw_value = os.getenv(name)
    if not raw_value:
        return None
    try:
        parsed_value = int(raw_value)
    except ValueError:
        return None
    return parsed_value if parsed_value > 0 else None


class LLMClient:
    """Unified LLM client supporting multiple providers via instructor."""

    def __init__(
        self,
        provider: str = None,
        model_name: str = None,
        api_key: str = None,
        base_url: str = None,
    ):
        self.model_name = (
            model_name or os.getenv("MODEL_NAME") or os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
        )
        resolved_provider = provider or os.getenv("PROVIDER")
        if not resolved_provider:
            resolved_provider = "gemini" if "gemini" in self.model_name.lower() else "openai"
        self.provider = resolved_provider.lower().replace("gemeni", "gemini")
        self.base_url = base_url or os.getenv("BASE_URL")
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.max_completion_tokens = (
            _env_optional_positive_int("OPENAI_CLIENT_MAX_COMPLETION_TOKENS")
            or _env_optional_positive_int("OPENAI_CLIENT_MAX_TOKENS")
        )
        reasoning_effort = os.getenv("OPENAI_CLIENT_REASONING_EFFORT")
        reasoning_exclude = os.getenv("OPENAI_CLIENT_REASONING_EXCLUDE")
        self.openai_extra_body: Optional[Dict[str, Any]] = None
        if reasoning_effort or reasoning_exclude:
            reasoning_config: Dict[str, Any] = {}
            if reasoning_effort:
                reasoning_config["effort"] = reasoning_effort
            if reasoning_exclude:
                reasoning_config["exclude"] = reasoning_exclude.strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "y",
                }
            if reasoning_config:
                self.openai_extra_body = {"reasoning": reasoning_config}

        # Initialize client based on provider
        if self.provider == "gemini":
            self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
            genai.configure(api_key=self.api_key)
            base_client = genai.GenerativeModel(self.model_name or "gemini-1.5-pro")
            self.raw_client = base_client
            self.client = instructor.from_gemini(
                client=base_client, mode=instructor.Mode.GEMINI_JSON
            )
        else:
            if self.base_url and "openrouter.ai" in self.base_url.lower():
                self.api_key = (
                    api_key
                    or openrouter_api_key
                    or os.getenv("OPENAI_API_KEY")
                    or os.getenv("API_KEY")
                )
            else:
                self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
            timeout_seconds = None
            timeout_raw = os.getenv("OPENAI_CLIENT_TIMEOUT_SECONDS")
            if timeout_raw:
                try:
                    parsed_timeout = float(timeout_raw)
                    if parsed_timeout > 0:
                        timeout_seconds = parsed_timeout
                except ValueError:
                    timeout_seconds = None
            # OpenAI-compatible (OpenAI, DeepSeek, Qwen, local)
            base_client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=timeout_seconds,
            )
            self.raw_client = base_client
            self.client = instructor.from_openai(base_client, mode=instructor.Mode.JSON)

    def begin_trace(self) -> Token:
        return _ACTIVE_LLM_TRACE.set([])

    def end_trace(self, token: Token) -> Dict[str, Any]:
        calls = list(_ACTIVE_LLM_TRACE.get() or [])
        _ACTIVE_LLM_TRACE.reset(token)
        return {"calls": calls, "summary": self._summarize_calls(calls)}

    def _detect_caller(self) -> str:
        frame = inspect.currentframe()
        fallback = "unknown"
        try:
            current = frame.f_back if frame is not None else None
            while current is not None:
                module_name = str(current.f_globals.get("__name__") or "")
                function_name = current.f_code.co_name
                label = f"{module_name}.{function_name}" if module_name else function_name
                if ".agents." in module_name:
                    return label
                if module_name != __name__ and not module_name.startswith("tenacity"):
                    fallback = label
                current = current.f_back
        finally:
            del frame
        return fallback

    def _usage_to_dict(self, usage_obj: Any) -> Optional[Dict[str, Any]]:
        if usage_obj is None:
            return None
        if isinstance(usage_obj, dict):
            return dict(usage_obj)

        model_dump = getattr(usage_obj, "model_dump", None)
        if callable(model_dump):
            try:
                payload = model_dump(exclude_none=True)
            except TypeError:
                payload = model_dump()
            if isinstance(payload, dict):
                return payload

        to_dict = getattr(usage_obj, "dict", None)
        if callable(to_dict):
            payload = to_dict()
            if isinstance(payload, dict):
                return payload

        if hasattr(usage_obj, "__dict__"):
            payload = {
                key: value for key, value in vars(usage_obj).items() if not key.startswith("_")
            }
            if payload:
                return payload
        return None

    def _extract_usage_payload(self, raw_response: Any) -> Optional[Dict[str, Any]]:
        if raw_response is None:
            return None
        if isinstance(raw_response, dict):
            usage_obj = raw_response.get("usage") or raw_response.get("usage_metadata")
            return self._usage_to_dict(usage_obj)

        usage_obj = getattr(raw_response, "usage", None)
        if usage_obj is None:
            usage_obj = getattr(raw_response, "usage_metadata", None)
        return self._usage_to_dict(usage_obj)

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _normalize_usage(self, usage_payload: Optional[Dict[str, Any]]) -> Dict[str, Optional[int]]:
        usage_payload = usage_payload or {}
        input_tokens = self._safe_int(
            usage_payload.get("prompt_tokens")
            or usage_payload.get("input_tokens")
            or usage_payload.get("prompt_token_count")
        )
        output_tokens = self._safe_int(
            usage_payload.get("completion_tokens")
            or usage_payload.get("output_tokens")
            or usage_payload.get("candidates_token_count")
        )
        total_tokens = self._safe_int(
            usage_payload.get("total_tokens") or usage_payload.get("total_token_count")
        )
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    def _record_call(
        self,
        *,
        call_type: str,
        raw_response: Any,
        started_at: datetime,
        latency_seconds: float,
        success: bool,
        error: Optional[str],
        temperature: float,
        response_model: Optional[str] = None,
    ) -> None:
        trace = _ACTIVE_LLM_TRACE.get()
        if trace is None:
            return

        usage_payload = self._extract_usage_payload(raw_response)
        normalized_usage = self._normalize_usage(usage_payload)
        trace.append(
            {
                "sequence": len(trace) + 1,
                "call_type": call_type,
                "caller": self._detect_caller(),
                "provider": self.provider,
                "model_name": self.model_name,
                "response_model": response_model,
                "temperature": float(temperature),
                "started_at": started_at.astimezone(timezone.utc).isoformat(),
                "latency_seconds": round(latency_seconds, 6),
                "success": success,
                "error": error,
                "input_tokens": normalized_usage["input_tokens"],
                "output_tokens": normalized_usage["output_tokens"],
                "total_tokens": normalized_usage["total_tokens"],
                "usage": usage_payload,
            }
        )

    def _summarize_calls(self, calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "call_count": len(calls),
            "successful_calls": 0,
            "failed_calls": 0,
            "calls_with_usage": 0,
            "calls_without_usage": 0,
            "structured_calls": 0,
            "code_generation_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "total_latency_seconds": 0.0,
            "avg_latency_seconds": None,
        }
        for call in calls:
            if call.get("success"):
                summary["successful_calls"] += 1
            else:
                summary["failed_calls"] += 1

            if call.get("call_type") == "structured":
                summary["structured_calls"] += 1
            elif call.get("call_type") == "code_generation":
                summary["code_generation_calls"] += 1

            latency_seconds = call.get("latency_seconds")
            if isinstance(latency_seconds, (int, float)):
                summary["total_latency_seconds"] += float(latency_seconds)

            total_tokens = self._safe_int(call.get("total_tokens"))
            input_tokens = self._safe_int(call.get("input_tokens"))
            output_tokens = self._safe_int(call.get("output_tokens"))
            if total_tokens is not None or input_tokens is not None or output_tokens is not None:
                summary["calls_with_usage"] += 1
            else:
                summary["calls_without_usage"] += 1

            summary["input_tokens"] += input_tokens or 0
            summary["output_tokens"] += output_tokens or 0
            summary["total_tokens"] += total_tokens or 0

        if calls:
            summary["total_latency_seconds"] = round(summary["total_latency_seconds"], 6)
            summary["avg_latency_seconds"] = round(
                summary["total_latency_seconds"] / len(calls),
                6,
            )
        return summary

    @retry(
        stop=stop_after_attempt(_env_retry_attempts()),
        wait=wait_exponential(multiplier=1, min=2, max=60),
    )
    def structured_call(
        self, sys_prompt: str, user_prompt: str, pyd_model: Type[T], temperature: float = 0.0
    ) -> T:
        """Generate structured output using any LLM provider."""
        started_at = datetime.now(timezone.utc)
        started_perf = time.perf_counter()
        result: Optional[T] = None
        try:
            if self.provider == "gemini":
                result = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_model=pyd_model,
                )
            else:
                request_kwargs: Dict[str, Any] = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "response_model": pyd_model,
                    "temperature": temperature,
                }
                if self.max_completion_tokens is not None:
                    request_kwargs["max_completion_tokens"] = self.max_completion_tokens
                if self.openai_extra_body is not None:
                    request_kwargs["extra_body"] = self.openai_extra_body
                result = self.client.chat.completions.create(**request_kwargs)

            self._record_call(
                call_type="structured",
                raw_response=getattr(result, "_raw_response", None),
                started_at=started_at,
                latency_seconds=time.perf_counter() - started_perf,
                success=True,
                error=None,
                temperature=temperature,
                response_model=getattr(pyd_model, "__name__", str(pyd_model)),
            )
            logger.info("structured_call_success", provider=self.provider)
            return result

        except Exception as e:
            self._record_call(
                call_type="structured",
                raw_response=getattr(result, "_raw_response", None) if result is not None else None,
                started_at=started_at,
                latency_seconds=time.perf_counter() - started_perf,
                success=False,
                error=str(e),
                temperature=temperature,
                response_model=getattr(pyd_model, "__name__", str(pyd_model)),
            )
            logger.error("structured_call_error", provider=self.provider, error=str(e))
            raise

    @retry(
        stop=stop_after_attempt(_env_retry_attempts()),
        wait=wait_exponential(multiplier=1, min=2, max=60),
    )
    def code_generation_call(
        self, sys_prompt: str, user_prompt: str, temperature: float = 0.0, validate: bool = True
    ) -> str:
        """Generate code with optional validation."""
        started_at = datetime.now(timezone.utc)
        started_perf = time.perf_counter()
        response: Any = None
        try:
            if self.provider == "gemini":
                response = self.raw_client.generate_content(
                    f"{sys_prompt}\n\n{user_prompt}",
                    generation_config=genai.GenerationConfig(temperature=temperature),
                )
                code = response.text
            else:
                request_kwargs: Dict[str, Any] = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                }
                if self.max_completion_tokens is not None:
                    request_kwargs["max_completion_tokens"] = self.max_completion_tokens
                if self.openai_extra_body is not None:
                    request_kwargs["extra_body"] = self.openai_extra_body
                response = self.raw_client.chat.completions.create(**request_kwargs)
                code = response.choices[0].message.content

            # Extract from markdown
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()

            # Add missing imports
            code = self._fix_imports(code)

            # Validate if requested
            if validate:
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    logger.warning("code_syntax_error", error=str(e))
                    # Try to fix common issues
                    code = self._fix_common_syntax_errors(code)
                    ast.parse(code)  # Re-validate

            self._record_call(
                call_type="code_generation",
                raw_response=response,
                started_at=started_at,
                latency_seconds=time.perf_counter() - started_perf,
                success=True,
                error=None,
                temperature=temperature,
            )
            return code

        except Exception as e:
            self._record_call(
                call_type="code_generation",
                raw_response=response,
                started_at=started_at,
                latency_seconds=time.perf_counter() - started_perf,
                success=False,
                error=str(e),
                temperature=temperature,
            )
            logger.error("code_generation_error", error=str(e))
            raise

    def _fix_imports(self, code: str) -> str:
        """Add missing imports based on usage."""
        imports_needed = set()

        # Check what's used
        if any(x in code for x in ["List[", ": List", "-> List"]):
            imports_needed.add("List")
        if any(x in code for x in ["Dict[", ": Dict", "-> Dict"]):
            imports_needed.add("Dict")
        if any(x in code for x in ["Optional[", ": Optional"]):
            imports_needed.add("Optional")
        if any(x in code for x in ["Tuple[", ": Tuple"]):
            imports_needed.add("Tuple")
        if "Any" in code:
            imports_needed.add("Any")

        # Add typing imports if needed
        if imports_needed and "from typing import" not in code:
            import_line = f"from typing import {', '.join(sorted(imports_needed))}\n"
            code = import_line + code

        # Add numpy if needed
        if "np." in code and "import numpy" not in code:
            code = "import numpy as np\n" + code

        # Add pyomo if needed
        if "pyo." in code and "import pyomo" not in code:
            code = "import pyomo.environ as pyo\n" + code

        return code

    def _fix_common_syntax_errors(self, code: str) -> str:
        """Fix common syntax errors in generated code."""
        # Fix string quotes in type hints
        code = code.replace('"Optional[List[str]] = None,', "Optional[List[str]] = None,")
        code = code.replace('"Optional[List[str]] = None)', "Optional[List[str]] = None)")

        # Fix f-string issues
        lines = code.split("\n")
        fixed_lines = []
        for line in lines:
            if 'f"' in line and "\\" in line:
                # Replace backslashes in f-strings
                line = line.replace("\\n", '" + "\\n" + "')
            fixed_lines.append(line)

        return "\n".join(fixed_lines)


# Global client
llm_client = LLMClient()
