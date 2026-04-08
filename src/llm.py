# modelpack/llm.py
import ast
import json
import inspect
import os
import sys
import time
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

import instructor
import structlog
from dotenv import load_dotenv
from litellm import completion as litellm_completion
from pydantic import BaseModel
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import (  # noqa: E402
    resolve_base_url,
    resolve_default_model,
    resolve_api_key,
    resolve_openrouter_extra_body,
    resolve_openrouter_headers,
    resolve_provider,
)

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)
_ACTIVE_LLM_TRACE: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar(
    "ACTIVE_LLM_TRACE",
    default=None,
)


class NonRetryableLLMError(RuntimeError):
    """Raised for deterministic LLM response issues that retries will not fix."""


DEFAULT_MODEL = resolve_default_model()
DEFAULT_LENGTH_RETRY_MAX_COMPLETION_TOKENS = 16384


def _env_retry_attempts(default: int = 3) -> int:
    raw_value = os.getenv("LLM_CLIENT_MAX_ATTEMPTS")
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
        configured_model_name = resolve_default_model(model_name)
        self.provider = resolve_provider(provider).strip().lower()
        self.model_name = configured_model_name or DEFAULT_MODEL
        self.structured_model_name = os.getenv("STRUCTURED_MODEL_NAME") or self.model_name
        self.code_generation_model_name = os.getenv("CODE_MODEL_NAME") or self.model_name
        self.base_url_override = base_url
        self.api_key_override = api_key
        self.max_completion_tokens = (
            _env_optional_positive_int("LLM_CLIENT_MAX_COMPLETION_TOKENS")
            or _env_optional_positive_int("LLM_CLIENT_MAX_TOKENS")
        )
        self.length_retry_max_completion_tokens = (
            _env_optional_positive_int("LLM_CLIENT_LENGTH_RETRY_MAX_COMPLETION_TOKENS")
            or DEFAULT_LENGTH_RETRY_MAX_COMPLETION_TOKENS
        )
        timeout_seconds = None
        timeout_raw = os.getenv("LLM_CLIENT_TIMEOUT_SECONDS")
        if timeout_raw:
            try:
                parsed_timeout = float(timeout_raw)
                if parsed_timeout > 0:
                    timeout_seconds = parsed_timeout
            except ValueError:
                timeout_seconds = None
        self.timeout_seconds = timeout_seconds

        self.client = instructor.from_litellm(litellm_completion, mode=instructor.Mode.JSON)

    def begin_trace(self) -> Token:
        return _ACTIVE_LLM_TRACE.set([])

    def end_trace(self, token: Token) -> Dict[str, Any]:
        calls = list(_ACTIVE_LLM_TRACE.get() or [])
        _ACTIVE_LLM_TRACE.reset(token)
        return {"calls": calls, "summary": self._summarize_calls(calls)}

    def trace_length(self) -> int:
        trace = _ACTIVE_LLM_TRACE.get()
        return len(trace) if trace is not None else 0

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

    def _extract_finish_reason(self, raw_response: Any) -> Optional[str]:
        if raw_response is None:
            return None
        if isinstance(raw_response, dict):
            choices = raw_response.get("choices")
            if isinstance(choices, list) and choices:
                finish_reason = choices[0].get("finish_reason")
                return str(finish_reason) if finish_reason is not None else None
            return None

        choices = getattr(raw_response, "choices", None)
        if not choices:
            return None
        finish_reason = getattr(choices[0], "finish_reason", None)
        return str(finish_reason) if finish_reason is not None else None

    def _extract_response_text(self, raw_response: Any) -> Optional[str]:
        if raw_response is None:
            return None
        if isinstance(raw_response, str):
            return raw_response
        if isinstance(raw_response, dict):
            choices = raw_response.get("choices")
            if isinstance(choices, list) and choices:
                message = choices[0].get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, list):
                        text_parts = [
                            str(item.get("text"))
                            for item in content
                            if isinstance(item, dict) and item.get("text") is not None
                        ]
                        if text_parts:
                            return "\n".join(text_parts)
                    if content is not None:
                        return str(content)
            text = raw_response.get("text")
            if text is not None:
                return str(text)
            candidates = raw_response.get("candidates")
            if isinstance(candidates, list) and candidates:
                first = candidates[0]
                if isinstance(first, dict):
                    content = first.get("content")
                    if isinstance(content, dict):
                        parts = content.get("parts")
                        if isinstance(parts, list):
                            text_parts = [
                                str(part.get("text"))
                                for part in parts
                                if isinstance(part, dict) and part.get("text") is not None
                            ]
                            if text_parts:
                                return "\n".join(text_parts)
            return None

        text = getattr(raw_response, "text", None)
        if text:
            return str(text)

        choices = getattr(raw_response, "choices", None)
        if choices:
            message = getattr(choices[0], "message", None)
            if message is not None:
                content = getattr(message, "content", None)
                if isinstance(content, list):
                    text_parts: List[str] = []
                    for item in content:
                        item_text = getattr(item, "text", None)
                        if item_text is not None:
                            text_parts.append(str(item_text))
                    if text_parts:
                        return "\n".join(text_parts)
                if content is not None:
                    return str(content)

        candidates = getattr(raw_response, "candidates", None)
        if candidates:
            first = candidates[0]
            content = getattr(first, "content", None)
            if content is not None:
                parts = getattr(content, "parts", None)
                if parts:
                    text_parts = [
                        str(getattr(part, "text"))
                        for part in parts
                        if getattr(part, "text", None) is not None
                    ]
                    if text_parts:
                        return "\n".join(text_parts)
        return None

    def _serialize_trace_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc).isoformat()
        if isinstance(value, dict):
            return {
                str(key): self._serialize_trace_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [self._serialize_trace_value(item) for item in value]

        model_dump_json = getattr(value, "model_dump_json", None)
        if callable(model_dump_json):
            try:
                return json.loads(model_dump_json(indent=2))
            except (TypeError, ValueError, json.JSONDecodeError):
                pass

        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                return self._serialize_trace_value(model_dump(mode="json"))
            except TypeError:
                return self._serialize_trace_value(model_dump())

        dict_fn = getattr(value, "dict", None)
        if callable(dict_fn):
            return self._serialize_trace_value(dict_fn())

        if hasattr(value, "__dict__"):
            return self._serialize_trace_value(
                {key: item for key, item in vars(value).items() if not key.startswith("_")}
            )

        return str(value)

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
        model_name: Optional[str] = None,
        response_model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        prompt_messages: Optional[List[Dict[str, Any]]] = None,
        raw_output_text: Optional[str] = None,
        extracted_output: Any = None,
        trace_input: Optional[Dict[str, Any]] = None,
    ) -> None:
        trace = _ACTIVE_LLM_TRACE.get()
        if trace is None:
            return

        usage_payload = self._extract_usage_payload(raw_response)
        normalized_usage = self._normalize_usage(usage_payload)
        finish_reason = self._extract_finish_reason(raw_response)
        trace.append(
            {
                "sequence": len(trace) + 1,
                "call_type": call_type,
                "caller": self._detect_caller(),
                "provider": self.provider,
                "model_name": model_name or self.model_name,
                "response_model": response_model,
                "temperature": float(temperature),
                "started_at": started_at.astimezone(timezone.utc).isoformat(),
                "latency_seconds": round(latency_seconds, 6),
                "success": success,
                "error": error,
                "finish_reason": finish_reason,
                "input_tokens": normalized_usage["input_tokens"],
                "output_tokens": normalized_usage["output_tokens"],
                "total_tokens": normalized_usage["total_tokens"],
                "usage": usage_payload,
                "prompt": {
                    "system": system_prompt,
                    "user": user_prompt,
                    "messages": self._serialize_trace_value(prompt_messages),
                },
                "raw_output_text": raw_output_text,
                "extracted_output": self._serialize_trace_value(extracted_output),
                "trace_input": self._serialize_trace_value(trace_input),
            }
        )

    def _normalize_chat_messages(
        self,
        *,
        sys_prompt: Optional[str],
        user_prompt: Optional[str],
        messages: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        if messages is not None:
            normalized_messages: List[Dict[str, Any]] = []
            for message in messages:
                role = str(message.get("role") or "").strip()
                content = message.get("content")
                if not role:
                    raise ValueError("Chat messages must include a non-empty role")
                if content is None:
                    raise ValueError("Chat messages must include content")
                normalized_messages.append({"role": role, "content": str(content)})
            if not normalized_messages:
                raise ValueError("Chat messages must be non-empty")
            return normalized_messages

        return [
            {"role": "system", "content": str(sys_prompt or "")},
            {"role": "user", "content": str(user_prompt or "")},
        ]

    def _build_completion_kwargs(
        self,
        *,
        model_name: str,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_completion_tokens: Optional[int],
    ) -> Dict[str, Any]:
        resolved_base_url = resolve_base_url(self.base_url_override, model=model_name)
        resolved_api_key = resolve_api_key(
            self.api_key_override,
            model=model_name,
            base_url=resolved_base_url,
        )
        extra_headers = (
            resolve_openrouter_headers()
            if resolved_base_url and "openrouter.ai" in resolved_base_url.lower()
            else None
        )
        extra_body = resolve_openrouter_extra_body(model=model_name, base_url=resolved_base_url)
        request_kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if max_completion_tokens is not None:
            request_kwargs["max_completion_tokens"] = max_completion_tokens
        if resolved_api_key:
            request_kwargs["api_key"] = resolved_api_key
        if resolved_base_url:
            request_kwargs["base_url"] = resolved_base_url
        if extra_headers:
            request_kwargs["extra_headers"] = dict(extra_headers)
        if extra_body:
            request_kwargs["extra_body"] = dict(extra_body)
        if self.timeout_seconds is not None:
            request_kwargs["timeout"] = self.timeout_seconds
        return request_kwargs

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
        retry=retry_if_not_exception_type(NonRetryableLLMError),
        stop=stop_after_attempt(_env_retry_attempts()),
        wait=wait_exponential(multiplier=1, min=2, max=60),
    )
    def structured_call(
        self,
        sys_prompt: str,
        user_prompt: str,
        pyd_model: Type[T],
        temperature: float = 0.0,
        trace_input: Optional[Dict[str, Any]] = None,
    ) -> T:
        """Generate structured output using any LLM provider."""
        started_at = datetime.now(timezone.utc)
        started_perf = time.perf_counter()
        result: Optional[T] = None
        try:
            request_kwargs = self._build_completion_kwargs(
                model_name=self.structured_model_name,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_completion_tokens=self.max_completion_tokens,
            )
            request_kwargs["response_model"] = pyd_model
            result = self.client.chat.completions.create(**request_kwargs)

            self._record_call(
                call_type="structured",
                raw_response=getattr(result, "_raw_response", None),
                started_at=started_at,
                latency_seconds=time.perf_counter() - started_perf,
                success=True,
                error=None,
                temperature=temperature,
                model_name=self.structured_model_name,
                response_model=getattr(pyd_model, "__name__", str(pyd_model)),
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
                raw_output_text=self._extract_response_text(getattr(result, "_raw_response", None)),
                extracted_output=result,
                trace_input=trace_input,
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
                model_name=self.structured_model_name,
                response_model=getattr(pyd_model, "__name__", str(pyd_model)),
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
                raw_output_text=self._extract_response_text(
                    getattr(result, "_raw_response", None) if result is not None else None
                ),
                extracted_output=result,
                trace_input=trace_input,
            )
            logger.error("structured_call_error", provider=self.provider, error=str(e))
            raise

    @retry(
        retry=retry_if_not_exception_type(NonRetryableLLMError),
        stop=stop_after_attempt(_env_retry_attempts()),
        wait=wait_exponential(multiplier=1, min=2, max=60),
    )
    def code_generation_call(
        self,
        sys_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        temperature: float = 0.0,
        validate: bool = True,
        messages: Optional[List[Dict[str, Any]]] = None,
        trace_input: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate code with optional validation."""
        started_at = datetime.now(timezone.utc)
        started_perf = time.perf_counter()
        response: Any = None
        raw_output_text: Optional[str] = None
        request_messages = self._normalize_chat_messages(
            sys_prompt=sys_prompt,
            user_prompt=user_prompt,
            messages=messages,
        )
        request_system_prompt = None
        request_user_prompt = None
        if messages is None:
            request_system_prompt = str(sys_prompt or "")
            request_user_prompt = str(user_prompt or "")
        try:
            request_kwargs = self._build_completion_kwargs(
                model_name=self.code_generation_model_name,
                messages=request_messages,
                temperature=temperature,
                max_completion_tokens=(
                    self.max_completion_tokens or self.length_retry_max_completion_tokens
                ),
            )
            response = litellm_completion(**request_kwargs)
            code = self._extract_response_text(response)
            if code is None:
                raise NonRetryableLLMError("model returned empty content")
            raw_output_text = code

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
                model_name=self.code_generation_model_name,
                system_prompt=request_system_prompt,
                user_prompt=request_user_prompt,
                prompt_messages=request_messages,
                raw_output_text=raw_output_text or self._extract_response_text(response),
                extracted_output=code,
                trace_input=trace_input,
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
                model_name=self.code_generation_model_name,
                system_prompt=request_system_prompt,
                user_prompt=request_user_prompt,
                prompt_messages=request_messages,
                raw_output_text=raw_output_text or self._extract_response_text(response),
                extracted_output=None,
                trace_input=trace_input,
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
