# modelpack/llm.py
import os
import ast
import instructor
from openai import OpenAI
import google.generativeai as genai
from pydantic import BaseModel
from typing import Type, TypeVar, Optional
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()
logger = structlog.get_logger(__name__)

T = TypeVar('T', bound=BaseModel)

class LLMClient:
    """Unified LLM client supporting multiple providers via instructor."""

    def __init__(
        self,
        provider: str = None,
        model_name: str = None,
        api_key: str = None,
        base_url: str = None
    ):
        self.provider = provider or os.getenv("PROVIDER", "openai")
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("API_KEY")
        self.base_url = base_url or os.getenv("BASE_URL")

        # Initialize client based on provider
        if self.provider == "gemini":
            genai.configure(api_key=self.api_key)
            base_client = genai.GenerativeModel(self.model_name or "gemini-1.5-pro")
            self.client = instructor.from_gemini(
                client=base_client,
                mode=instructor.Mode.GEMINI_JSON
            )
        else:
            # OpenAI-compatible (OpenAI, DeepSeek, Qwen, local)
            base_client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            self.client = instructor.from_openai(
                base_client,
                mode=instructor.Mode.JSON
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=60)
    )
    def structured_call(
        self,
        sys_prompt: str,
        user_prompt: str,
        pyd_model: Type[T],
        temperature: float = 0.7
    ) -> T:
        """Generate structured output using any LLM provider."""
        try:
            if self.provider == "gemini":
                combined_prompt = f"{sys_prompt}\n\n{user_prompt}"
                result = self.client.generate_content(
                    combined_prompt,
                    response_model=pyd_model,
                    temperature=temperature
                )
            else:
                result = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_model=pyd_model,
                    temperature=temperature
                )

            logger.info("structured_call_success", provider=self.provider)
            return result

        except Exception as e:
            logger.error("structured_call_error", provider=self.provider, error=str(e))
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=60)
    )
    def code_generation_call(
        self,
        sys_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        validate: bool = True
    ) -> str:
        """Generate code with optional validation."""
        try:
            if self.provider == "gemini":
                model = genai.GenerativeModel(self.model_name or "gemini-1.5-pro")
                response = model.generate_content(
                    f"{sys_prompt}\n\n{user_prompt}",
                    generation_config=genai.GenerationConfig(temperature=temperature)
                )
                code = response.text
            else:
                client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature
                )
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

            return code

        except Exception as e:
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
        code = code.replace('"Optional[List[str]] = None,', 'Optional[List[str]] = None,')
        code = code.replace('"Optional[List[str]] = None)', 'Optional[List[str]] = None)')

        # Fix f-string issues
        lines = code.split('\n')
        fixed_lines = []
        for line in lines:
            if 'f"' in line and '\\' in line:
                # Replace backslashes in f-strings
                line = line.replace('\\n', '" + "\\n" + "')
            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

# Global client
llm_client = LLMClient()
