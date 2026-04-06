# Efficient Modeling MAS

A benchmark-focused multi-agent system that converts optimization problems into executable Pyomo code.

## Overview

This repo keeps the benchmarked paths in active use:
1. Parse natural language problem descriptions
2. Extract mathematical components (sets, parameters, variables, constraints)
3. Generate LaTeX formulations
4. Produce working Pyomo optimization code
5. Validate solutions through cross-checking

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and set the LLM settings there:

```env
MODEL_NAME=gpt-4.1
API_KEY=your-api-key
LLM_CLIENT_TIMEOUT_SECONDS=120
# Optional: stronger codegen-specific override
# CODE_MODEL_NAME=gpt-4.1
# Optional: OpenRouter / local endpoint
# BASE_URL=https://openrouter.ai/api/v1
# Optional provider-specific keys
# OPENROUTER_API_KEY=
# GEMINI_API_KEY=
```

The OR_MAS client now reads model name, base URL, API key, and provider-specific overrides
from the repo root `.env`.

## Usage

```bash
# From text
python -m src "Minimize cost of shipping goods from 3 warehouses to 5 customers..."

# From file
python -m src problem.txt

# Interactive mode
python -m src
```

## Architecture

```
Full graph:
specify_problem  → Contract + NL components
derive_math      → Convert to LaTeX
build_model      → Generate model code
generate_data    → Generate test data
screen_data      → Feasibility testing
check_solution   → Solution validator
solve_model      → Execute optimization
judge_solution   → Cross-validate results

Single-agent baseline:
Problem input   → create_model directly
```

The full graph keeps the `screen_data` and `judge_solution` feedback loops that are exercised by the benchmark.

## License

MIT
