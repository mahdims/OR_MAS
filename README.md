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

Copy `.env.example` to `.env` and set your API key:

```env
PROVIDER=openai          # or: gemini, deepseek, qwen
MODEL_NAME=gpt-4.1
API_KEY=your-api-key
# Optional: stronger codegen-specific override
# CODE_MODEL_NAME=gpt-4.1
```

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
A0+A1 Frontend  → Contract + NL components
A3 Mathifier    → Convert to LaTeX
A4 Pyomo        → Generate model code
A5 DataGen      → Generate test data
A6 Screen       → Feasibility testing
A7 Checker      → Solution validator
A8 Solver       → Execute optimization
A9 Judge        → Cross-validate results

Single-agent baseline:
Problem input   → create_model directly
```

The full graph keeps the A6/A9 feedback loops that are exercised by the benchmark.

## License

MIT
