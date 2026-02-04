# Efficient Modeling MAS

A multi-agent system that converts natural language optimization problems into executable Pyomo code.

## Overview

This system uses a 10-agent pipeline to automatically:
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
MODEL_NAME=gpt-4o-mini
API_KEY=your-api-key
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
A0 Specifier    → Extract problem contract
A1 Extractor    → Identify components from NL
A2 Reviser      → Reflexion-based cleanup
A3 Mathifier    → Convert to LaTeX
A3B DataExtract → Extract numerical values
A3C Schema      → Generate Data classes
A4 Pyomo        → Generate model builder
A5 DataGen      → Generate data functions
A6 Screen       → Feasibility testing
A7 Checker      → Solution validator
A8 Solver       → Execute optimization
A9 Judge        → Cross-validate results
```

Feedback loops route errors back to appropriate agents for automatic correction.

## License

MIT
