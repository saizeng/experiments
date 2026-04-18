# LangGraph Multi Source Router Scaffold

This package contains a LangGraph based scaffold for routing a user question across two chatbot backed data sources:

1. Structured deal and client data
2. Unstructured banking documents for IPO, M&A, BizDev, and Debt workflows

## Files

- `langgraph_multi_source_router.py`

## What the graph does

The graph contains these nodes:

- `plan`
- `validate_plan`
- `execute_step`
- `evaluate_progress`
- `synthesize_answer`

It uses an LLM only for planning and final synthesis.
Execution remains deterministic.

## Installation

```bash
pip install langgraph
```

## Run

```bash
python langgraph_multi_source_router.py
```

## Replace for production

You should replace:

- `FakeLLM` with your real LLM SDK wrapper
- `StructuredChatbotClient` with your structured data chatbot API
- `UnstructuredChatbotClient` with your unstructured document chatbot API

## Recommended production additions

- strict structured output validation using your LLM provider schema support
- entity resolution for clients, deals, aliases, and time ranges
- richer confidence and sufficiency checks
- retries and fallback when one source returns poor coverage
- execution logging and trace IDs
- optional human review for high impact workflows
