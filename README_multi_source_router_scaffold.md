# Multi-source router scaffold

Files:
- `multi_source_router_scaffold.py`

What it includes:
- LLM-based planning step that outputs a routing plan
- Validation layer for plan guardrails
- Router that executes structured and unstructured chatbot calls
- Simple answer synthesis step
- `FakeLLM` and dummy chatbot clients for local testing

How to run:

```bash
python multi_source_router_scaffold.py
```

What to replace for production:
- `LLMClient.complete_json`
- `StructuredChatbotClient.ask`
- `UnstructuredChatbotClient.ask`
- sufficiency logic in `Router._is_sufficient`
- optional stronger entity resolution and retries
