import json
from openai import OpenAI
from skill_loader import load_skill_registry
from runner import run_python_script

client = OpenAI()

SKILL_REGISTRY = load_skill_registry("skills")

SYSTEM_PROMPT = f"""
You are a local skill executor.

When work requires PDFs or files:
- choose the correct skill
- call run_python_script
- scripts run inside workdir/

Skill documentation:

{SKILL_REGISTRY}
"""

TOOLS = [
    {
        "type": "function",
        "name": "run_python_script",
        "description": "Execute a python script inside skills/<skill>/scripts/",
        "parameters": {
            "type": "object",
            "properties": {
                "skill": {"type": "string"},
                "script": {"type": "string"},
                "args": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["skill", "script"]
        }
    }
]


def dispatch_tool(name, arguments):
    args = json.loads(arguments)
    if name == "run_python_script":
        return json.dumps(run_python_script(**args))
    raise ValueError("Unknown tool")


def run_agent(prompt: str):
    resp = client.responses.create(
        model="gpt-5",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        tools=TOOLS
    )

    while True:
        calls = [o for o in resp.output if o.type == "function_call"]
        if not calls:
            return resp.output_text

        outputs = []
        for call in calls:
            result = dispatch_tool(call.name, call.arguments)
            outputs.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": result
            })

        resp = client.responses.create(
            model="gpt-5",
            input=resp.output + outputs,
            tools=TOOLS
        )


if __name__ == "__main__":
    while True:
        q = input(">>> ")
        print(run_agent(q))
