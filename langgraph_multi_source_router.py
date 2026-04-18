from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, TypedDict

try:
    from langgraph.graph import StateGraph, END
except ImportError as e:
    raise ImportError(
        "langgraph is required for this example. Install with: pip install langgraph"
    ) from e


# ---------------------------
# Enums and Plan Schema
# ---------------------------

class Source(str, Enum):
    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"


class Intent(str, Enum):
    LOOKUP = "lookup"
    SUMMARY = "summary"
    COMPARISON = "comparison"
    DIAGNOSIS = "diagnosis"
    BRIEFING = "briefing"
    DISCOVERY = "discovery"
    RECOMMENDATION = "recommendation"
    OTHER = "other"


class OutputShape(str, Enum):
    FACT = "fact"
    TABLE = "table"
    BULLETS = "bullets"
    MEMO = "memo"
    RANKED_LIST = "ranked_list"
    OTHER = "other"


class EvidenceType(str, Enum):
    NUMERIC = "numeric"
    NARRATIVE = "narrative"
    DOCUMENT_GROUNDING = "document_grounding"
    AGGREGATION = "aggregation"
    ENTITY_DISCOVERY = "entity_discovery"
    TEMPORAL = "temporal"


class StepGoal(str, Enum):
    FILTER_CANDIDATES = "filter_candidates"
    DISCOVER_THEMES = "discover_themes"
    FETCH_FACTS = "fetch_facts"
    FETCH_DOCUMENTS = "fetch_documents"
    ENRICH = "enrich"
    SYNTHESIZE = "synthesize"


@dataclass
class PlanStep:
    step_id: int
    source: Source
    goal: StepGoal
    instruction: str
    stop_if_sufficient: bool = False


@dataclass
class RoutingPlan:
    user_question: str
    intent: Intent
    output_shape: OutputShape
    entities: Dict[str, Any]
    evidence_types: List[EvidenceType]
    source_required: Literal["structured", "unstructured", "both"]
    order: Literal["structured_first", "unstructured_first", "parallel", "single"]
    rationale: str
    confidence: float
    steps: List[PlanStep] = field(default_factory=list)


@dataclass
class SourceResult:
    source: Source
    success: bool
    confidence: float
    payload: Dict[str, Any]
    citations: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


class GraphState(TypedDict, total=False):
    question: str
    plan: RoutingPlan
    current_step_index: int
    results: List[SourceResult]
    final_answer: str
    stop_early: bool
    debug_log: List[str]


# ---------------------------
# LLM Client Interface
# Replace with your provider SDK
# ---------------------------

class LLMClient:
    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        raise NotImplementedError

    def complete_text(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


# ---------------------------
# Source Adapters
# Replace these with real chatbot API clients
# ---------------------------

class StructuredChatbotClient:
    def ask(self, question: str, context: Optional[Dict[str, Any]] = None) -> SourceResult:
        dummy_payload = {
            "answer": f"[Structured bot answer] {question}",
            "resolved_entities": context or {},
            "records_found": 5,
            "sample_records": ["deal_123", "deal_456"],
        }
        return SourceResult(
            source=Source.STRUCTURED,
            success=True,
            confidence=0.88,
            payload=dummy_payload,
            citations=[{"type": "record_ids", "ids": ["deal_123", "client_456"]}],
        )


class UnstructuredChatbotClient:
    def ask(self, question: str, context: Optional[Dict[str, Any]] = None) -> SourceResult:
        dummy_payload = {
            "answer": f"[Unstructured bot answer] {question}",
            "resolved_entities": context or {},
            "documents_found": 7,
            "sample_documents": ["doc_a", "doc_b"],
        }
        return SourceResult(
            source=Source.UNSTRUCTURED,
            success=True,
            confidence=0.83,
            payload=dummy_payload,
            citations=[{"type": "document_ids", "ids": ["doc_a", "doc_b"]}],
        )


# ---------------------------
# Prompts
# ---------------------------

PLANNER_SYSTEM_PROMPT = """
You are a routing planner for an enterprise banking assistant.

Your only job is to decide how to route a user question across two data sources:
1. structured deal and client data
2. unstructured documents used in IPO, M&A, BizDev, and Debt workflows

You must not answer the user question.
You must only return a routing plan as strict JSON.

Rules:
- Prefer a single source when sufficient.
- Use structured first for ranking, counts, filters, wallet, revenue, expense, interactions, pipeline, trends.
- Use unstructured first for summaries, reasons, risks, themes, notes, documents, precedent language, what was said.
- Use both if the question needs numeric facts plus document context, or entity discovery plus quantification.
- Sequential execution is preferred over parallel unless both are obviously needed and entity mapping is already explicit.
- If the first step is sufficient, allow stopping early.

Return strict JSON with this shape:
{
  "intent": "lookup|summary|comparison|diagnosis|briefing|discovery|recommendation|other",
  "output_shape": "fact|table|bullets|memo|ranked_list|other",
  "entities": {
    "clients": [],
    "deals": [],
    "products": [],
    "time_range": null,
    "other_filters": {}
  },
  "evidence_types": [],
  "source_required": "structured|unstructured|both",
  "order": "structured_first|unstructured_first|parallel|single",
  "rationale": "short explanation",
  "confidence": 0.0,
  "steps": [
    {
      "step_id": 1,
      "source": "structured|unstructured",
      "goal": "filter_candidates|discover_themes|fetch_facts|fetch_documents|enrich|synthesize",
      "instruction": "what to do",
      "stop_if_sufficient": true
    }
  ]
}
"""

ANSWER_SYSTEM_PROMPT = """
You are an enterprise banking assistant.
Use the execution results to answer the user's question.

Rules:
- Separate structured facts from document-based narrative.
- Do not invent facts not present in the results.
- If confidence is low, say so.
- If evidence conflicts, note the conflict.
- Keep provenance clear by labeling which source contributed what.
"""


# ---------------------------
# Planning and Validation
# ---------------------------

class QueryPlanner:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def plan(self, question: str) -> RoutingPlan:
        raw = self.llm.complete_json(PLANNER_SYSTEM_PROMPT, question)
        return RoutingPlan(
            user_question=question,
            intent=Intent(raw["intent"]),
            output_shape=OutputShape(raw["output_shape"]),
            entities=raw.get("entities", {}),
            evidence_types=[EvidenceType(x) for x in raw.get("evidence_types", [])],
            source_required=raw["source_required"],
            order=raw["order"],
            rationale=raw["rationale"],
            confidence=float(raw["confidence"]),
            steps=[
                PlanStep(
                    step_id=s["step_id"],
                    source=Source(s["source"]),
                    goal=StepGoal(s["goal"]),
                    instruction=s["instruction"],
                    stop_if_sufficient=bool(s.get("stop_if_sufficient", False)),
                )
                for s in raw.get("steps", [])
            ],
        )


class PlanValidator:
    @staticmethod
    def validate(plan: RoutingPlan) -> RoutingPlan:
        if not plan.steps:
            if plan.source_required == "structured":
                plan.steps = [
                    PlanStep(1, Source.STRUCTURED, StepGoal.FETCH_FACTS, "Answer using structured data only.", True)
                ]
                plan.order = "single"
            elif plan.source_required == "unstructured":
                plan.steps = [
                    PlanStep(1, Source.UNSTRUCTURED, StepGoal.FETCH_DOCUMENTS, "Answer using documents only.", True)
                ]
                plan.order = "single"
            else:
                first = Source.STRUCTURED if EvidenceType.AGGREGATION in plan.evidence_types else Source.UNSTRUCTURED
                second = Source.UNSTRUCTURED if first == Source.STRUCTURED else Source.STRUCTURED
                plan.order = "structured_first" if first == Source.STRUCTURED else "unstructured_first"
                plan.steps = [
                    PlanStep(1, first, StepGoal.FETCH_FACTS if first == Source.STRUCTURED else StepGoal.FETCH_DOCUMENTS, "Execute the first source.", False),
                    PlanStep(2, second, StepGoal.ENRICH, "Enrich with the second source.", True),
                ]

        if plan.order == "parallel":
            first = Source.STRUCTURED if EvidenceType.AGGREGATION in plan.evidence_types else plan.steps[0].source
            second = Source.UNSTRUCTURED if first == Source.STRUCTURED else Source.STRUCTURED
            plan.order = "structured_first" if first == Source.STRUCTURED else "unstructured_first"
            plan.steps = [
                PlanStep(1, first, StepGoal.FETCH_FACTS if first == Source.STRUCTURED else StepGoal.FETCH_DOCUMENTS, "Execute first source conservatively.", False),
                PlanStep(2, second, StepGoal.ENRICH, "Execute second source if needed.", True),
            ]

        return plan


# ---------------------------
# LangGraph Nodes
# ---------------------------

class MultiSourceLangGraphAssistant:
    def __init__(
        self,
        llm: LLMClient,
        structured_client: StructuredChatbotClient,
        unstructured_client: UnstructuredChatbotClient,
    ):
        self.llm = llm
        self.planner = QueryPlanner(llm)
        self.validator = PlanValidator()
        self.structured_client = structured_client
        self.unstructured_client = unstructured_client
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(GraphState)

        builder.add_node("plan", self.plan_node)
        builder.add_node("validate_plan", self.validate_plan_node)
        builder.add_node("execute_step", self.execute_step_node)
        builder.add_node("evaluate_progress", self.evaluate_progress_node)
        builder.add_node("synthesize_answer", self.synthesize_answer_node)

        builder.set_entry_point("plan")
        builder.add_edge("plan", "validate_plan")
        builder.add_edge("validate_plan", "execute_step")
        builder.add_edge("execute_step", "evaluate_progress")

        builder.add_conditional_edges(
            "evaluate_progress",
            self.route_after_evaluation,
            {
                "continue": "execute_step",
                "finish": "synthesize_answer",
            },
        )
        builder.add_edge("synthesize_answer", END)

        return builder.compile()

    def plan_node(self, state: GraphState) -> GraphState:
        question = state["question"]
        plan = self.planner.plan(question)
        debug_log = state.get("debug_log", []) + [f"Planned route: {plan.source_required}, order={plan.order}"]
        return {
            **state,
            "plan": plan,
            "current_step_index": 0,
            "results": [],
            "stop_early": False,
            "debug_log": debug_log,
        }

    def validate_plan_node(self, state: GraphState) -> GraphState:
        plan = self.validator.validate(state["plan"])
        debug_log = state.get("debug_log", []) + [f"Validated plan with {len(plan.steps)} step(s)"]
        return {**state, "plan": plan, "debug_log": debug_log}

    def execute_step_node(self, state: GraphState) -> GraphState:
        plan = state["plan"]
        step_index = state.get("current_step_index", 0)
        results = list(state.get("results", []))

        if step_index >= len(plan.steps):
            debug_log = state.get("debug_log", []) + ["No more steps to execute"]
            return {**state, "debug_log": debug_log}

        step = plan.steps[step_index]
        prior_context = self._build_cross_source_context(state)
        materialized_question = self._materialize_step_question(state["question"], step, prior_context)

        if step.source == Source.STRUCTURED:
            result = self.structured_client.ask(materialized_question, prior_context)
        else:
            result = self.unstructured_client.ask(materialized_question, prior_context)

        results.append(result)
        stop_early = False
        if step.stop_if_sufficient and self._is_sufficient(result, plan):
            stop_early = True

        debug_log = state.get("debug_log", []) + [
            f"Executed step {step.step_id} on {step.source.value} with confidence={result.confidence:.2f}",
            f"Stop early={stop_early}",
        ]

        return {
            **state,
            "results": results,
            "current_step_index": step_index + 1,
            "stop_early": stop_early,
            "debug_log": debug_log,
        }

    def evaluate_progress_node(self, state: GraphState) -> GraphState:
        plan = state["plan"]
        current_step_index = state.get("current_step_index", 0)
        stop_early = state.get("stop_early", False)

        finished = stop_early or current_step_index >= len(plan.steps)
        debug_log = state.get("debug_log", []) + [f"Evaluation finished={finished}"]
        return {**state, "stop_early": finished, "debug_log": debug_log}

    def route_after_evaluation(self, state: GraphState) -> str:
        if state.get("stop_early", False):
            return "finish"
        return "continue"

    def synthesize_answer_node(self, state: GraphState) -> GraphState:
        answer_input = {
            "question": state["question"],
            "plan": {
                "intent": state["plan"].intent.value,
                "source_required": state["plan"].source_required,
                "order": state["plan"].order,
                "rationale": state["plan"].rationale,
            },
            "results": [
                {
                    "source": r.source.value,
                    "success": r.success,
                    "confidence": r.confidence,
                    "payload": r.payload,
                    "citations": r.citations,
                    "error": r.error,
                }
                for r in state.get("results", [])
            ],
        }
        final_answer = self.llm.complete_text(ANSWER_SYSTEM_PROMPT, json.dumps(answer_input))
        debug_log = state.get("debug_log", []) + ["Synthesized final answer"]
        return {**state, "final_answer": final_answer, "debug_log": debug_log}

    def run(self, question: str) -> GraphState:
        return self.graph.invoke({"question": question})

    @staticmethod
    def _materialize_step_question(original_question: str, step: PlanStep, prior_context: Dict[str, Any]) -> str:
        return (
            f"Original user question: {original_question}\n"
            f"Current step goal: {step.goal.value}\n"
            f"Instruction: {step.instruction}\n"
            f"Prior context: {json.dumps(prior_context)}"
        )

    @staticmethod
    def _build_cross_source_context(state: GraphState) -> Dict[str, Any]:
        plan = state["plan"]
        results = state.get("results", [])
        context: Dict[str, Any] = {
            "entities": plan.entities,
            "prior_results": [],
        }
        for r in results:
            context["prior_results"].append(
                {
                    "source": r.source.value,
                    "confidence": r.confidence,
                    "payload_summary": r.payload,
                    "citations": r.citations,
                }
            )
        return context

    @staticmethod
    def _is_sufficient(result: SourceResult, plan: RoutingPlan) -> bool:
        if not result.success or result.confidence < 0.75:
            return False

        if plan.source_required == "both":
            return False

        if result.source == Source.STRUCTURED:
            return result.payload.get("records_found", 0) > 0
        if result.source == Source.UNSTRUCTURED:
            return result.payload.get("documents_found", 0) > 0
        return False


# ---------------------------
# Fake LLM for Local Testing
# Replace in production
# ---------------------------

class FakeLLM(LLMClient):
    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        q = user_prompt.lower()

        if ("wallet" in q or "revenue" in q or "top" in q or "interaction" in q) and (
            "latest" in q or "discussion" in q or "notes" in q or "why" in q or "topics" in q
        ):
            return {
                "intent": "briefing",
                "output_shape": "table",
                "entities": {
                    "clients": [],
                    "deals": [],
                    "products": [],
                    "time_range": None,
                    "other_filters": {},
                },
                "evidence_types": ["numeric", "narrative", "aggregation"],
                "source_required": "both",
                "order": "structured_first",
                "rationale": "Need structured filtering first, then document context.",
                "confidence": 0.93,
                "steps": [
                    {
                        "step_id": 1,
                        "source": "structured",
                        "goal": "filter_candidates",
                        "instruction": "Find top relevant clients or deals based on wallet, revenue, and interaction filters.",
                        "stop_if_sufficient": False,
                    },
                    {
                        "step_id": 2,
                        "source": "unstructured",
                        "goal": "enrich",
                        "instruction": "Pull latest notes and document context for the selected entities.",
                        "stop_if_sufficient": True,
                    },
                ],
            }

        if "antitrust" in q and ("fee" in q or "revenue" in q):
            return {
                "intent": "diagnosis",
                "output_shape": "table",
                "entities": {
                    "clients": [],
                    "deals": [],
                    "products": ["M&A"],
                    "time_range": None,
                    "other_filters": {},
                },
                "evidence_types": ["entity_discovery", "narrative", "numeric"],
                "source_required": "both",
                "order": "unstructured_first",
                "rationale": "Need to discover relevant deals in text first, then quantify commercial impact.",
                "confidence": 0.92,
                "steps": [
                    {
                        "step_id": 1,
                        "source": "unstructured",
                        "goal": "discover_themes",
                        "instruction": "Find live M&A deals mentioning antitrust concerns.",
                        "stop_if_sufficient": False,
                    },
                    {
                        "step_id": 2,
                        "source": "structured",
                        "goal": "enrich",
                        "instruction": "Fetch fee revenue or expected commercial value for the identified deals.",
                        "stop_if_sufficient": True,
                    },
                ],
            }

        if "wallet" in q or "revenue" in q or "expense" in q or "top" in q:
            return {
                "intent": "lookup",
                "output_shape": "table",
                "entities": {
                    "clients": [],
                    "deals": [],
                    "products": [],
                    "time_range": None,
                    "other_filters": {},
                },
                "evidence_types": ["numeric", "aggregation"],
                "source_required": "structured",
                "order": "single",
                "rationale": "Structured data is sufficient.",
                "confidence": 0.95,
                "steps": [
                    {
                        "step_id": 1,
                        "source": "structured",
                        "goal": "fetch_facts",
                        "instruction": "Answer using structured records only.",
                        "stop_if_sufficient": True,
                    }
                ],
            }

        return {
            "intent": "summary",
            "output_shape": "bullets",
            "entities": {
                "clients": [],
                "deals": [],
                "products": [],
                "time_range": None,
                "other_filters": {},
            },
            "evidence_types": ["narrative", "document_grounding"],
            "source_required": "unstructured",
            "order": "single",
            "rationale": "Question appears document-oriented.",
            "confidence": 0.89,
            "steps": [
                {
                    "step_id": 1,
                    "source": "unstructured",
                    "goal": "fetch_documents",
                    "instruction": "Retrieve and summarize relevant documents.",
                    "stop_if_sufficient": True,
                }
            ],
        }

    def complete_text(self, system_prompt: str, user_prompt: str) -> str:
        payload = json.loads(user_prompt)
        lines = [f"Question: {payload['question']}", ""]

        for result in payload.get("results", []):
            if result["source"] == "structured":
                lines.append("Structured evidence:")
                lines.append(f"- Confidence: {result['confidence']:.2f}")
                lines.append(f"- Summary: {result['payload'].get('answer', '')}")
            elif result["source"] == "unstructured":
                lines.append("Document evidence:")
                lines.append(f"- Confidence: {result['confidence']:.2f}")
                lines.append(f"- Summary: {result['payload'].get('answer', '')}")
            lines.append("")

        lines.append("Integrated answer:")
        lines.append("- This is a mock synthesized response. Replace FakeLLM with a real model for production use.")
        return "\n".join(lines)


# ---------------------------
# Example Usage
# ---------------------------

if __name__ == "__main__":
    llm = FakeLLM()
    assistant = MultiSourceLangGraphAssistant(
        llm=llm,
        structured_client=StructuredChatbotClient(),
        unstructured_client=UnstructuredChatbotClient(),
    )

    questions = [
        "What are our top healthcare clients by wallet with low recent interaction, and what are the latest topics discussed?",
        "Which live M&A deals mention antitrust concerns, and how much fee revenue is at stake?",
        "Summarize the key covenants in the latest debt term sheet for Acme.",
    ]

    for idx, question in enumerate(questions, start=1):
        print("=" * 100)
        print(f"Example {idx}")
        print(f"Question: {question}\n")

        state = assistant.run(question)
        print("Debug log:")
        for line in state.get("debug_log", []):
            print(f"- {line}")

        print("\nFinal answer:\n")
        print(state.get("final_answer", ""))
        print()
