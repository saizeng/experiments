from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


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


@dataclass
class ExecutionContext:
    question: str
    plan: RoutingPlan
    results: List[SourceResult] = field(default_factory=list)
    final_answer: Optional[str] = None


class LLMClient:
    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        raise NotImplementedError("Implement with your preferred LLM provider")


class StructuredChatbotClient:
    def ask(self, question: str, context: Optional[Dict[str, Any]] = None) -> SourceResult:
        dummy_payload = {
            "answer": f"[Structured bot answer] {question}",
            "resolved_entities": context or {},
            "records_found": 5,
        }
        return SourceResult(
            source=Source.STRUCTURED,
            success=True,
            confidence=0.87,
            payload=dummy_payload,
            citations=[{"type": "record_ids", "ids": ["deal_123", "client_456"]}],
        )


class UnstructuredChatbotClient:
    def ask(self, question: str, context: Optional[Dict[str, Any]] = None) -> SourceResult:
        dummy_payload = {
            "answer": f"[Unstructured bot answer] {question}",
            "resolved_entities": context or {},
            "documents_found": 7,
        }
        return SourceResult(
            source=Source.UNSTRUCTURED,
            success=True,
            confidence=0.81,
            payload=dummy_payload,
            citations=[{"type": "document_ids", "ids": ["doc_a", "doc_b"]}],
        )


PLANNER_SYSTEM_PROMPT = """
You are a routing planner for an enterprise banking assistant.

You must decide whether a user question should route to:
1. structured data chatbot
2. unstructured document chatbot
3. both

Rules:
- You are only planning. Do not answer the question.
- Prefer a single source when sufficient.
- Use structured first for ranking, counts, filters, wallet, revenue, expense, interactions, pipeline, trends.
- Use unstructured first for summaries, reasons, risks, themes, notes, documents, precedent language, what was said.
- Use both if the question needs numeric facts plus document context, or entity discovery plus quantification.
- Sequential execution is preferred over parallel unless both are obviously needed and entity mapping is already explicit.
- If the first step is sufficient, allow stopping early.

Return strict JSON only with this shape:

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


class QueryPlanner:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def plan(self, question: str) -> RoutingPlan:
        raw = self.llm.complete_json(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=question,
        )

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
        if plan.source_required == "structured" and len(plan.steps) == 0:
            plan.steps = [
                PlanStep(
                    step_id=1,
                    source=Source.STRUCTURED,
                    goal=StepGoal.FETCH_FACTS,
                    instruction="Answer using structured data only.",
                    stop_if_sufficient=True,
                )
            ]
            plan.order = "single"

        if plan.source_required == "unstructured" and len(plan.steps) == 0:
            plan.steps = [
                PlanStep(
                    step_id=1,
                    source=Source.UNSTRUCTURED,
                    goal=StepGoal.FETCH_DOCUMENTS,
                    instruction="Answer using unstructured documents only.",
                    stop_if_sufficient=True,
                )
            ]
            plan.order = "single"

        if plan.order == "parallel":
            if plan.source_required == "both":
                first = Source.STRUCTURED if EvidenceType.AGGREGATION in plan.evidence_types else Source.UNSTRUCTURED
                second = Source.UNSTRUCTURED if first == Source.STRUCTURED else Source.STRUCTURED
                plan.order = "structured_first" if first == Source.STRUCTURED else "unstructured_first"
                plan.steps = [
                    PlanStep(
                        step_id=1,
                        source=first,
                        goal=StepGoal.FETCH_FACTS if first == Source.STRUCTURED else StepGoal.FETCH_DOCUMENTS,
                        instruction="Execute first source conservatively before second.",
                        stop_if_sufficient=False,
                    ),
                    PlanStep(
                        step_id=2,
                        source=second,
                        goal=StepGoal.ENRICH,
                        instruction="Enrich with second source if needed.",
                        stop_if_sufficient=True,
                    ),
                ]

        return plan


class Router:
    def __init__(
        self,
        structured_client: StructuredChatbotClient,
        unstructured_client: UnstructuredChatbotClient,
    ):
        self.structured_client = structured_client
        self.unstructured_client = unstructured_client

    def execute(self, plan: RoutingPlan) -> ExecutionContext:
        ctx = ExecutionContext(question=plan.user_question, plan=plan)

        for step in sorted(plan.steps, key=lambda s: s.step_id):
            result = self._execute_step(step, ctx)
            ctx.results.append(result)

            if result.success and step.stop_if_sufficient and self._is_sufficient(result, plan):
                break

        return ctx

    def _execute_step(self, step: PlanStep, ctx: ExecutionContext) -> SourceResult:
        prior_context = self._build_cross_source_context(ctx)

        if step.source == Source.STRUCTURED:
            return self.structured_client.ask(
                question=self._materialize_step_question(ctx.question, step, prior_context),
                context=prior_context,
            )

        if step.source == Source.UNSTRUCTURED:
            return self.unstructured_client.ask(
                question=self._materialize_step_question(ctx.question, step, prior_context),
                context=prior_context,
            )

        raise ValueError(f"Unsupported source: {step.source}")

    def _materialize_step_question(
        self,
        original_question: str,
        step: PlanStep,
        prior_context: Dict[str, Any],
    ) -> str:
        return (
            f"Original user question: {original_question}\n"
            f"Current step goal: {step.goal.value}\n"
            f"Instruction: {step.instruction}\n"
            f"Prior context: {json.dumps(prior_context)}"
        )

    def _build_cross_source_context(self, ctx: ExecutionContext) -> Dict[str, Any]:
        context: Dict[str, Any] = {
            "entities": ctx.plan.entities,
            "prior_results": [],
        }

        for r in ctx.results:
            context["prior_results"].append({
                "source": r.source.value,
                "confidence": r.confidence,
                "payload_summary": r.payload,
                "citations": r.citations,
            })

        return context

    def _is_sufficient(self, result: SourceResult, plan: RoutingPlan) -> bool:
        if plan.source_required == result.source.value:
            return result.confidence >= 0.75

        if plan.source_required == "both":
            return False

        return False


ANSWER_SYNTHESIS_SYSTEM_PROMPT = """
You are an enterprise banking assistant.

Use the source results to answer the user's question.
Rules:
- Separate structured facts from document-based narrative.
- Do not invent facts not present in the results.
- If confidence is low, say so.
- If evidence conflicts, note the conflict.
- Keep provenance clear by labeling which source contributed what.
Return plain text as JSON with shape: {"answer": "..."}
"""


class AnswerSynthesizer:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def synthesize(self, ctx: ExecutionContext) -> str:
        payload = {
            "question": ctx.question,
            "plan": {
                "intent": ctx.plan.intent.value,
                "source_required": ctx.plan.source_required,
                "order": ctx.plan.order,
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
                for r in ctx.results
            ],
        }

        raw = self.llm.complete_json(
            system_prompt=ANSWER_SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=json.dumps(payload),
        )
        return raw.get("answer", "No answer generated.")


class MultiSourceAssistant:
    def __init__(
        self,
        planner: QueryPlanner,
        router: Router,
        synthesizer: AnswerSynthesizer,
        validator: Optional[PlanValidator] = None,
    ):
        self.planner = planner
        self.router = router
        self.synthesizer = synthesizer
        self.validator = validator or PlanValidator()

    def answer(self, question: str) -> ExecutionContext:
        plan = self.planner.plan(question)
        plan = self.validator.validate(plan)
        ctx = self.router.execute(plan)
        ctx.final_answer = self.synthesizer.synthesize(ctx)
        return ctx


class FakeLLM(LLMClient):
    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        q = user_prompt.lower()

        if "wallet" in q or "revenue" in q or "top" in q:
            if "latest discussion" in q or "notes" in q or "why" in q:
                return {
                    "intent": "briefing",
                    "output_shape": "table",
                    "entities": {
                        "clients": [],
                        "deals": [],
                        "products": [],
                        "time_range": None,
                        "other_filters": {}
                    },
                    "evidence_types": ["numeric", "narrative", "aggregation"],
                    "source_required": "both",
                    "order": "structured_first",
                    "rationale": "Need structured ranking first, then unstructured context.",
                    "confidence": 0.91,
                    "steps": [
                        {
                            "step_id": 1,
                            "source": "structured",
                            "goal": "filter_candidates",
                            "instruction": "Find top relevant clients or deals based on revenue, wallet, or filters.",
                            "stop_if_sufficient": False
                        },
                        {
                            "step_id": 2,
                            "source": "unstructured",
                            "goal": "enrich",
                            "instruction": "Pull latest notes and document context for the selected entities.",
                            "stop_if_sufficient": True
                        }
                    ]
                }
            else:
                return {
                    "intent": "lookup",
                    "output_shape": "table",
                    "entities": {
                        "clients": [],
                        "deals": [],
                        "products": [],
                        "time_range": None,
                        "other_filters": {}
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
                            "stop_if_sufficient": True
                        }
                    ]
                }

        if system_prompt == ANSWER_SYNTHESIS_SYSTEM_PROMPT:
            data = json.loads(user_prompt)
            return {
                "answer": f"Synthesized answer using {len(data.get('results', []))} source result(s)."
            }

        return {
            "intent": "summary",
            "output_shape": "bullets",
            "entities": {
                "clients": [],
                "deals": [],
                "products": [],
                "time_range": None,
                "other_filters": {}
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
                    "stop_if_sufficient": True
                }
            ]
        }


def demo() -> None:
    llm = FakeLLM()
    planner = QueryPlanner(llm=llm)
    router = Router(
        structured_client=StructuredChatbotClient(),
        unstructured_client=UnstructuredChatbotClient(),
    )
    synthesizer = AnswerSynthesizer(llm=llm)
    assistant = MultiSourceAssistant(
        planner=planner,
        router=router,
        synthesizer=synthesizer,
    )

    question = (
        "What are our top healthcare clients by wallet with low recent interaction, "
        "and what are the latest topics discussed?"
    )
    ctx = assistant.answer(question)

    print("PLAN")
    print(ctx.plan)
    print("\nRESULTS")
    for r in ctx.results:
        print(r.source, r.success, r.confidence, r.payload)
    print("\nFINAL ANSWER")
    print(ctx.final_answer)


if __name__ == "__main__":
    demo()
