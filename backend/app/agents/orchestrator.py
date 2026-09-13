# backend/app/agents/orchestrator.py
import os
import json
import logging
from typing import TypedDict, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL_NAME   = "llama3-70b-8192"
ALLOWED_LEVELS = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
PROHIBITED_FINDINGS = ("cease and desist", "has violated", "is guilty", "violation at")

class EnforcementState(TypedDict):
    cell_id:            int
    aqi_value:          float
    primary_violator:   str
    attribution_matrix: list        
    escalation_level:   Optional[str]   
    applicable_statutes: Optional[list] 
    enforcement_brief:  Optional[str]   
    statute_violated:   Optional[str]
    legal_notice_draft: Optional[str]
    dispatch_priority:  Optional[str]
    case_summary:       Optional[str]
    error:              Optional[str]

def _get_llm():
    # The deterministic path is the deployment default: it is fast, reproducible,
    # and keeps enforcement language inside reviewed templates. Operators may
    # explicitly enable Groq drafting without making it a runtime dependency.
    llm_enabled = os.getenv("AEROSHIELD_ENABLE_LLM", "false").strip().lower() in {"1", "true", "yes", "on"}
    if not llm_enabled:
        return None
    if not GROQ_API_KEY or GROQ_API_KEY == "MOCK_KEY_FOR_DEV": return None
    try:
        from langchain_groq import ChatGroq
        return ChatGroq(model_name=MODEL_NAME, api_key=GROQ_API_KEY, temperature=0.1, max_tokens=800)
    except Exception as e:
        logger.warning(f"Failed to init ChatGroq: {e}")
        return None


def _safe_text(value, fallback: str) -> str:
    """Reject empty or enforcement-assertive model text at the output boundary."""
    if not isinstance(value, str) or not value.strip():
        return fallback
    lowered = value.lower()
    if any(phrase in lowered for phrase in PROHIBITED_FINDINGS):
        return fallback
    return value.strip()


def _safe_level(value, fallback="HIGH") -> str:
    return value if value in ALLOWED_LEVELS else fallback

PLANNER_SYSTEM = """You are the AeroShield IQ Field Verification Planner. Model-based source attribution is screening evidence, not proof of a violation. Never declare guilt, fabricate measurements, or claim a statute was violated. Identify potentially relevant legal provisions for a human officer to review.
Respond ONLY with a valid JSON object. No markdown. Schema:
{"escalation_level": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW", "applicable_statutes": ["Section X..."], "enforcement_brief": "..."}"""

def planner_node(state: EnforcementState) -> EnforcementState:
    llm = _get_llm()
    if llm is None:
        aqi = state["aqi_value"]
        level = "CRITICAL" if aqi >= 400 else "HIGH" if aqi >= 200 else "MEDIUM" if aqi >= 100 else "LOW"
        return {**state, "escalation_level": level, "applicable_statutes": ["Air (Prevention and Control of Pollution) Act, 1981 — applicability requires officer review"], "enforcement_brief": f"Prioritise field verification near {state['primary_violator']}; do not treat the model output as a violation finding."}

    prompt = f"Cell ID: {state['cell_id']}\nAQI: {state['aqi_value']:.1f}\nViolator: {state['primary_violator']}\nRanking: {json.dumps(state['attribution_matrix'][:3])}"
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        response = llm.invoke([SystemMessage(content=PLANNER_SYSTEM), HumanMessage(content=prompt)])
        raw = response.content.strip().replace("```json", "").replace("```", "")
        parsed = json.loads(raw)
        statutes = parsed.get("applicable_statutes", [])
        if not isinstance(statutes, list) or not all(isinstance(item, str) for item in statutes):
            statutes = []
        return {
            **state,
            "escalation_level": _safe_level(parsed.get("escalation_level")),
            "applicable_statutes": statutes[:5],
            "enforcement_brief": _safe_text(parsed.get("enforcement_brief"), "Field verification required; no violation finding has been made."),
        }
    except Exception as e:
        return {**state, "escalation_level": "HIGH", "applicable_statutes": ["Air Act, 1981 — applicability requires officer review"], "enforcement_brief": "Field verification required; no violation finding has been made.", "error": str(e)}

LEGAL_SYSTEM = """You draft a DPCC field-verification brief for human review. The evidence is probabilistic screening output, not legal proof. Do not issue a cease-and-desist order or state that a facility violated the law. Draft only an inspection request and list legal provisions as potentially relevant.
Respond ONLY with a valid JSON object. No markdown. Schema:
{"statute_violated": "...", "legal_notice_draft": "...", "dispatch_priority": "CRITICAL" | "HIGH" | "MEDIUM", "case_summary": "..."}"""

def legal_drafter_node(state: EnforcementState) -> EnforcementState:
    llm = _get_llm()
    if llm is None:
        return {**state, "statute_violated": "Potentially relevant: Air Act, 1981 (human legal review required)", "legal_notice_draft": f"Inspection request: verify emissions and control-equipment operation in the vicinity of {state['primary_violator']}. This model output is not a finding of violation.", "dispatch_priority": state.get("escalation_level", "HIGH"), "case_summary": f"Field verification recommended near {state['primary_violator']}; preserve measurements and document findings."}

    brief_prompt = f"Brief:\n{state.get('enforcement_brief', '')}\nEscalation: {state.get('escalation_level', 'HIGH')}\nStatutes: {', '.join(state.get('applicable_statutes', []))}\nFacility: {state['primary_violator']}\nAQI: {state['aqi_value']:.1f}"
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        response = llm.invoke([SystemMessage(content=LEGAL_SYSTEM), HumanMessage(content=brief_prompt)])
        raw = response.content.strip().replace("```json", "").replace("```", "")
        parsed = json.loads(raw)
        return {
            **state,
            "statute_violated": _safe_text(parsed.get("statute_violated"), "Potentially relevant: Air Act, 1981 (human review required)"),
            "legal_notice_draft": _safe_text(parsed.get("legal_notice_draft"), "Inspection request only; no violation finding has been made."),
            "dispatch_priority": _safe_level(parsed.get("dispatch_priority")),
            "case_summary": _safe_text(parsed.get("case_summary"), "Field verification recommended; preserve measurements and document findings."),
        }
    except Exception as e:
        return {**state, "statute_violated": "Potentially relevant: Air Act, 1981 (human review required)", "legal_notice_draft": "Inspection request only; no violation finding has been made.", "dispatch_priority": "HIGH", "case_summary": "Field verification recommended; preserve measurements and document findings.", "error": str(e)}

def _build_graph():
    from langgraph.graph import StateGraph, END
    graph = StateGraph(EnforcementState)
    graph.add_node("planner", planner_node)
    graph.add_node("legal_drafter", legal_drafter_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "legal_drafter")
    graph.add_edge("legal_drafter", END)
    return graph.compile()

_enforcement_graph = None

def generate_enforcement_mandate(cell_id: int, aqi_value: float, primary_violator: str, attribution_matrix: list = None) -> dict:
    global _enforcement_graph
    initial_state = {"cell_id": cell_id, "aqi_value": aqi_value, "primary_violator": primary_violator, "attribution_matrix": attribution_matrix or [], "escalation_level": None, "applicable_statutes": None, "enforcement_brief": None, "statute_violated": None, "legal_notice_draft": None, "dispatch_priority": None, "case_summary": None, "error": None}
    try:
        if _enforcement_graph is None:
            _enforcement_graph = _build_graph()
        final_state = _enforcement_graph.invoke(initial_state)
    except Exception as e:
        # Keep the no-key/no-LangGraph path operational and conservative.
        final_state = legal_drafter_node(planner_node(initial_state))
        final_state["error"] = str(e)

    llm_enabled = os.getenv("AEROSHIELD_ENABLE_LLM", "false").strip().lower() in {"1", "true", "yes", "on"}
    return {
        "statute_violated": final_state.get("statute_violated") or "Potentially relevant: Air Act, 1981 (human review required)",
        "legal_notice_draft": final_state.get("legal_notice_draft") or "Inspection request only; no violation finding has been made.",
        "dispatch_priority": final_state.get("dispatch_priority", "HIGH"),
        "case_summary": final_state.get("case_summary", ""),
        "escalation_level": final_state.get("escalation_level", "HIGH"),
        "applicable_statutes": final_state.get("applicable_statutes", []),
        "enforcement_brief": final_state.get("enforcement_brief", ""),
        "_pipeline": "langgraph_2_agent",
        "_generation_mode": "groq_llm" if llm_enabled and GROQ_API_KEY else "deterministic_guardrail",
        "_error": final_state.get("error")
    }
