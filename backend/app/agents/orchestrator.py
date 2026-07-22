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
    if not GROQ_API_KEY or GROQ_API_KEY == "MOCK_KEY_FOR_DEV": return None
    try:
        from langchain_groq import ChatGroq
        return ChatGroq(model_name=MODEL_NAME, api_key=GROQ_API_KEY, temperature=0.1, max_tokens=800)
    except Exception as e:
        logger.warning(f"Failed to init ChatGroq: {e}")
        return None

PLANNER_SYSTEM = """You are the AeroShield IQ Enforcement Planner Agent operating under the Environment Protection Act 1986 and Air Act 1981.
Respond ONLY with a valid JSON object. No markdown. Schema:
{"escalation_level": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW", "applicable_statutes": ["Section X..."], "enforcement_brief": "..."}"""

def planner_node(state: EnforcementState) -> EnforcementState:
    llm = _get_llm()
    if llm is None:
        aqi = state["aqi_value"]
        level = "CRITICAL" if aqi >= 400 else "HIGH" if aqi >= 200 else "MEDIUM" if aqi >= 100 else "LOW"
        return {**state, "escalation_level": level, "applicable_statutes": ["Section 21 of Air Act, 1981"], "enforcement_brief": f"Violation at {state['primary_violator']}."}

    from langchain_core.messages import HumanMessage, SystemMessage
    prompt = f"Cell ID: {state['cell_id']}\nAQI: {state['aqi_value']:.1f}\nViolator: {state['primary_violator']}\nRanking: {json.dumps(state['attribution_matrix'][:3])}"
    try:
        response = llm.invoke([SystemMessage(content=PLANNER_SYSTEM), HumanMessage(content=prompt)])
        raw = response.content.strip().replace("```json", "").replace("```", "")
        parsed = json.loads(raw)
        return {**state, "escalation_level": parsed.get("escalation_level", "HIGH"), "applicable_statutes": parsed.get("applicable_statutes", []), "enforcement_brief": parsed.get("enforcement_brief", "")}
    except Exception as e:
        return {**state, "escalation_level": "HIGH", "applicable_statutes": ["Section 21, Air Act 1981"], "enforcement_brief": "Fallback brief.", "error": str(e)}

LEGAL_SYSTEM = """You are the DPCC Legal Compliance Agent.
Respond ONLY with a valid JSON object. No markdown. Schema:
{"statute_violated": "...", "legal_notice_draft": "...", "dispatch_priority": "CRITICAL" | "HIGH" | "MEDIUM", "case_summary": "..."}"""

def legal_drafter_node(state: EnforcementState) -> EnforcementState:
    llm = _get_llm()
    if llm is None:
        return {**state, "statute_violated": "Section 21, Air Act 1981", "legal_notice_draft": f"Cease and desist issued to {state['primary_violator']}.", "dispatch_priority": state.get("escalation_level", "HIGH"), "case_summary": f"Field squad alert: violation at {state['primary_violator']}."}

    from langchain_core.messages import HumanMessage, SystemMessage
    brief_prompt = f"Brief:\n{state.get('enforcement_brief', '')}\nEscalation: {state.get('escalation_level', 'HIGH')}\nStatutes: {', '.join(state.get('applicable_statutes', []))}\nFacility: {state['primary_violator']}\nAQI: {state['aqi_value']:.1f}"
    try:
        response = llm.invoke([SystemMessage(content=LEGAL_SYSTEM), HumanMessage(content=brief_prompt)])
        raw = response.content.strip().replace("```json", "").replace("```", "")
        parsed = json.loads(raw)
        return {**state, "statute_violated": parsed.get("statute_violated", "Air Act 1981"), "legal_notice_draft": parsed.get("legal_notice_draft", ""), "dispatch_priority": parsed.get("dispatch_priority", "HIGH"), "case_summary": parsed.get("case_summary", "")}
    except Exception as e:
        return {**state, "statute_violated": "Air Act 1981", "legal_notice_draft": "Emergency advisory.", "dispatch_priority": "HIGH", "case_summary": "Alert dispatched.", "error": str(e)}

def _build_graph():
    from langgraph.graph import StateGraph, END
    graph = StateGraph(EnforcementState)
    graph.add_node("planner", planner_node)
    graph.add_node("legal_drafter", legal_drafter_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "legal_drafter")
    graph.add_edge("legal_drafter", END)
    return graph.compile()

_enforcement_graph = _build_graph()

def generate_enforcement_mandate(cell_id: int, aqi_value: float, primary_violator: str, attribution_matrix: list = None) -> dict:
    initial_state = {"cell_id": cell_id, "aqi_value": aqi_value, "primary_violator": primary_violator, "attribution_matrix": attribution_matrix or [], "escalation_level": None, "applicable_statutes": None, "enforcement_brief": None, "statute_violated": None, "legal_notice_draft": None, "dispatch_priority": None, "case_summary": None, "error": None}
    try: final_state = _enforcement_graph.invoke(initial_state)
    except Exception as e: final_state = {**initial_state, "error": str(e)}

    return {
        "statute_violated": final_state.get("statute_violated", "Section 21, Air Act 1981"),
        "legal_notice_draft": final_state.get("legal_notice_draft", "Advisory issued."),
        "dispatch_priority": final_state.get("dispatch_priority", "HIGH"),
        "case_summary": final_state.get("case_summary", ""),
        "escalation_level": final_state.get("escalation_level", "HIGH"),
        "applicable_statutes": final_state.get("applicable_statutes", []),
        "enforcement_brief": final_state.get("enforcement_brief", ""),
        "_pipeline": "langgraph_2_agent",
        "_error": final_state.get("error")
    }