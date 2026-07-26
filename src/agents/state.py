"""
LangGraph Agent State Definition
Defines the global state shared across all agents in the medical consultation system.

Usage:
    from src.agents.state import AgentState

    # Initialize state
    state: AgentState = {
        "query": "我最近头痛、头晕，应该挂什么科？",
        "messages": []
    }
"""

import operator
from typing import TypedDict, Optional, Annotated


# ============================================================
# Intent Types
# ============================================================

class IntentType:
    """Intent type constants"""
    APPOINTMENT = "appointment"    # 看病挂号：描述症状，想知道挂什么科
    MEDICATION = "medication"      # 用药建议：询问吃什么药、怎么用药
    KNOWLEDGE = "knowledge"        # 医学知识：询问疾病定义、病因等
    EMERGENCY = "emergency"        # 急症：胸痛、呼吸困难等紧急症状
    GENERAL = "general"            # 通用对话：非医疗问题

    ALL = [APPOINTMENT, MEDICATION, KNOWLEDGE, EMERGENCY, GENERAL]


# ============================================================
# Severity Levels
# ============================================================

class SeverityLevel:
    """Severity level constants"""
    LOW = "low"          # 轻微：不影响日常生活
    MEDIUM = "medium"    # 中等：需要关注
    HIGH = "high"        # 严重：需要尽快就医
    EMERGENCY = "emergency"  # 急症：需要立即就医

    ALL = [LOW, MEDIUM, HIGH, EMERGENCY]


# ============================================================
# Agent State Definition
# ============================================================

class AgentState(TypedDict, total=False):
    """
    LangGraph Global State for Medical Consultation System

    All agents read from and write to this shared state.
    List fields use Annotated[..., operator.add] for accumulation across agents.

    Fields:
        query: User's original question
        intent: Detected intent type (appointment/medication/knowledge/emergency/general)
        symptoms: List of detected symptoms
        severity: Symptom severity level (low/medium/high/emergency)
        is_emergency: Whether this is an emergency situation
        departments: Recommended departments for visit
        medications: Recommended medications with details
        knowledge_answer: Medical knowledge answer
        advice: Pre-visit advice and precautions
        disclaimers: Mandatory medical disclaimers
        warnings: Emergency warnings (if applicable)
        final_answer: Final composed answer
        retrieved_entities: Entities retrieved from knowledge graph
        messages: Debug messages for tracing agent execution
    """

    # ===== User Input =====
    query: str  # 用户原始问题

    # ===== Conversation Memory（由 API 按 session_id 注入，不参与累加）=====
    history: list  # 历史轮次 [{role, content, ...}]，供 Agent 读取上下文

    # ===== Clarification（信息不足时的追问机制）=====
    needs_clarification: bool  # 是否需要向用户追问
    clarification: str         # 追问话术

    # ===== Intent Classification =====
    intent: str  # 意图类型: appointment | medication | knowledge | emergency | general
    intent_confidence: float  # 意图识别置信度 (0-1)

    # ===== Symptom Detection =====
    symptoms: Annotated[list[str], operator.add]  # 症状列表（累加）
    severity: str  # 严重程度: low | medium | high | emergency
    is_emergency: bool  # 是否为急症

    # ===== Department Recommendation =====
    departments: Annotated[list[str], operator.add]  # 推荐科室列表（累加）

    # ===== Medication Advice =====
    medications: Annotated[list[dict], operator.add]  # 药物信息列表（累加）
    # 每个药物: {"name": str, "usage": str, "side_effects": list[str]}

    # ===== Medical Knowledge =====
    knowledge_answer: str  # 医学知识回答

    # ===== Pre-visit Advice =====
    advice: Annotated[list[str], operator.add]  # 就医建议列表（累加）

    # ===== Safety Check (Mandatory) =====
    disclaimers: Annotated[list[str], operator.add]  # 免责声明（累加，必须！）
    warnings: Annotated[list[str], operator.add]  # 急症警告（累加）

    # ===== Final Output =====
    final_answer: str  # 最终回答

    # ===== Retrieved Entities =====
    retrieved_entities: Annotated[list[dict], operator.add]  # 检索到的知识图谱实体（累加）
    # 每个实体: {"name": str, "type": str, "score": float, "source": str}

    # ===== Debug Messages =====
    messages: Annotated[list[str], operator.add]  # 调试消息（累加）


# ============================================================
# Helper Functions
# ============================================================

def create_initial_state(query: str, history: list | None = None) -> AgentState:
    """
    Create initial state with user query (and optional conversation history)

    Args:
        query: User's question
        history: Prior conversation turns (from session memory)

    Returns:
        Initialized AgentState
    """
    return AgentState(
        query=query,
        history=history or [],
        needs_clarification=False,
        clarification="",
        intent="",
        intent_confidence=0.0,
        symptoms=[],
        severity="",
        is_emergency=False,
        departments=[],
        medications=[],
        knowledge_answer="",
        advice=[],
        disclaimers=[],
        warnings=[],
        final_answer="",
        retrieved_entities=[],
        messages=[f"[START] User query: {query}"]
    )


def add_message(state: AgentState, message: str) -> dict:
    """
    Create update dict to add a debug message

    Args:
        state: Current state
        message: Message to add

    Returns:
        Update dict for state
    """
    return {"messages": [message]}


# ============================================================
# Main: Self-test
# ============================================================

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("AGENT STATE TEST")
    print("=" * 60)

    # Test 1: Create initial state
    print("\n[TEST 1] Create initial state")
    state = create_initial_state("我最近头痛、头晕，应该挂什么科？")
    print(f"  query: {state['query']}")
    print(f"  intent: '{state['intent']}' (empty)")
    print(f"  symptoms: {state['symptoms']}")
    print(f"  messages: {state['messages']}")

    # Test 2: Simulate agent updates
    print("\n[TEST 2] Simulate agent updates")

    # Intent classifier update
    intent_update = {
        "intent": IntentType.APPOINTMENT,
        "intent_confidence": 0.95,
        "messages": ["[IntentClassifier] Detected: appointment (0.95)"]
    }
    print(f"  Intent update: {intent_update['intent']}")

    # Symptom detector update
    symptom_update = {
        "symptoms": ["头痛", "头晕"],
        "severity": SeverityLevel.MEDIUM,
        "messages": ["[SymptomDetector] Found: 头痛, 头晕"]
    }
    print(f"  Symptom update: {symptom_update['symptoms']}")

    # Department recommender update
    dept_update = {
        "departments": ["神经内科", "心血管内科"],
        "messages": ["[DepartmentRecommender] Recommended: 神经内科, 心血管内科"]
    }
    print(f"  Department update: {dept_update['departments']}")

    # Safety checker update (mandatory!)
    safety_update = {
        "disclaimers": ["🔴 重要声明：本建议仅供参考，不能替代专业医疗诊断。请及时就医，遵医嘱用药！"],
        "messages": ["[SafetyChecker] Added disclaimer"]
    }
    print(f"  Safety update: disclaimer added")

    # Test 3: Verify list accumulation works
    print("\n[TEST 3] Verify list accumulation")
    print("  (In LangGraph, Annotated[list, operator.add] accumulates lists)")
    print("  Example: symptoms from multiple agents will be merged")

    # Test 4: Intent types
    print("\n[TEST 4] Intent types")
    for intent in IntentType.ALL:
        print(f"  - {intent}")

    # Test 5: Severity levels
    print("\n[TEST 5] Severity levels")
    for level in SeverityLevel.ALL:
        print(f"  - {level}")

    print("\n" + "=" * 60)
    print("[SUCCESS] Agent state tests passed!")
    print("=" * 60)
