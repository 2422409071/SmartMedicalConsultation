"""
Answer Fusion Agent
Combines all agent outputs into a coherent final answer
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from langchain_core.prompts import ChatPromptTemplate

from src.common.llm import get_llm
from src.common.logger import setup_logger
from src.agents.state import AgentState

logger = setup_logger(__name__, "agents.log")


SYSTEM_PROMPT = """你是一个专业的医疗客服。请将以下信息整合成一条连贯、专业的回答。

要求：
1. 语言专业但易懂
2. 结构清晰，可以使用标题和列表
3. 确保免责声明在回答末尾
4. 如果有紧急警告，放在回答开头

用户问题：{query}

意图类型：{intent}

检测到的症状：{symptoms}

推荐科室：{departments}

药物建议：{medications}

医学知识：{knowledge}

就医建议：{advice}

免责声明：{disclaimers}

紧急警告：{warnings}

请整合以上信息，生成最终回答。"""


class AnswerFusionAgent:
    """Answer Fusion Agent"""

    def __init__(self):
        self.llm = get_llm(temperature=0.3)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", "请整合信息生成回答")
        ])

    def __call__(self, state: AgentState) -> dict:
        """Process state and return update"""
        logger.info("[AnswerFusion] Fusing answers...")

        query = state.get("query", "")
        intent = state.get("intent", "")
        symptoms = state.get("symptoms", [])
        departments = state.get("departments", [])
        medications = state.get("medications", [])
        knowledge = state.get("knowledge_answer", "")
        advice = state.get("advice", [])
        disclaimers = state.get("disclaimers", [])
        warnings = state.get("warnings", [])
        needs_clarify = state.get("needs_clarification", False)
        clarification = state.get("clarification", "")

        # Build simple answer without LLM for reliability
        parts = []

        # Emergency warnings first
        if warnings:
            parts.extend(warnings)
            parts.append("")

        # 信息不足：以追问为主，不输出空的科室/药物行
        if needs_clarify and clarification:
            parts.append("❓ " + clarification)
            if symptoms:
                parts.append("")
                parts.append(f"📝 我已记录的信息：{'、'.join(symptoms)}")
        else:
            # Main content based on intent
            if intent == "appointment":
                if symptoms:
                    parts.append(f"📋 检测到的症状：{'、'.join(symptoms)}")
                    parts.append("")
                if departments:
                    parts.append(f"🏥 推荐科室：{'、'.join(departments)}")
                    parts.append("")
                else:
                    # 有症状但三级回退仍未命中科室时，给出兜底引导
                    parts.append("🏥 科室建议：建议先到**全科/导诊台**初筛，或补充症状细节后我再为您精确推荐。")
                    parts.append("")
                if advice:
                    parts.append("💡 就医建议：")
                    for a in advice:
                        parts.append(f"  - {a}")
                    parts.append("")

            elif intent == "medication":
                if medications:
                    parts.append("💊 药物建议：")
                    for med in medications:
                        parts.append(f"  - {med.get('name', '')}")
                    parts.append("")

            elif intent == "knowledge":
                if knowledge:
                    parts.append(knowledge)
                    parts.append("")

        # Disclaimers at the end (MANDATORY)
        if disclaimers:
            parts.append("")
            parts.extend(disclaimers)

        final_answer = "\n".join(parts)

        logger.info(f"[AnswerFusion] Final answer: {len(final_answer)} chars")

        return {
            "final_answer": final_answer,
            "messages": [f"[AnswerFusion] Generated final answer: {len(final_answer)} chars"]
        }


# Singleton instance
answer_fusion = AnswerFusionAgent()
