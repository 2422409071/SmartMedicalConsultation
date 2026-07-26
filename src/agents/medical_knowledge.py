"""
Medical Knowledge Agent —— 就地升级为 ReAct 自主工具调用 Agent
（改进指令 §5.5 的"推荐就地改造"路径：本模块名与导出单例名 `medical_knowledge` 不变，
 故 graph.py 的 import 与 `add_node("medical_knowledge", ...)` 及所有边**零改动**，无死文件。）

只写 state["knowledge_answer"]（+ 调试 messages，含 tool 名单便于观测），
其后的 safety_checker -> answer_fusion 仍照常运行，免责声明不丢。
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

from src.common.llm import get_llm
from src.common.logger import setup_logger
from src.common.memory import format_history
from src.agents.state import AgentState
from src.agents.tools import search_knowledge_graph, semantic_search, get_medication_info

logger = setup_logger(__name__, "agents.log")

# F5：SYSTEM 必须命令模型"先取证、查不到别编造、通俗中文、用药提醒遵医嘱"
SYSTEM_PROMPT = """你是智能医疗问诊系统中的"医学知识 / 复杂问答"专家，配备以下工具，需要在时主动调用：
- search_knowledge_graph(entity)：查知识图谱（疾病↔症状/科室/药物/检查）。
- semantic_search(query)：语义向量检索，用于表述与实体名不一致时。
- get_medication_info(name)：查药物类别/副作用/禁忌/相互作用。

工作准则：
1. 涉及具体疾病/症状/药物/相互作用的**事实性**问题时，**先调用工具取证**，再据工具结果作答，绝不凭空编造。
2. 若工具查无结果，**如实说明**并给一般性建议，不要捏造。
3. 用通俗中文作答，结构清晰，可用列表。
4. 涉及治疗/用药时，提醒"请遵医嘱、咨询医生"。"""

_FALLBACK = "抱歉，我暂时无法回答这个问题，请咨询专业医生。"


class MedicalKnowledgeAgent:
    """Tool-calling ReAct agent for the knowledge / complex-QA branch."""

    def __init__(self):
        self.llm = get_llm(temperature=0.1)
        # 无需新增 pip 依赖：create_react_agent 已在 langgraph 内
        self.agent = create_react_agent(
            self.llm,
            tools=[search_knowledge_graph, semantic_search, get_medication_info],
        )

    @staticmethod
    def _tool_calls_of(msg) -> list[str]:
        # F3：工具轨迹 = AIMessage.tool_calls（list of {name, args, id}）
        tc = getattr(msg, "tool_calls", None)
        if not tc and isinstance(msg, dict):
            tc = msg.get("tool_calls")
        if not tc:
            return []
        return [(c.get("name") if isinstance(c, dict) else getattr(c, "name", "")) for c in tc]

    @staticmethod
    def _final_text(msgs) -> str:
        # F3：最终答 = res["messages"] 里最后一个 AIMessage 的 content
        for m in reversed(msgs):
            if isinstance(m, AIMessage) and m.content:
                return m.content
            if isinstance(m, dict) and m.get("type") == "ai" and m.get("content"):
                return m["content"]
        return ""

    def __call__(self, state: AgentState) -> dict:
        query = state.get("query", "")
        history = state.get("history", [])
        logger.info(f"[MedicalKnowledge] Query: {query[:50]}... (history {len(history)})")

        user_text = (f"【对话历史】\n{format_history(history)}\n\n【本轮问题】\n{query}") if history else query

        try:
            res = self.agent.invoke({
                "messages": [SystemMessage(SYSTEM_PROMPT), HumanMessage(user_text)]
            })
            msgs = res["messages"]
            tool_calls = [name for m in msgs for name in self._tool_calls_of(m)]
            final = self._final_text(msgs) or _FALLBACK

            logger.info(f"[MedicalKnowledge] tools={tool_calls} answer={len(final)} chars")
            return {
                "knowledge_answer": final,           # 只写 knowledge_answer，不写 final_answer
                "retrieved_entities": [],
                "messages": [f"[MedicalKnowledge] tools={tool_calls} answer={len(final)} chars"],
            }
        except Exception as e:
            logger.error(f"[MedicalKnowledge] Error: {e}")
            return {
                "knowledge_answer": _FALLBACK,
                "retrieved_entities": [],
                "messages": [f"[MedicalKnowledge] Error: {e}"],
            }


# 单例名保持 medical_knowledge → graph.py 零改动
medical_knowledge = MedicalKnowledgeAgent()
