"""
Department Recommender Agent
三级回退推荐就诊科室：
  Tier1 图谱精确匹配（症状→疾病→科室）
  Tier2 向量模糊匹配（症状语义检索最近疾病/症状→科室）
  Tier3 LLM 知识兜底（图谱/向量都没有时，靠大模型医学常识推荐）
解决"知识图谱症状词表覆盖不足导致推荐为空"的问题（如"左眼疼痛/上睑肿胀"）。
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from pydantic import BaseModel, Field

from src.common.llm import get_llm
from src.common.logger import setup_logger
from src.common.memory import format_history
from src.agents.state import AgentState

logger = setup_logger(__name__, "agents.log")


class DeptFallback(BaseModel):
    """LLM 兜底导诊结果"""
    departments: list[str] = Field(description="推荐的最合适就诊科室 1-3 个；若信息确实不足则返回空列表")
    reason: str = Field(default="", description="推荐理由（简短）")
    follow_up: str = Field(default="", description="若信息不足，需要向用户追问的具体内容")


DEPT_FALLBACK_PROMPT = """你是资深医院导诊专家。请根据【症状】【本轮输入】【对话历史】推荐最合适的 1-3 个就诊科室。

规则：
- 依据症状的解剖部位/系统判断科室（如眼部症状→眼科；胸痛→心内科/急诊；腹痛→消化内科/普外科）。
- 给出 1-3 个科室，按优先级排序，并简述理由。
- 只有当症状信息确实太少、无法判断时，departments 才返回空列表，并在 follow_up 写明需要追问的内容（部位、持续时间、伴随症状、诱因等）。
- 只输出结构化结果，不要寒暄。"""


class DepartmentRecommenderAgent:
    """Department Recommender Agent with 3-tier fallback"""

    def __init__(self):
        self.llm = get_llm(temperature=0.1)
        self.fallback_chain = self.llm.with_structured_output(DeptFallback)

    # ---------- Tier 1: 图谱精确 ----------
    def _tier_graph(self, symptoms: list[str]) -> list[str]:
        depts: list[str] = []
        try:
            from src.knowledge_graph.graph_queries import GraphQueries
            q = GraphQueries()
            for s in symptoms:
                for disease in q.get_symptom_diseases(s)[:3]:
                    depts.extend(q.get_disease_departments(disease))
            q.close()
        except Exception as e:
            logger.warning(f"[DepartmentRecommender] graph tier error: {e}")
        return list(dict.fromkeys(depts))

    # ---------- Tier 2: LLM 兜底 ----------
    def _tier_llm(self, symptoms: list[str], query: str, history: list) -> DeptFallback:
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", DEPT_FALLBACK_PROMPT),
            ("user", "【症状】{symptoms}\n【本轮输入】{query}\n【对话历史】{history}")
        ])
        msgs = prompt.format_messages(
            symptoms="、".join(symptoms) if symptoms else "（未明确）",
            query=query,
            history=format_history(history),
        )
        return self.fallback_chain.invoke(msgs)

    def __call__(self, state: AgentState) -> dict:
        symptoms = state.get("symptoms", [])
        query = state.get("query", "")
        history = state.get("history", [])
        logger.info(f"[DepartmentRecommender] Symptoms: {symptoms}")

        departments: list[str] = []
        source = "none"
        follow_up = ""

        # Tier 1：图谱精确匹配（高置信）
        if symptoms:
            departments = self._tier_graph(symptoms)
            if departments:
                source = "graph"

        # Tier 2：LLM 知识兜底（图谱未覆盖时，靠大模型医学常识；
        # 注：不再使用向量层做科室映射——在小知识图谱上向量近邻噪声大，
        # 会把"左眼疼痛"错误映射到无关科室，反而遮蔽正确的 LLM 判断）
        if not departments:
            try:
                fb = self._tier_llm(symptoms, query, history)
                departments = fb.departments or []
                follow_up = fb.follow_up or ""
                source = "llm" if departments else "llm(empty)"
                logger.info(f"[DepartmentRecommender] LLM fallback -> {departments} | reason={fb.reason}")
            except Exception as e:
                logger.error(f"[DepartmentRecommender] LLM fallback error: {e}")

        departments = departments[:5]
        logger.info(f"[DepartmentRecommender] Recommended: {departments} (source={source})")

        update = {
            "departments": departments,
            "messages": [f"[DepartmentRecommender] Recommended: {departments} (source={source})"]
        }

        # 兜底仍为空且有追问建议 → 触发追问
        if not departments and follow_up:
            update["needs_clarification"] = True
            update["clarification"] = follow_up

        return update


# Singleton instance
department_recommender = DepartmentRecommenderAgent()


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("DEPARTMENT RECOMMENDER AGENT TEST (3-tier)")
    print("=" * 60)

    for symptoms in [["头痛", "头晕"], ["左眼疼痛", "上眼睑肿胀", "上眼睑硬结", "触痛"]]:
        print(f"\nSymptoms: {symptoms}")
        result = department_recommender({"symptoms": symptoms, "query": "挂什么科", "history": []})
        print(f"  -> {result['departments']}")

    print("\n" + "=" * 60)
