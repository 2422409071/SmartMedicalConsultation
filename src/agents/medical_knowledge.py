"""
Medical Knowledge Agent
Answers medical knowledge questions using vector search and LLM
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


SYSTEM_PROMPT = """你是一个专业的医学科普专家。请用通俗易懂的语言回答用户的医学知识问题。

要求：
1. 回答要科学准确，基于现代医学知识
2. 语言通俗易懂，避免过多专业术语
3. 如果涉及具体治疗，提醒用户咨询医生
4. 回答结构清晰，可以使用列表

参考知识：
{context}

请回答用户的问题。"""


class MedicalKnowledgeAgent:
    """Medical Knowledge Agent"""

    def __init__(self):
        self.llm = get_llm(temperature=0.3)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", "{query}")
        ])

    def __call__(self, state: AgentState) -> dict:
        """Process state and return update"""
        query = state.get("query", "")
        logger.info(f"[MedicalKnowledge] Query: {query[:50]}...")

        # Get context from vector search
        context = ""
        retrieved_entities = []

        try:
            from src.vector_store.search import VectorRetriever
            retriever = VectorRetriever()
            results = retriever.search(query, top_k=5)

            if results:
                context_parts = []
                for r in results:
                    context_parts.append(f"- {r['type']}: {r['name']}")
                    retrieved_entities.append(r)
                context = "\n".join(context_parts)

        except Exception as e:
            logger.warning(f"[MedicalKnowledge] Vector search failed: {e}")

        try:
            messages = self.prompt.format_messages(query=query, context=context)
            response = self.llm.invoke(messages)
            answer = response.content

            logger.info(f"[MedicalKnowledge] Answer generated: {len(answer)} chars")

        except Exception as e:
            logger.error(f"[MedicalKnowledge] Error: {e}")
            answer = "抱歉，我无法回答这个问题。请咨询专业医生。"

        return {
            "knowledge_answer": answer,
            "retrieved_entities": retrieved_entities,
            "messages": [f"[MedicalKnowledge] Answer generated: {len(answer)} chars"]
        }


# Singleton instance
medical_knowledge = MedicalKnowledgeAgent()
