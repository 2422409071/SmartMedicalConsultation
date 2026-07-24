"""
Pre-visit Advisor Agent
Provides pre-visit advice and precautions
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.common.logger import setup_logger
from src.agents.state import AgentState

logger = setup_logger(__name__, "agents.log")


class PreVisitAdvisorAgent:
    """Pre-visit Advisor Agent"""

    def __call__(self, state: AgentState) -> dict:
        """Process state and return update"""
        symptoms = state.get("symptoms", [])
        departments = state.get("departments", [])
        logger.info(f"[PreVisitAdvisor] Symptoms: {symptoms}, Departments: {departments}")

        advice = []

        # General pre-visit advice
        advice.append("就诊前准备好医保卡、身份证")
        advice.append("记录症状出现的时间、频率、持续时间")
        advice.append("如有既往病历、检查报告，请一并携带")

        # Department-specific advice
        if departments:
            advice.append(f"建议挂号科室：{'、'.join(departments[:3])}")

        # Symptom-specific advice
        if "头痛" in symptoms or "头晕" in symptoms:
            advice.append("就诊前避免剧烈运动，保持充足睡眠")

        if any(s in symptoms for s in ["腹痛", "恶心", "呕吐"]):
            advice.append("如需做腹部检查，建议空腹就诊")

        # Try to get examinations from knowledge graph
        try:
            from src.knowledge_graph.graph_queries import GraphQueries
            queries = GraphQueries()

            for symptom in symptoms[:2]:
                diseases = queries.get_symptom_diseases(symptom)
                for disease in diseases[:1]:
                    exams = queries.get_disease_examinations(disease)
                    if exams:
                        advice.append(f"可能需要的检查：{'、'.join(exams[:3])}")
                        break

            queries.close()

        except Exception as e:
            logger.warning(f"[PreVisitAdvisor] KG query failed: {e}")

        return {
            "advice": advice,
            "messages": [f"[PreVisitAdvisor] Generated {len(advice)} advice items"]
        }


# Singleton instance
pre_visit_advisor = PreVisitAdvisorAgent()
