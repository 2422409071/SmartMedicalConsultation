"""
Department Recommender Agent
Recommends departments based on symptoms using Neo4j knowledge graph
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.common.logger import setup_logger
from src.agents.state import AgentState

logger = setup_logger(__name__, "agents.log")


class DepartmentRecommenderAgent:
    """Department Recommender Agent"""

    def __call__(self, state: AgentState) -> dict:
        """Process state and return update"""
        symptoms = state.get("symptoms", [])
        logger.info(f"[DepartmentRecommender] Symptoms: {symptoms}")

        departments = []

        try:
            from src.knowledge_graph.graph_queries import GraphQueries
            queries = GraphQueries()

            # Query departments for each symptom
            for symptom in symptoms:
                diseases = queries.get_symptom_diseases(symptom)
                for disease in diseases[:3]:  # Top 3 diseases per symptom
                    depts = queries.get_disease_departments(disease)
                    departments.extend(depts)

            queries.close()

            # Remove duplicates and limit
            departments = list(dict.fromkeys(departments))[:5]

            logger.info(f"[DepartmentRecommender] Recommended: {departments}")

        except Exception as e:
            logger.error(f"[DepartmentRecommender] Error: {e}")
            # Fallback: general recommendations
            departments = ["内科", "急诊科"]

        return {
            "departments": departments,
            "messages": [f"[DepartmentRecommender] Recommended: {departments}"]
        }


# Singleton instance
department_recommender = DepartmentRecommenderAgent()


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("DEPARTMENT RECOMMENDER AGENT TEST")
    print("=" * 60)

    state = {"symptoms": ["头痛", "头晕"]}
    result = department_recommender(state)
    print(f"Symptoms: {state['symptoms']}")
    print(f"Recommended departments: {result['departments']}")

    print("\n" + "=" * 60)
