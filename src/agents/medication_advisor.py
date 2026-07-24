"""
Medication Advisor Agent
Provides medication advice using Neo4j knowledge graph
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.common.logger import setup_logger
from src.agents.state import AgentState

logger = setup_logger(__name__, "agents.log")


class MedicationAdvisorAgent:
    """Medication Advisor Agent"""

    def __call__(self, state: AgentState) -> dict:
        """Process state and return update"""
        query = state.get("query", "")
        symptoms = state.get("symptoms", [])
        logger.info(f"[MedicationAdvisor] Query: {query[:50]}...")

        medications = []

        try:
            from src.knowledge_graph.graph_queries import GraphQueries
            queries = GraphQueries()

            # Try to find disease from query or symptoms
            diseases = []
            for symptom in symptoms:
                diseases.extend(queries.get_symptom_diseases(symptom))

            # Get medications for found diseases
            for disease in list(dict.fromkeys(diseases))[:3]:
                meds = queries.get_disease_medications(disease)
                for med in meds:
                    medications.append({
                        "name": med.get("name", ""),
                        "disease": disease,
                        "usage": "请遵医嘱",
                        "side_effects": med.get("side_effects", []),
                        "contraindications": med.get("contraindications", [])
                    })

            queries.close()

            # Limit results
            medications = medications[:5]

            logger.info(f"[MedicationAdvisor] Found {len(medications)} medications")

        except Exception as e:
            logger.error(f"[MedicationAdvisor] Error: {e}")

        return {
            "medications": medications,
            "messages": [f"[MedicationAdvisor] Found {len(medications)} medications"]
        }


# Singleton instance
medication_advisor = MedicationAdvisorAgent()
