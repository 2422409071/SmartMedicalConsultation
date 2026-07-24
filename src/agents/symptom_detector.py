"""
Symptom Detector Agent
Extracts symptoms from user description and assesses severity
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.common.llm import get_llm
from src.common.logger import setup_logger
from src.agents.state import AgentState, SeverityLevel

logger = setup_logger(__name__, "agents.log")


class SymptomResult(BaseModel):
    """Symptom detection result"""
    symptoms: list[str] = Field(description="List of detected symptoms")
    severity: str = Field(description="Severity level: low/medium/high/emergency")
    is_emergency: bool = Field(description="Whether this is an emergency situation")


SYSTEM_PROMPT = """你是一个专业的症状分析专家。请从用户描述中提取症状信息。

任务：
1. 提取所有提到的症状（使用标准医学术语）
2. 判断整体严重程度
3. 检测是否为急症

严重程度定义：
- low：轻微症状，不影响日常生活（如偶尔头痛、轻微咳嗽）
- medium：中等症状，需要关注（如持续头痛、发烧、乏力）
- high：严重症状，需要尽快就医（如剧烈疼痛、持续高烧、严重乏力）
- emergency：急症，需要立即就医（如胸痛、呼吸困难、意识模糊、大出血）

急症关键词：
胸痛、胸闷、呼吸困难、喘不过气、意识模糊、昏迷、大出血、吐血、
剧烈腹痛、突然失明、肢体麻木无力、抽搐、高烧不退（>40°C）

示例：
用户：我最近总是头痛、头晕，有时候还会恶心
症状：["头痛", "头晕", "恶心"]
严重程度：medium
是否急症：false

用户：我突然胸口很痛，呼吸都困难
症状：["胸痛", "呼吸困难"]
严重程度：emergency
是否急症：true

请分析以下用户描述，提取症状信息。"""


class SymptomDetectorAgent:
    """Symptom Detector Agent"""

    def __init__(self):
        self.llm = get_llm(temperature=0.1)
        self.chain = self.llm.with_structured_output(SymptomResult)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", "{query}")
        ])

    def __call__(self, state: AgentState) -> dict:
        """Process state and return update"""
        query = state.get("query", "")
        logger.info(f"[SymptomDetector] Processing: {query[:50]}...")

        try:
            messages = self.prompt.format_messages(query=query)
            result = self.chain.invoke(messages)

            logger.info(f"[SymptomDetector] Found: {result.symptoms}, severity: {result.severity}")

            return {
                "symptoms": result.symptoms,
                "severity": result.severity,
                "is_emergency": result.is_emergency,
                "messages": [f"[SymptomDetector] Symptoms: {result.symptoms}, Severity: {result.severity}"]
            }

        except Exception as e:
            logger.error(f"[SymptomDetector] Error: {e}")
            return {
                "symptoms": [],
                "severity": SeverityLevel.MEDIUM,
                "is_emergency": False,
                "messages": [f"[SymptomDetector] Error: {e}"]
            }


# Singleton instance
symptom_detector = SymptomDetectorAgent()


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("SYMPTOM DETECTOR AGENT TEST")
    print("=" * 60)

    test_queries = [
        "我最近头痛、头晕，有时候还会恶心",
        "我突然胸痛、呼吸困难",
        "我有点咳嗽，喉咙痛"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        state = {"query": query}
        result = symptom_detector(state)
        print(f"  Symptoms: {result['symptoms']}")
        print(f"  Severity: {result['severity']}")
        print(f"  Is Emergency: {result['is_emergency']}")

    print("\n" + "=" * 60)
    print("[SUCCESS] Symptom detector tests completed!")
