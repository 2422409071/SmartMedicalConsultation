"""
Disease Entity Extractor
Uses LLM to extract structured disease information from raw text.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
_project_root = Path(__file__).parent.parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from config.settings import settings
from src.common.llm import get_llm
from src.common.logger import setup_logger
from src.common.utils import retry_on_error
from src.extraction.schemas import (
    DiseaseEntity,
    SymptomEntity,
    MedicationEntity,
    DepartmentEntity,
    ExaminationEntity,
    TreatmentEntity,
    BodyPartEntity,
    MedicalConceptEntity,
    HasSymptomRelation,
    TreatedByDrugRelation,
    BelongsToDepartmentRelation,
    NeedsExaminationRelation,
    AffectsBodyPartRelation,
    DiseaseExtractionResult,
    SeverityLevel,
    FrequencyLevel,
    EvidenceLevel,
)

logger = setup_logger(__name__, "extraction.log")


class DiseaseExtractor:
    """
    疾病实体抽取器

    从原始医疗文本中抽取疾病及其相关实体和关系。

    Usage:
        >>> extractor = DiseaseExtractor()
        >>> result = extractor.extract("高血压是一种以动脉血压升高为特征的疾病...")
        >>> print(result.disease.name)
        >>> print(result.symptoms)
    """

    def __init__(self, temperature: float = 0.1):
        """
        初始化抽取器

        Args:
            temperature: LLM 温度参数，抽取任务建议低温度（0.1-0.3）
        """
        self.llm = get_llm(temperature=temperature)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的医疗信息抽取助手。请从以下文本中提取疾病及其相关信息。

要求：
1. 疾病名称使用标准医学术语
2. 症状列表要完整，包含所有提到的症状
3. 科室名称使用标准科室名称（如"心血管内科"而非"心内科"）
4. 药物名称使用通用名（如"氨氯地平"而非商品名）
5. 如果没有相关信息，对应字段返回空列表
6. 严格按照 JSON 格式输出"""),
            ("user", """请从以下文本中提取疾病信息：

{text}

请以 JSON 格式输出，包含以下字段：
{{
    "disease": {{
        "name": "疾病名称",
        "description": "疾病描述",
        "icd_code": "ICD编码（如知道）或 null"
    }},
    "symptoms": ["症状1", "症状2"],
    "medications": ["药物1", "药物2"],
    "departments": ["科室1"],
    "examinations": ["检查1", "检查2"],
    "treatments": ["治疗方案1"],
    "body_parts": ["身体部位1"]
}}""")
        ])

        # Use structured output for type safety
        self.chain = self.prompt | self.llm.with_structured_output(
            DiseaseExtractionResult,
            method="json_mode"
        )

    @retry_on_error(max_retries=3, delay=1.0)
    def extract(self, text: str) -> DiseaseExtractionResult:
        """
        从文本中抽取疾病实体

        Args:
            text: 原始医疗文本

        Returns:
            DiseaseExtractionResult: 包含疾病及所有相关实体和关系

        Raises:
            ValidationError: LLM 返回格式错误（会自动重试）
        """
        logger.info(f"[EXTRACT] Starting disease extraction, text length: {len(text)}")

        try:
            # Invoke the chain
            result = self.chain.invoke({"text": text})

            # Validate and log
            logger.info(f"[EXTRACT] Disease: {result.disease.name}")
            logger.info(f"[EXTRACT] Symptoms: {len(result.symptoms)}")
            logger.info(f"[EXTRACT] Medications: {len(result.medications)}")
            logger.info(f"[EXTRACT] Departments: {len(result.departments)}")

            return result

        except ValidationError as e:
            logger.error(f"[EXTRACT] Validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"[EXTRACT] Extraction failed: {e}")
            raise

    def extract_from_dict(self, data: dict) -> DiseaseExtractionResult:
        """
        从字典数据创建 DiseaseExtractionResult

        Args:
            data: 包含疾病信息的字典

        Returns:
            DiseaseExtractionResult
        """
        # Build disease entity
        disease = DiseaseEntity(
            name=data.get("disease", {}).get("name", ""),
            description=data.get("disease", {}).get("description"),
            icd_code=data.get("disease", {}).get("icd_code"),
        )

        # Build symptom entities
        symptoms = [
            SymptomEntity(name=s)
            for s in data.get("symptoms", [])
        ]

        # Build medication entities
        medications = [
            MedicationEntity(name=m)
            for m in data.get("medications", [])
        ]

        # Build department entities
        departments = [
            DepartmentEntity(name=d)
            for d in data.get("departments", [])
        ]

        # Build examination entities
        examinations = [
            ExaminationEntity(name=e)
            for e in data.get("examinations", [])
        ]

        # Build treatment entities
        treatments = [
            TreatmentEntity(name=t)
            for t in data.get("treatments", [])
        ]

        # Build body part entities
        body_parts = [
            BodyPartEntity(name=bp)
            for bp in data.get("body_parts", [])
        ]

        # Build relations
        has_symptom_relations = [
            HasSymptomRelation(
                disease_name=disease.name,
                symptom_name=s.name,
                frequency=FrequencyLevel.COMMON
            )
            for s in symptoms
        ]

        treated_by_relations = [
            TreatedByDrugRelation(
                disease_name=disease.name,
                drug_name=m.name,
                evidence_level=EvidenceLevel.B
            )
            for m in medications
        ]

        department_relations = [
            BelongsToDepartmentRelation(
                disease_name=disease.name,
                department_name=d.name,
                priority=1
            )
            for d in departments
        ]

        examination_relations = [
            NeedsExaminationRelation(
                disease_name=disease.name,
                examination_name=e.name,
                necessity="recommended"
            )
            for e in examinations
        ]

        body_part_relations = [
            AffectsBodyPartRelation(
                disease_name=disease.name,
                body_part_name=bp.name
            )
            for bp in body_parts
        ]

        return DiseaseExtractionResult(
            disease=disease,
            symptoms=symptoms,
            medications=medications,
            departments=departments,
            examinations=examinations,
            treatments=treatments,
            body_parts=body_parts,
            has_symptom_relations=has_symptom_relations,
            treated_by_relations=treated_by_relations,
            department_relations=department_relations,
            examination_relations=examination_relations,
            body_part_relations=body_part_relations,
            extraction_confidence=0.9
        )


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("Disease Extractor Test")
    print("=" * 60)

    # Test 1: Create extractor
    print("\n[TEST 1] Create DiseaseExtractor")
    extractor = DiseaseExtractor(temperature=0.1)
    print("[PASS] DiseaseExtractor created")

    # Test 2: Extract from dict (no LLM call)
    print("\n[TEST 2] Extract from dict")
    test_data = {
        "disease": {
            "name": "高血压",
            "description": "以体循环动脉血压增高为主要特征的临床综合征",
            "icd_code": "I10"
        },
        "symptoms": ["头痛", "头晕", "心悸"],
        "medications": ["氨氯地平", "缬沙坦"],
        "departments": ["心血管内科"],
        "examinations": ["血压测量", "心电图"],
        "treatments": ["药物治疗", "生活方式干预"],
        "body_parts": ["心脏", "血管"]
    }

    result = extractor.extract_from_dict(test_data)
    print(f"[PASS] Disease: {result.disease.name} ({result.disease.icd_code})")
    print(f"[PASS] Symptoms: {[s.name for s in result.symptoms]}")
    print(f"[PASS] Medications: {[m.name for m in result.medications]}")
    print(f"[PASS] Departments: {[d.name for d in result.departments]}")
    print(f"[PASS] HAS_SYMPTOM relations: {len(result.has_symptom_relations)}")
    print(f"[PASS] TREATED_BY relations: {len(result.treated_by_relations)}")

    print("\n" + "=" * 60)
    print("[SUCCESS] Disease extractor tests passed!")
    print("=" * 60)
