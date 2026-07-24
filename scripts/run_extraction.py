"""
Batch Entity Extraction Script
Reads processed disease data and extracts entities/relations using LLM.

Usage:
    python scripts/run_extraction.py
    python scripts/run_extraction.py --input data/processed/diseases.json
    python scripts/run_extraction.py --resume  # Resume from checkpoint
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from tqdm import tqdm

from config.paths import DATA_PROCESSED_DIR, DATA_KG_DIR, LOGS_DIR
from src.common.logger import setup_logger
from src.extraction.disease_extractor import DiseaseExtractor

logger = setup_logger(__name__, "extraction.log")


# ============================================================
# Paths
# ============================================================

INPUT_FILE = DATA_PROCESSED_DIR / "diseases.json"
ENTITIES_FILE = DATA_KG_DIR / "entities.json"
RELATIONS_FILE = DATA_KG_DIR / "relations.json"
CHECKPOINT_FILE = DATA_KG_DIR / ".extraction_checkpoint.json"


# ============================================================
# Checkpoint Management
# ============================================================

class CheckpointManager:
    """Manages extraction checkpoint for resume capability"""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.data = self._load()

    def _load(self) -> dict:
        """Load checkpoint from file"""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"[CHECKPOINT] Failed to load checkpoint: {e}")
        return {
            "processed_indices": [],
            "entities": [],
            "relations": [],
            "stats": {
                "total": 0,
                "success": 0,
                "failed": 0,
            }
        }

    def save(self):
        """Save checkpoint to file"""
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def is_processed(self, index: int) -> bool:
        """Check if index already processed"""
        return index in self.data["processed_indices"]

    def add_result(self, index: int, entities: list, relations: list, success: bool):
        """Add extraction result to checkpoint"""
        self.data["processed_indices"].append(index)
        if success:
            self.data["entities"].extend(entities)
            self.data["relations"].extend(relations)
            self.data["stats"]["success"] += 1
        else:
            self.data["stats"]["failed"] += 1

    def get_stats(self) -> dict:
        """Get current stats"""
        return self.data["stats"]

    def clear(self):
        """Clear checkpoint"""
        self.data = {
            "processed_indices": [],
            "entities": [],
            "relations": [],
            "stats": {"total": 0, "success": 0, "failed": 0}
        }
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()


# ============================================================
# Entity/Relation Serialization
# ============================================================

def serialize_entities(result) -> list[dict]:
    """Convert DiseaseExtractionResult entities to serializable dicts"""
    entities = []

    # Disease entity
    entities.append({
        "type": "Disease",
        "data": result.disease.model_dump()
    })

    # Symptoms
    for s in result.symptoms:
        entities.append({
            "type": "Symptom",
            "data": s.model_dump()
        })

    # Medications
    for m in result.medications:
        entities.append({
            "type": "Medication",
            "data": m.model_dump()
        })

    # Departments
    for d in result.departments:
        entities.append({
            "type": "Department",
            "data": d.model_dump()
        })

    # Examinations
    for e in result.examinations:
        entities.append({
            "type": "Examination",
            "data": e.model_dump()
        })

    # Treatments
    for t in result.treatments:
        entities.append({
            "type": "Treatment",
            "data": t.model_dump()
        })

    # Body Parts
    for bp in result.body_parts:
        entities.append({
            "type": "BodyPart",
            "data": bp.model_dump()
        })

    # Concepts
    for c in result.concepts:
        entities.append({
            "type": "MedicalConcept",
            "data": c.model_dump()
        })

    return entities


def serialize_relations(result) -> list[dict]:
    """Convert DiseaseExtractionResult relations to serializable dicts"""
    relations = []

    # HAS_SYMPTOM
    for r in result.has_symptom_relations:
        relations.append({
            "type": "HAS_SYMPTOM",
            "data": r.model_dump()
        })

    # TREATED_BY_DRUG
    for r in result.treated_by_relations:
        relations.append({
            "type": "TREATED_BY_DRUG",
            "data": r.model_dump()
        })

    # BELONG_TO_DEPARTMENT
    for r in result.department_relations:
        relations.append({
            "type": "BELONG_TO_DEPARTMENT",
            "data": r.model_dump()
        })

    # NEEDS_EXAMINATION
    for r in result.examination_relations:
        relations.append({
            "type": "NEEDS_EXAMINATION",
            "data": r.model_dump()
        })

    # AFFECTS_BODY_PART
    for r in result.body_part_relations:
        relations.append({
            "type": "AFFECTS_BODY_PART",
            "data": r.model_dump()
        })

    return relations


# ============================================================
# Main Extraction Logic
# ============================================================

def load_input_data(input_path: Path) -> list[dict]:
    """Load processed disease data"""
    if not input_path.exists():
        logger.error(f"[ERROR] Input file not found: {input_path}")
        return []

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        diseases = data.get("diseases", [])
        logger.info(f"[LOAD] Loaded {len(diseases)} diseases from {input_path}")
        return diseases

    except Exception as e:
        logger.error(f"[ERROR] Failed to load input: {e}")
        return []


def save_results(entities: list, relations: list, stats: dict):
    """Save extraction results to files"""
    DATA_KG_DIR.mkdir(parents=True, exist_ok=True)

    # Save entities
    entities_output = {
        "metadata": {
            "extracted_at": datetime.now().isoformat(),
            "total_entities": len(entities),
            "stats": stats
        },
        "entities": entities
    }
    with open(ENTITIES_FILE, "w", encoding="utf-8") as f:
        json.dump(entities_output, f, ensure_ascii=False, indent=2)
    logger.info(f"[SAVE] Entities saved to {ENTITIES_FILE}")

    # Save relations
    relations_output = {
        "metadata": {
            "extracted_at": datetime.now().isoformat(),
            "total_relations": len(relations)
        },
        "relations": relations
    }
    with open(RELATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(relations_output, f, ensure_ascii=False, indent=2)
    logger.info(f"[SAVE] Relations saved to {RELATIONS_FILE}")


def print_statistics(entities: list, relations: list, stats: dict):
    """Print extraction statistics"""
    print("\n" + "=" * 60)
    print("EXTRACTION STATISTICS")
    print("=" * 60)

    # Overall stats
    print(f"\n[OVERVIEW]")
    print(f"  Total records: {stats.get('total', 0)}")
    print(f"  Success: {stats.get('success', 0)}")
    print(f"  Failed: {stats.get('failed', 0)}")

    # Entity stats by type
    entity_counter = Counter(e["type"] for e in entities)
    print(f"\n[ENTITIES] Total: {len(entities)}")
    for etype, count in entity_counter.most_common():
        print(f"  {etype}: {count}")

    # Relation stats by type
    relation_counter = Counter(r["type"] for r in relations)
    print(f"\n[RELATIONS] Total: {len(relations)}")
    for rtype, count in relation_counter.most_common():
        print(f"  {rtype}: {count}")

    print("\n" + "=" * 60)


def run_extraction(
    input_path: Path = INPUT_FILE,
    resume: bool = False,
    use_llm: bool = True
):
    """
    Run batch entity extraction

    Args:
        input_path: Path to input JSON file
        resume: If True, resume from checkpoint
        use_llm: If True, use LLM extraction; else use dict-based extraction
    """
    print("=" * 60)
    print("BATCH ENTITY EXTRACTION")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Resume: {resume}")
    print(f"Use LLM: {use_llm}")
    print("=" * 60)

    # Load input data
    diseases = load_input_data(input_path)
    if not diseases:
        print("[ERROR] No data to process")
        return

    # Initialize checkpoint
    checkpoint = CheckpointManager(CHECKPOINT_FILE)
    if not resume:
        checkpoint.clear()
        logger.info("[CHECKPOINT] Cleared previous checkpoint")
    else:
        processed = len(checkpoint.data["processed_indices"])
        logger.info(f"[CHECKPOINT] Resuming from checkpoint, {processed} already processed")

    # Initialize extractor
    extractor = DiseaseExtractor(temperature=0.1)

    # Process each disease
    checkpoint.data["stats"]["total"] = len(diseases)

    for idx, disease_data in enumerate(tqdm(diseases, desc="Extracting", unit="disease")):
        # Skip if already processed
        if resume and checkpoint.is_processed(idx):
            continue

        disease_name = disease_data.get("name", f"disease_{idx}")

        try:
            # Build comprehensive text for LLM extraction
            # Include all available raw text for better extraction
            text_parts = [
                f"疾病名称：{disease_name}",
            ]

            if disease_data.get('description'):
                text_parts.append(f"疾病描述：{disease_data['description']}")

            if disease_data.get('symptoms'):
                text_parts.append(f"症状：{', '.join(disease_data['symptoms'])}")

            if disease_data.get('treatment'):
                # Truncate very long treatment text to avoid token limits
                treatment = disease_data['treatment']
                if len(treatment) > 1500:
                    treatment = treatment[:1500] + "..."
                text_parts.append(f"治疗方法：{treatment}")

            if disease_data.get('department'):
                text_parts.append(f"就诊科室：{disease_data['department']}")

            if disease_data.get('medications'):
                text_parts.append(f"常用药物：{', '.join(disease_data['medications'])}")

            text = "\n".join(text_parts)

            # Use LLM extraction for better structured data
            if use_llm:
                logger.info(f"[EXTRACT] Using LLM for: {disease_name}")
                result = extractor.extract(text)
            else:
                # Fallback to dict-based extraction (for testing/offline)
                extraction_input = {
                    "disease": {
                        "name": disease_name,
                        "description": disease_data.get("description"),
                    },
                    "symptoms": disease_data.get("symptoms", []),
                    "medications": disease_data.get("medications", []),
                    "departments": [disease_data.get("department", "")] if disease_data.get("department") else [],
                    "examinations": [],
                    "treatments": [disease_data.get("treatment", "")] if disease_data.get("treatment") else [],
                    "body_parts": []
                }
                result = extractor.extract_from_dict(extraction_input)

            # Serialize
            entities = serialize_entities(result)
            relations = serialize_relations(result)

            checkpoint.add_result(idx, entities, relations, success=True)
            logger.debug(f"[EXTRACT] {disease_name}: {len(entities)} entities, {len(relations)} relations")

        except Exception as e:
            logger.error(f"[EXTRACT] Failed for {disease_name}: {e}")
            checkpoint.add_result(idx, [], [], success=False)

        # Save checkpoint periodically
        if (idx + 1) % 10 == 0:
            checkpoint.save()

    # Final save
    checkpoint.save()

    # Save final results
    all_entities = checkpoint.data["entities"]
    all_relations = checkpoint.data["relations"]
    stats = checkpoint.get_stats()

    save_results(all_entities, all_relations, stats)

    # Print statistics
    print_statistics(all_entities, all_relations, stats)

    print("\n[SUCCESS] Extraction completed!")
    print(f"  Entities: {ENTITIES_FILE}")
    print(f"  Relations: {RELATIONS_FILE}")
    print(f"  Checkpoint: {CHECKPOINT_FILE}")


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch Entity Extraction for Medical Knowledge Graph"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(INPUT_FILE),
        help=f"Input JSON file path (default: {INPUT_FILE})"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if exists"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use dict-based extraction (no LLM calls)"
    )
    parser.add_argument(
        "--clear-checkpoint",
        action="store_true",
        help="Clear checkpoint before starting"
    )

    args = parser.parse_args()

    if args.clear_checkpoint:
        checkpoint = CheckpointManager(CHECKPOINT_FILE)
        checkpoint.clear()
        print("[INFO] Checkpoint cleared")

    run_extraction(
        input_path=Path(args.input),
        resume=args.resume,
        use_llm=not args.no_llm
    )


if __name__ == "__main__":
    main()
