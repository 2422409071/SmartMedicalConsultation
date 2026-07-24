"""
Semantic Search Interface
Provides VectorRetriever class for hybrid search (FAISS + Neo4j).
"""

import sys
import json
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
_project_root = Path(__file__).parent.parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import faiss

from config.settings import settings
from config.paths import DATA_INDEXES_DIR
from src.common.logger import setup_logger

logger = setup_logger(__name__, "vector_search.log")


# ============================================================
# Paths
# ============================================================

FAISS_INDEX_FILE = DATA_INDEXES_DIR / "faiss.index"
ENTITIES_MAPPING_FILE = DATA_INDEXES_DIR / "entities.json"


# ============================================================
# Vector Retriever
# ============================================================

class VectorRetriever:
    """
    Vector Retriever with Hybrid Search

    Combines FAISS semantic search with Neo4j graph search for better results.

    Usage:
        >>> retriever = VectorRetriever()
        >>> results = retriever.search("高血压的症状")
        >>> for r in results:
        ...     print(f"{r['name']} ({r['type']}): {r['score']:.4f}")
    """

    def __init__(
        self,
        index_path: Path = FAISS_INDEX_FILE,
        entities_path: Path = ENTITIES_MAPPING_FILE,
        score_threshold: float = 0.5
    ):
        """
        Initialize the vector retriever

        Args:
            index_path: Path to FAISS index file
            entities_path: Path to entity mapping JSON file
            score_threshold: Minimum similarity score (default: 0.5)
        """
        self.index_path = index_path
        self.entities_path = entities_path
        self.score_threshold = score_threshold

        self.index = None
        self.entities = []
        self.embedding_model = None

        # Load index and entities
        self._load_index()

    def _load_index(self):
        """Load FAISS index and entity mapping"""
        # Load FAISS index
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            logger.info(f"[LOAD] FAISS index loaded: {self.index.ntotal} vectors")
        else:
            logger.error(f"[LOAD] FAISS index not found: {self.index_path}")
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")

        # Load entity mapping
        if self.entities_path.exists():
            with open(self.entities_path, "r", encoding="utf-8") as f:
                self.entities = json.load(f)
            logger.info(f"[LOAD] Entity mapping loaded: {len(self.entities)} entities")
        else:
            logger.error(f"[LOAD] Entity mapping not found: {self.entities_path}")
            raise FileNotFoundError(f"Entity mapping not found: {self.entities_path}")

    def _get_embedding_model(self):
        """Lazy load embedding model"""
        if self.embedding_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"[MODEL] Loading BGE-M3 from {settings.embedding_model_path}")
            self.embedding_model = SentenceTransformer(settings.embedding_model_path)
            logger.info("[MODEL] BGE-M3 loaded")
        return self.embedding_model

    def search(
        self,
        query: str,
        top_k: int = 5,
        use_graph: bool = True
    ) -> list[dict]:
        """
        Hybrid search: FAISS semantic search + Neo4j graph search

        Args:
            query: Search query text
            top_k: Number of results to return
            use_graph: Whether to use Neo4j graph search

        Returns:
            List of result dicts with name, type, score, source
        """
        results = []

        # 1. FAISS semantic search
        vector_results = self._vector_search(query, top_k * 2)  # Get more for merging
        results.extend(vector_results)

        # 2. Neo4j graph search (exact match)
        if use_graph:
            graph_results = self._graph_search(query, top_k)
            results.extend(graph_results)

        # 3. Merge and deduplicate
        merged = self._merge_results(results, top_k)

        return merged

    def _vector_search(self, query: str, top_k: int) -> list[dict]:
        """FAISS semantic search"""
        if self.index is None:
            return []

        # Encode query
        model = self._get_embedding_model()
        query_vec = model.encode([query], normalize_embeddings=True).astype("float32")

        # Search
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.entities) and score >= self.score_threshold:
                entity = self.entities[idx].copy()
                results.append({
                    "name": entity.get("name", ""),
                    "type": entity.get("type", ""),
                    "score": float(score),
                    "source": "vector",
                    "data": entity.get("data", {})
                })

        return results

    def _graph_search(self, query: str, top_k: int) -> list[dict]:
        """Neo4j graph search (exact match on entity name)"""
        try:
            from src.knowledge_graph.graph_queries import GraphQueries
            queries = GraphQueries()

            results = []

            # Try exact disease match
            disease_info = queries.get_disease_info(query)
            if disease_info:
                results.append({
                    "name": disease_info["name"],
                    "type": "Disease",
                    "score": 1.0,
                    "source": "graph",
                    "data": disease_info
                })

            # Try symptom match
            symptom_diseases = queries.get_symptom_diseases(query)
            for disease in symptom_diseases[:3]:
                results.append({
                    "name": disease,
                    "type": "Disease",
                    "score": 0.9,
                    "source": "graph",
                    "data": {"matched_symptom": query}
                })

            queries.close()
            return results

        except Exception as e:
            logger.warning(f"[GRAPH] Graph search failed: {e}")
            return []

    def _merge_results(self, results: list[dict], top_k: int) -> list[dict]:
        """Merge and deduplicate results"""
        # Group by name
        by_name = {}
        for r in results:
            name = r["name"]
            if name not in by_name:
                by_name[name] = r
            else:
                # Keep higher score
                if r["score"] > by_name[name]["score"]:
                    by_name[name] = r
                # Prefer graph source for same score
                elif r["score"] == by_name[name]["score"] and r["source"] == "graph":
                    by_name[name] = r

        # Sort by score and take top_k
        merged = sorted(by_name.values(), key=lambda x: x["score"], reverse=True)
        return merged[:top_k]

    def search_diseases(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for diseases only"""
        results = self.search(query, top_k * 2, use_graph=True)
        return [r for r in results if r["type"] == "Disease"][:top_k]

    def search_symptoms(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for symptoms only"""
        results = self.search(query, top_k * 2, use_graph=False)
        return [r for r in results if r["type"] == "Symptom"][:top_k]

    def search_medications(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for medications only"""
        results = self.search(query, top_k * 2, use_graph=False)
        return [r for r in results if r["type"] in ("Medication", "Drug")]


# ============================================================
# Main: Self-test
# ============================================================

if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("VECTOR RETRIEVER TEST")
    print("=" * 60)

    # Initialize retriever
    print("\n[INIT] Loading VectorRetriever...")
    retriever = VectorRetriever()
    print("[INIT] VectorRetriever loaded")

    # Test queries
    test_queries = [
        "高血压的症状",
        "头痛挂什么科",
        "阿司匹林的副作用"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: \"{query}\"")
        print("=" * 60)

        results = retriever.search(query, top_k=5)

        if results:
            print("Results:")
            for i, r in enumerate(results, 1):
                print(f"  {i}. [{r['type']}] {r['name']} (score: {r['score']:.4f}, source: {r['source']})")
        else:
            print("  No results found")

    print("\n" + "=" * 60)
    print("[SUCCESS] Vector retriever tests completed!")
    print("=" * 60)
