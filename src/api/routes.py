"""
API Routes
FastAPI router with consultation, health, and stats endpoints.
"""

import sys
import time
from pathlib import Path
from datetime import datetime

_project_root = Path(__file__).parent.parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from fastapi import APIRouter, HTTPException

from src.common.logger import setup_logger
from src.api.models import (
    ConsultationRequest,
    ConsultationResponse,
    HealthResponse,
    StatsResponse,
    ErrorResponse
)

logger = setup_logger(__name__, "api.log")

router = APIRouter()


# ============================================================
# POST /api/consult - Consultation Endpoint
# ============================================================

@router.post(
    "/consult",
    response_model=ConsultationResponse,
    responses={500: {"model": ErrorResponse}},
    tags=["问诊"],
    summary="智能问诊接口",
    description="接收用户问题，返回 AI 生成的医疗建议"
)
async def consult(req: ConsultationRequest):
    """
    Process user consultation request

    - Accepts user question
    - Runs through Agent system
    - Returns medical advice with disclaimers
    """
    start_time = time.time()
    logger.info(f"[API] Consultation request: {req.query[:50]}... (session: {req.session_id})")

    try:
        # Call Agent system
        from src.agents.graph import run

        result = run(req.query)

        duration_ms = int((time.time() - start_time) * 1000)

        response = ConsultationResponse(
            answer=result.get("final_answer", ""),
            intent=result.get("intent", ""),
            symptoms=result.get("symptoms", []),
            departments=result.get("departments", []),
            medications=result.get("medications", []),
            disclaimers=result.get("disclaimers", []),
            warnings=result.get("warnings", []),
            duration_ms=duration_ms
        )

        logger.info(f"[API] Consultation completed in {duration_ms}ms")
        return response

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error(f"[API] Consultation failed: {e}")

        # Return friendly error
        return ConsultationResponse(
            answer="抱歉，系统暂时无法处理您的请求。请稍后重试，或拨打医疗咨询热线。",
            intent="error",
            disclaimers=["🔴 重要声明：本建议仅供参考，不能替代专业医疗诊断。"],
            duration_ms=duration_ms
        )


# ============================================================
# GET /api/health - Health Check Endpoint
# ============================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["系统"],
    summary="健康检查",
    description="检查服务状态、Neo4j 连接、模型加载状态"
)
async def health():
    """
    Health check endpoint

    Returns status of all system components.
    """
    logger.debug("[API] Health check requested")

    # Check Neo4j connection
    neo4j_connected = False
    try:
        from src.knowledge_graph.graph_queries import GraphQueries
        queries = GraphQueries()
        stats = queries.get_statistics()
        neo4j_connected = stats.get("total_nodes", 0) > 0
        queries.close()
    except Exception as e:
        logger.warning(f"[API] Neo4j check failed: {e}")

    # Check vector index
    vector_index_loaded = False
    try:
        from config.paths import DATA_INDEXES_DIR
        vector_index_loaded = (DATA_INDEXES_DIR / "faiss.index").exists()
    except Exception as e:
        logger.warning(f"[API] Vector index check failed: {e}")

    # Check agent system
    agent_system_ready = False
    try:
        from src.agents.graph import app as agent_app
        agent_system_ready = agent_app is not None
    except Exception as e:
        logger.warning(f"[API] Agent system check failed: {e}")

    # Determine overall status
    if neo4j_connected and vector_index_loaded and agent_system_ready:
        status = "healthy"
    elif agent_system_ready:
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        neo4j_connected=neo4j_connected,
        vector_index_loaded=vector_index_loaded,
        agent_system_ready=agent_system_ready
    )


# ============================================================
# GET /api/stats - System Statistics Endpoint
# ============================================================

@router.get(
    "/stats",
    response_model=StatsResponse,
    tags=["系统"],
    summary="系统统计",
    description="返回知识图谱、向量索引、API 统计信息"
)
async def stats():
    """
    System statistics endpoint

    Returns knowledge graph stats, vector index stats, and API stats.
    """
    logger.debug("[API] Stats requested")

    # Knowledge graph stats
    kg_stats = {}
    try:
        from src.knowledge_graph.graph_queries import GraphQueries
        queries = GraphQueries()
        kg_data = queries.get_statistics()
        kg_stats = {
            "total_nodes": kg_data.get("total_nodes", 0),
            "total_relations": kg_data.get("total_relations", 0),
            "nodes_by_type": kg_data.get("nodes", {}),
            "relations_by_type": kg_data.get("relations", {}),
            "density": kg_data.get("density", 0)
        }
        queries.close()
    except Exception as e:
        logger.warning(f"[API] KG stats failed: {e}")
        kg_stats = {"error": str(e)}

    # Vector index stats
    vi_stats = {}
    try:
        from config.paths import DATA_INDEXES_DIR
        import json

        index_file = DATA_INDEXES_DIR / "faiss.index"
        entities_file = DATA_INDEXES_DIR / "entities.json"

        if index_file.exists():
            import faiss
            index = faiss.read_index(str(index_file))
            vi_stats["total_vectors"] = index.ntotal
            vi_stats["dimension"] = index.d

        if entities_file.exists():
            with open(entities_file, encoding="utf-8") as f:
                entities = json.load(f)
            vi_stats["total_entities"] = len(entities)

    except Exception as e:
        logger.warning(f"[API] Vector stats failed: {e}")
        vi_stats = {"error": str(e)}

    # API stats
    api_stats = {
        "version": "1.0.0",
        "uptime": "N/A"
    }

    return StatsResponse(
        knowledge_graph=kg_stats,
        vector_index=vi_stats,
        api=api_stats
    )
