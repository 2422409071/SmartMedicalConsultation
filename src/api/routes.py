"""
API Routes
FastAPI router with consultation, health, and stats endpoints.
Includes timeout handling, concurrency control, and request logging.
"""

import sys
import time
import asyncio
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

# Timeout configuration
REQUEST_TIMEOUT = 30  # seconds


# ============================================================
# Request Logging Helper
# ============================================================

def log_request(
    query: str,
    intent: str,
    duration_ms: int,
    has_disclaimer: bool,
    success: bool
):
    """Log request details for monitoring"""
    status = "✅" if success else "❌"
    logger.info(
        f"[REQUEST] {status} "
        f"query='{query[:30]}...' | "
        f"intent={intent} | "
        f"duration={duration_ms}ms | "
        f"disclaimer={'Y' if has_disclaimer else 'N'}"
    )


# ============================================================
# POST /api/consult - Consultation Endpoint
# ============================================================

@router.post(
    "/consult",
    response_model=ConsultationResponse,
    responses={
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse}
    },
    tags=["问诊"],
    summary="智能问诊接口",
    description="接收用户问题，返回 AI 生成的医疗建议（30秒超时，5并发限制）"
)
async def consult(req: ConsultationRequest):
    """
    Process user consultation request

    - Accepts user question
    - Runs through Agent system with timeout and concurrency control
    - Returns medical advice with disclaimers
    """
    from src.api.main import app_state

    # Check if agent system is ready
    if not app_state.agent_system_ready:
        raise HTTPException(
            status_code=503,
            detail="Agent system not ready. Please try again later."
        )

    start_time = time.time()
    app_state.total_requests += 1

    logger.info(f"[API] Consultation request: '{req.query[:50]}...' (session: {req.session_id})")

    try:
        # Acquire semaphore for concurrency control
        async with app_state.request_semaphore:
            # Call Agent system with timeout
            from src.agents.graph import run

            try:
                # Run in thread pool to not block event loop
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, run, req.query
                    ),
                    timeout=REQUEST_TIMEOUT
                )

                duration_ms = int((time.time() - start_time) * 1000)

                # Build response
                disclaimers = result.get("disclaimers", [])
                response = ConsultationResponse(
                    answer=result.get("final_answer", ""),
                    intent=result.get("intent", ""),
                    symptoms=result.get("symptoms", []),
                    departments=result.get("departments", []),
                    medications=result.get("medications", []),
                    disclaimers=disclaimers,
                    warnings=result.get("warnings", []),
                    duration_ms=duration_ms
                )

                # Log request
                log_request(
                    query=req.query,
                    intent=response.intent,
                    duration_ms=duration_ms,
                    has_disclaimer=len(disclaimers) > 0,
                    success=True
                )

                return response

            except asyncio.TimeoutError:
                duration_ms = int((time.time() - start_time) * 1000)
                app_state.total_errors += 1

                log_request(
                    query=req.query,
                    intent="timeout",
                    duration_ms=duration_ms,
                    has_disclaimer=False,
                    success=False
                )

                raise HTTPException(
                    status_code=504,
                    detail=f"Request timeout after {REQUEST_TIMEOUT}s. Please try a simpler question."
                )

    except HTTPException:
        raise
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        app_state.total_errors += 1

        log_request(
            query=req.query,
            intent="error",
            duration_ms=duration_ms,
            has_disclaimer=False,
            success=False
        )

        logger.error(f"[API] Consultation failed: {e}")

        # Return friendly error
        return ConsultationResponse(
            answer="抱歉，系统暂时无法处理您的请求。请稍后重试，或拨打医疗咨询热线 12320。",
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
    from src.api.main import app_state

    # Determine overall status
    if app_state.agent_system_ready and app_state.vector_index_ready and app_state.neo4j_ready:
        status = "healthy"
    elif app_state.agent_system_ready:
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        neo4j_connected=app_state.neo4j_ready,
        vector_index_loaded=app_state.vector_index_ready,
        agent_system_ready=app_state.agent_system_ready
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
    from src.api.main import app_state

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
            "density": round(kg_data.get("density", 0), 2)
        }
        queries.close()
    except Exception as e:
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
        vi_stats = {"error": str(e)}

    # API stats
    api_stats = {
        "version": "1.0.0",
        "total_requests": app_state.total_requests,
        "total_errors": app_state.total_errors,
        "concurrency_limit": 5,
        "request_timeout": REQUEST_TIMEOUT
    }

    return StatsResponse(
        knowledge_graph=kg_stats,
        vector_index=vi_stats,
        api=api_stats
    )
