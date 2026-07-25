"""
FastAPI Application Entry Point
Medical Consultation Assistant API with Agent System Integration
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

# Ensure project root is in sys.path
_project_root = Path(__file__).parent.parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config.settings import settings
from src.common.logger import setup_logger
from src.api.routes import router as api_router

logger = setup_logger(__name__, "api.log")


# ============================================================
# Global State
# ============================================================

class AppState:
    """Application state container"""
    agent_system_ready: bool = False
    vector_index_ready: bool = False
    neo4j_ready: bool = False
    request_semaphore: asyncio.Semaphore = None
    total_requests: int = 0
    total_errors: int = 0


app_state = AppState()


# ============================================================
# Lifespan (Startup/Shutdown Events)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events"""
    logger.info("[API] Starting Medical Consultation Assistant API...")

    # Initialize semaphore for concurrency control (max 5 concurrent requests)
    app_state.request_semaphore = asyncio.Semaphore(5)
    logger.info("[API] Concurrency limit: 5 requests")

    # 1. Initialize Agent system
    try:
        from src.agents.graph import app as agent_app
        app_state.agent_system_ready = True
        logger.info("[API] ✅ Agent system loaded")
    except Exception as e:
        app_state.agent_system_ready = False
        logger.error(f"[API] ❌ Failed to load agent system: {e}")

    # 2. Initialize Vector index
    try:
        from src.vector_store.search import VectorRetriever
        retriever = VectorRetriever()
        app_state.vector_index_ready = True
        logger.info("[API] ✅ Vector index loaded")
    except Exception as e:
        app_state.vector_index_ready = False
        logger.warning(f"[API] ⚠️ Failed to load vector index: {e}")

    # 3. Check Neo4j connection
    try:
        from src.knowledge_graph.graph_queries import GraphQueries
        queries = GraphQueries()
        stats = queries.get_statistics()
        app_state.neo4j_ready = stats.get("total_nodes", 0) > 0
        queries.close()
        logger.info(f"[API] ✅ Neo4j connected ({stats.get('total_nodes', 0)} nodes)")
    except Exception as e:
        app_state.neo4j_ready = False
        logger.warning(f"[API] ⚠️ Neo4j connection failed: {e}")

    # Summary
    logger.info("[API] " + "=" * 50)
    logger.info(f"[API] Agent System: {'✅ Ready' if app_state.agent_system_ready else '❌ Not Ready'}")
    logger.info(f"[API] Vector Index: {'✅ Ready' if app_state.vector_index_ready else '❌ Not Ready'}")
    logger.info(f"[API] Neo4j: {'✅ Ready' if app_state.neo4j_ready else '❌ Not Ready'}")
    logger.info("[API] " + "=" * 50)
    logger.info("[API] Startup complete!")

    yield

    # Shutdown
    logger.info("[API] Shutting down...")
    logger.info(f"[API] Total requests: {app_state.total_requests}")
    logger.info(f"[API] Total errors: {app_state.total_errors}")


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="智能医疗问诊助手 API",
    version="1.0.0",
    description="基于知识图谱的智能医疗问诊系统",
    lifespan=lifespan
)

# CORS middleware
# 开发模式（前端 :5173 跨域调 :8000）需要 CORS；
# 生产单进程部署时前端与 API 同源（都是 :8000），浏览器不触发 CORS。
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite 开发服务器
        "http://127.0.0.1:5173",
        "*"                       # 开发期放宽
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register API routes
app.include_router(api_router, prefix="/api")


# ============================================================
# 前端静态文件挂载（单进程部署）
# 必须在 /api 路由注册之后挂载，否则静态文件会拦截 API 请求。
# 路由优先级（按注册顺序）：/docs、/openapi.json → /api/* → /assets → SPA 兜底
# ============================================================

_frontend_dist = _project_root / "frontend" / "dist"

if _frontend_dist.exists() and (_frontend_dist / "index.html").exists():
    # 静态资源（JS/CSS/图片）由 /assets 显式托管，带正确 MIME 与缓存
    _assets_dir = _frontend_dist / "assets"
    if _assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="frontend-assets")

    # SPA 兜底：除 /api、/docs、/assets 外的所有路径都回退到 index.html
    # （支持 Vue Router history 模式刷新不 404；当前为单页，"/" 也由此返回 index.html）
    @app.get("/{full_path:path}", include_in_schema=False)
    async def _spa_fallback(full_path: str):
        return FileResponse(str(_frontend_dist / "index.html"))

    logger.info("[API] ✅ 前端已挂载 frontend/dist（单进程部署，访问 / 即为聊天界面）")
else:
    # 前端尚未构建时的占位根路由，避免 "/" 直接 404
    @app.get("/", tags=["根路径"], include_in_schema=False)
    async def _root_placeholder():
        return {
            "message": "前端尚未构建。请先执行：cd frontend && npm run build",
            "docs": "/docs",
            "api": {
                "consult": "/api/consult",
                "health": "/api/health",
                "stats": "/api/stats",
            },
        }

    logger.warning("[API] ⚠️ 未找到 frontend/dist，仅提供 API。请执行 cd frontend && npm run build 以启用界面。")


# ============================================================
# Main Entry
# ============================================================

if __name__ == "__main__":
    import uvicorn

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("智能医疗问诊助手 API")
    print("=" * 60)
    print(f"  Host: 0.0.0.0")
    print(f"  Port: {settings.api_port}")
    print(f"  Docs: http://localhost:{settings.api_port}/docs")
    print(f"  Concurrency Limit: 5")
    print(f"  Request Timeout: 30s")
    print("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=settings.api_port,
        log_level="info"
    )
