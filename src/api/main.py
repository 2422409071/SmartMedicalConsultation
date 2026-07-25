"""
FastAPI Application Entry Point
Medical Consultation Assistant API
"""

import sys
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

# Ensure project root is in sys.path
_project_root = Path(__file__).parent.parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from src.common.logger import setup_logger
from src.api.routes import router as api_router

logger = setup_logger(__name__, "api.log")


# ============================================================
# Lifespan (Startup/Shutdown Events)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events"""
    # Startup
    logger.info("[API] Starting Medical Consultation Assistant API...")

    # Pre-load Agent system (optional, for faster first response)
    try:
        from src.agents.graph import app as agent_app
        logger.info("[API] Agent system loaded")
    except Exception as e:
        logger.warning(f"[API] Failed to pre-load agent system: {e}")

    # Pre-load vector index (optional)
    try:
        from src.vector_store.search import VectorRetriever
        retriever = VectorRetriever()
        logger.info("[API] Vector index loaded")
    except Exception as e:
        logger.warning(f"[API] Failed to pre-load vector index: {e}")

    logger.info("[API] Startup complete!")

    yield

    # Shutdown
    logger.info("[API] Shutting down...")


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="智能医疗问诊助手 API",
    version="1.0.0",
    description="基于知识图谱的智能医疗问诊系统",
    lifespan=lifespan
)

# CORS middleware (allow Streamlit frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",  # Streamlit default port
        "http://127.0.0.1:8501",
        "*"  # Allow all for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register API routes
app.include_router(api_router, prefix="/api")


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/", tags=["根路径"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "智能医疗问诊助手 API",
        "version": "1.0.0",
        "description": "基于知识图谱的智能医疗问诊系统",
        "endpoints": {
            "health": "/health",
            "consultation": "/api/process",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


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
    print("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=settings.api_port,
        log_level="info"
    )
