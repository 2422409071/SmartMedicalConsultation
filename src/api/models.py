"""
API Request/Response Models
Pydantic models for API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ============================================================
# Request Models
# ============================================================

class ConsultationRequest(BaseModel):
    """Consultation request model"""
    query: str = Field(..., description="User question", min_length=1, max_length=1000)
    session_id: str = Field(default="default", description="Session ID for tracking")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "我最近头痛、头晕，应该挂什么科？",
                "session_id": "user123_session1"
            }
        }


# ============================================================
# Response Models
# ============================================================

class ConsultationResponse(BaseModel):
    """Consultation response model"""
    answer: str = Field(..., description="AI generated answer")
    intent: str = Field(default="", description="Detected intent type")
    symptoms: list[str] = Field(default_factory=list, description="Detected symptoms")
    departments: list[str] = Field(default_factory=list, description="Recommended departments")
    medications: list[dict] = Field(default_factory=list, description="Medication advice")
    disclaimers: list[str] = Field(default_factory=list, description="Medical disclaimers")
    warnings: list[str] = Field(default_factory=list, description="Emergency warnings")
    linked_entities: list[dict] = Field(
        default_factory=list,
        description="实体链接结果：用户口语词 → 图谱规范实体 "
                    "[{input_entity, matched_entity, type, similarity}]"
    )
    duration_ms: int = Field(default=0, description="Processing time in milliseconds")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status: healthy/degraded/unhealthy")
    neo4j_connected: bool = Field(default=False, description="Neo4j connection status")
    vector_index_loaded: bool = Field(default=False, description="Vector index status")
    agent_system_ready: bool = Field(default=False, description="Agent system status")
    version: str = Field(default="1.0.0")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class StatsResponse(BaseModel):
    """System statistics response model"""
    knowledge_graph: dict = Field(default_factory=dict, description="Knowledge graph stats")
    vector_index: dict = Field(default_factory=dict, description="Vector index stats")
    api: dict = Field(default_factory=dict, description="API stats")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Error detail")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
