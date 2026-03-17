from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=3, description="User question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")


class SourceItem(BaseModel):
    source: str
    page: int
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


class HealthResponse(BaseModel):
    status: str
    details: str | None = None
    index: str | None = None


class DocumentListResponse(BaseModel):
    count: int
    files: list[str]


class UploadResponse(BaseModel):
    message: str
    saved_files: list[str]
    count: int


class IngestResponse(BaseModel):
    message: str
    indexed_files: list[str]
    engine_ready: bool
    logs: list[str]
