from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from api.config import get_settings
from api.schemas import (
    AskRequest,
    AskResponse,
    DocumentListResponse,
    HealthResponse,
    IngestResponse,
    SourceItem,
    UploadResponse,
)
from api.service import AppService

ajustes_configuracion = get_settings()
servicio_aplicacion = AppService(ajustes=ajustes_configuracion)


@asynccontextmanager
async def lifespan(app: FastAPI):
    servicio_aplicacion.inicializar_motor()
    yield


app = FastAPI(title="RAG Semantic Search API", version="2.0.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
async def home() -> FileResponse:
    if not ajustes_configuracion.static_index_file.exists():
        raise HTTPException(status_code=404, detail="UI file not found.")
    return FileResponse(ajustes_configuracion.static_index_file)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(**servicio_aplicacion.obtener_estado_salud())


@app.post("/engine/reload", response_model=HealthResponse)
async def reload_engine() -> HealthResponse:
    return HealthResponse(**servicio_aplicacion.recargar_motor())


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    archivos_pdf = servicio_aplicacion.listar_documentos()
    return DocumentListResponse(count=len(archivos_pdf), files=archivos_pdf)


@app.post("/upload", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)) -> UploadResponse:
    saved = await servicio_aplicacion.guardar_subidas(files)
    return UploadResponse(
        message="Files uploaded successfully.",
        saved_files=saved,
        count=len(saved),
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest_documents() -> IngestResponse:
    resultado_ingestion = servicio_aplicacion.ingestar()
    return IngestResponse(**resultado_ingestion)


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    respuesta_generada, fragmentos_recuperados = servicio_aplicacion.consultar(
        pregunta=request.query, cantidad_fragmentos=request.top_k
    )
    fuentes_formateadas = [
        SourceItem(
            source=fragmento.source,
            page=fragmento.page,
            score=fragmento.score,
            snippet=(fragmento.text[:280] + "...") if len(fragmento.text) > 280 else fragmento.text,
        )
        for fragmento in fragmentos_recuperados
    ]
    return AskResponse(answer=respuesta_generada, sources=fuentes_formateadas)
