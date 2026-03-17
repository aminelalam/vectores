import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


def _convertir_a_entero(nombre: str, valor_predeterminado: int, minimo: int | None = None) -> int:
    valor_original = os.getenv(nombre)
    if valor_original is None or valor_original.strip() == "":
        valor_calculado = valor_predeterminado
    else:
        try:
            valor_calculado = int(valor_original)
        except ValueError as exc:
            raise ValueError(f"Invalid integer for {nombre}: {valor_original}") from exc

    if minimo is not None and valor_calculado < minimo:
        raise ValueError(f"{nombre} must be >= {minimo}. Got {valor_calculado}.")
    return valor_calculado


@dataclass(frozen=True)
class Settings:
    pinecone_api_key: str
    pinecone_index: str
    pinecone_namespace: str
    pinecone_cloud: str
    pinecone_region: str

    embedding_model: str
    rag_top_k: int
    rag_max_context_chars: int

    llm_provider: str
    openai_api_key: str
    openai_model: str
    ollama_base_url: str
    ollama_model: str
    llm_timeout_seconds: int

    ingest_chunk_size: int
    ingest_chunk_overlap: int
    ingest_batch_size: int
    upload_max_size_mb: int

    project_root: Path
    data_raw_dir: Path
    data_processed_dir: Path
    data_chunks_file: Path
    static_dir: Path
    static_index_file: Path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()

    raiz_proyecto = Path(__file__).resolve().parents[1]
    directorio_datos = raiz_proyecto / "data"
    directorio_estatico = Path(__file__).resolve().parent / "static"

    tamano_fragmento = _convertir_a_entero("INGEST_CHUNK_SIZE", 1000, minimo=1)
    solapamiento_fragmento = _convertir_a_entero("INGEST_CHUNK_OVERLAP", 200, minimo=0)
    if solapamiento_fragmento >= tamano_fragmento:
        raise ValueError("INGEST_CHUNK_OVERLAP must be lower than INGEST_CHUNK_SIZE.")

    return Settings(
        pinecone_api_key=os.getenv("PINECONE_API_KEY", "").strip(),
        pinecone_index=os.getenv("PINECONE_INDEX", "rag-semantic-search").strip(),
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "default").strip(),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws").strip(),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1").strip(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip(),
        rag_top_k=_convertir_a_entero("RAG_TOP_K", 5, minimo=1),
        rag_max_context_chars=_convertir_a_entero("RAG_MAX_CONTEXT_CHARS", 12000, minimo=1000),
        llm_provider=os.getenv("LLM_PROVIDER", "openai").strip().lower(),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip(),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1").strip(),
        llm_timeout_seconds=_convertir_a_entero("LLM_TIMEOUT_SECONDS", 120, minimo=1),
        ingest_chunk_size=tamano_fragmento,
        ingest_chunk_overlap=solapamiento_fragmento,
        ingest_batch_size=_convertir_a_entero("INGEST_BATCH_SIZE", 64, minimo=1),
        upload_max_size_mb=_convertir_a_entero("UPLOAD_MAX_SIZE_MB", 40, minimo=1),
        project_root=raiz_proyecto,
        data_raw_dir=directorio_datos / "raw",
        data_processed_dir=directorio_datos / "processed",
        data_chunks_file=directorio_datos / "chunks" / "chunks.jsonl",
        static_dir=directorio_estatico,
        static_index_file=directorio_estatico / "index.html",
    )
