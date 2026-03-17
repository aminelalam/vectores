import os
import sys

from dotenv import load_dotenv


def require(nombre: str) -> str:
    valor = os.getenv(nombre, "").strip()
    if not valor:
        raise ValueError(f"Missing required env var: {nombre}")
    return valor


def check_int(nombre: str, minimo: int = 0) -> int:
    valor_original = require(nombre)
    try:
        valor_entero = int(valor_original)
    except ValueError as exc:
        raise ValueError(f"{nombre} must be an integer. Got: {valor_original}") from exc
    if valor_entero < minimo:
        raise ValueError(f"{nombre} must be >= {minimo}. Got: {valor_entero}")
    return valor_entero


def main() -> int:
    load_dotenv()

    try:
        proveedor = require("LLM_PROVIDER").lower()
        require("PINECONE_API_KEY")
        require("PINECONE_INDEX")
        require("PINECONE_NAMESPACE")
        require("EMBEDDING_MODEL")
        check_int("RAG_TOP_K", minimo=1)
        check_int("RAG_MAX_CONTEXT_CHARS", minimo=1000)
        check_int("INGEST_CHUNK_SIZE", minimo=1)
        check_int("INGEST_CHUNK_OVERLAP", minimo=0)
        check_int("INGEST_BATCH_SIZE", minimo=1)
        check_int("UPLOAD_MAX_SIZE_MB", minimo=1)
        check_int("LLM_TIMEOUT_SECONDS", minimo=1)

        if proveedor == "openai":
            require("OPENAI_API_KEY")
            require("OPENAI_MODEL")
        elif proveedor == "ollama":
            require("OLLAMA_BASE_URL")
            require("OLLAMA_MODEL")
        else:
            raise ValueError("LLM_PROVIDER must be one of: openai, ollama")

        print("Environment validation successful.")
        return 0
    except Exception as exc:
        print(f"Environment validation failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
