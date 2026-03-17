# Buscador Semantico Inteligente con RAG

Proyecto RAG para hacer preguntas sobre documentos propios con fuentes trazables.

## Arquitectura

- Ingestion: `PDF -> texto limpio -> chunks -> embeddings -> Pinecone`.
- Retrieval: busqueda hibrida (embedding denso + BM25 local) con fusion de rankings.
- Generation: respuesta con LLM (`openai` u `ollama`) usando solo contexto recuperado.
- Interfaces:
  - API FastAPI (`/ask`, `/upload`, `/documents`, `/ingest`).
  - Web UI basica en `http://localhost:8010/`.
  - Frontend Streamlit opcional.

## Busqueda optimizada

- Consultas densas multiples (query original + expansion semantica).
- Matching lexical ponderado (BM25) con variantes singular/plural.
- Expansion semantica ES/CA para terminos frecuentes (`tiempo muerto` <-> `temps mort`).
- Fuzzy matching por n-gramas para variaciones cercanas de escritura.
- Rerank final por cobertura de la pregunta y bonus numerico para preguntas de cantidad.
- Reintento de generacion cuando el LLM responde "no se" pese a haber evidencia en contexto.

## Estructura

```text
api/
  config.py
  main.py
  rag.py
  schemas.py
  service.py
  static/index.html
scripts/
  extract_text.py
  chunk_text.py
  index_pinecone.py
  ingest_all.py
frontend/
  app.py
tests/
  test_extract_text.py
  test_chunk_text.py
  test_service.py
  test_rag_utils.py
data/
  raw/
  processed/
  chunks/
```

## Instalacion

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## Configuracion

1. Copia `.env.example` a `.env`.
2. Rellena como minimo:

- `PINECONE_API_KEY`
- `PINECONE_INDEX`
- `PINECONE_NAMESPACE`
- `LLM_PROVIDER=openai` o `LLM_PROVIDER=ollama`
- `OPENAI_API_KEY` (si usas OpenAI)

Variables utiles:

- `RAG_TOP_K`
- `RAG_MAX_CONTEXT_CHARS`
- `EMBEDDING_MODEL` (recomendado: `paraphrase-multilingual-MiniLM-L12-v2` para ES/CA)
- `INGEST_CHUNK_SIZE`
- `INGEST_CHUNK_OVERLAP`
- `INGEST_BATCH_SIZE`
- `UPLOAD_MAX_SIZE_MB`

Validacion rapida de entorno:

```bash
python scripts/verify_setup.py
```

Diagnostico rapido del proveedor LLM:

```bash
python scripts/check_llm.py
```

## Ejecucion

### Opcion Rapida (Windows)

```powershell
.\start.ps1
```

Si PowerShell bloquea la ejecucion de scripts, usa:

```cmd
start.cmd
```

Con puerto custom:

```powershell
.\start.ps1 -Port 8010
```

En `cmd`:

```cmd
start.cmd -Port 8010
```

Sin validaciones previas:

```powershell
.\start.ps1 -SkipChecks
```

### Opcion A: API + Web UI

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8010
```

Abrir:

- `http://localhost:8010/`

### Opcion B: Pipeline por scripts

Coloca PDFs en `data/raw/` y ejecuta:

```bash
python scripts/ingest_all.py
```

El indexado limpia el namespace de Pinecone por defecto para evitar vectores obsoletos.
Si necesitas modo acumulativo, usa `python scripts/index_pinecone.py --append`.

### Opcion C: Streamlit

```bash
streamlit run frontend/app.py
```

## Test y validacion

```bash
python -m compileall api scripts frontend tests
python -m pytest -q tests --basetemp .pytest_tmp
```

## Docker

```bash
docker compose up --build
```

- API + Web UI: `http://localhost:8000`
- Streamlit: `http://localhost:8501`
