<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h1>🔍 Ask Questions About Your PDFs</h1>
  <h3>RAG Semantic Search</h3>

  <p>
    Ask questions about your own PDFs and get concise, grounded answers with traceable sources.
  </p>
</div>
<!-- BADGES -->
<p align="center">
  <img alt="Python Version" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.111.0%2B-009688.svg?logo=fastapi">
  <img alt="LangChain" src="https://img.shields.io/badge/LangChain-0.2.0%2B-blue.svg">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg">
</p>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#ℹ️-about-the-project">About The Project</a>
      <ul>
        <li><a href="#🎯-what-i-built-and-why">What I Built and Why</a></li>
        <li><a href="#🌟-highlights">Highlights</a></li>
        <li><a href="#🛠️-built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#🏁-getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#🏗️-architecture">Architecture</a></li>
    <li><a href="#🚀-usage">Usage</a></li>
    <li><a href="#🗺️-roadmap">Roadmap</a></li>
    <li><a href="#🤝-contributing">Contributing</a></li>
    <li><a href="#📝-license">License</a></li>
    <li><a href="#📫-contact">Contact</a></li>
  </ol>
</details>

## ℹ️ About The Project

This project implements a highly robust **Retrieval-Augmented Generation (RAG)** search system for PDF documents. It transforms how you interact with your personal or corporate document sets by combining advanced vector search with local lexical indexing.

Whether you are performing deep research, analyzing financial reports, or studying textbooks, this tool provides precise, typo-tolerant, and context-aware answers.

### 🎯 What I Built and Why

I built this project to answer real questions from real PDFs without losing trust in the results. By employing strict RAG prompting, the LLM generates answers exclusively based on retrieved context, ensuring that every output is grounded and auditable with precise source citations.

Here is what I chose to build and why:
- **PDF Ingestion Pipeline**: Cleans text, chunks it, and builds embeddings so documents are fully searchable and the process is repeatable.
- **Pinecone Vector Database**: Chosen for vector indexing so semantic search stays fast even as the document set grows.
- **Local BM25 Index**: Added because exact keyword matches still matter and dense embeddings alone can miss them.
- **Reciprocal Rank Fusion**: Blends dense and lexical results so one signal never overwhelms the other.
- **Query Expansion & Fuzzy Matching**: Handles singular/plural forms, ES/CA terminology variants, and small typos gracefully.
- **Coverage-Aware Reranking**: Re-ranks by query coverage and numeric signals so the top chunks are the ones most likely to contain the answer, especially for quantity questions.
- **Strict RAG Prompting**: Uses only retrieved context and cites precise sources so answers are grounded and auditable.
- **Multiple Interfaces**: Exposed the system through FastAPI, a simple Web UI, and an optional Streamlit app so developers and non-technical users can work with it effortlessly.

### 🌟 Highlights

Here's why this RAG pipeline stands out:

*   **Hybrid Retrieval:** Merges dense embeddings (`Pinecone`) with local `BM25`, fused seamlessly using Reciprocal Rank Fusion.
*   **Intelligent Query Expansion & Fuzzy Matching:** Automatically handles singular/plural forms, ES/CA term variants, and gracefully catches small typos.
*   **Coverage-Aware Reranking:** Includes custom numeric boosting to vastly improve the accuracy of quantity and math-related queries.
*   **Strict Source Grounding:** LLM generation uses *only* retrieved context with fully traceable source references.
*   **Flexible Access:** Interact via a blazing-fast FastAPI backend, a sleek Web UI, or an optional interactive Streamlit application.
*   **End-to-End Ingestion:** Fully automated pipeline taking raw PDFs through text extraction, semantic chunking, embedding generation, and Pinecone ingestion.
*   **LLM Agnostic:** Switch effortlessly between your local Ollama instance and the OpenAI API with a simple environment variable toggle.

### 🛠️ Built With

This project relies on modern, industry-standard tools for AI, backend engineering, and retrieval.

*   [![Python][Python-shield]][Python-url]
*   [![FastAPI][FastAPI-shield]][FastAPI-url]
*   [![LangChain][LangChain-shield]][LangChain-url]
*   [![Pinecone][Pinecone-shield]][Pinecone-url]
*   [![Streamlit][Streamlit-shield]][Streamlit-url]
*   [![Docker][Docker-shield]][Docker-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 🏁 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

*   Python 3.10+
*   A [Pinecone](https://www.pinecone.io/) account and API key
*   Either an **OpenAI API Key** or a local **Ollama** instance

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your_username/rag-semantic-search.git
   cd rag-semantic-search
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate
   # Linux/macOS
   source .venv/bin/activate
   ```

3. Install required packages:
   ```sh
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory (using the configuration below as a guide):
   ```env
   # Pinecone Configuration
   PINECONE_API_KEY=YOUR_PINECONE_KEY
   PINECONE_INDEX=rag-semantic-search
   PINECONE_NAMESPACE=default
   PINECONE_CLOUD=aws
   PINECONE_REGION=us-east-1

   # LLM Provider Options: "openai" or "ollama"
   LLM_PROVIDER=openai
   OPENAI_API_KEY=YOUR_OPENAI_KEY
   OPENAI_MODEL=gpt-4o-mini
   # OLLAMA_BASE_URL=http://localhost:11434
   # OLLAMA_MODEL=llama3.1
   
   # Emdeddings & Tuning
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   RAG_TOP_K=5
   RAG_MAX_CONTEXT_CHARS=12000
   
   # Ingestion
   INGEST_CHUNK_SIZE=1000
   INGEST_CHUNK_OVERLAP=200
   INGEST_BATCH_SIZE=64
   UPLOAD_MAX_SIZE_MB=40
   ```
   > [!NOTE]
   > For `LLM_PROVIDER=openai`, make sure to add the `OPENAI_API_KEY` and model. For `ollama`, set up `OLLAMA_BASE_URL` and `OLLAMA_MODEL`. If working heavily in Spanish/Catalan, consider a multilingual embedding model.

5. Run the application:
   ```powershell
   # Windows Execution Script
   .\start.ps1
   
   # Alternatively via Uvicorn (Linux/macOS compatible)
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8010
   ```

### Using Docker

Alternatively, spin up the entire stack seamlessly using Docker Compose:

```sh
docker compose up --build
```
Once deployed, access the services here:
*   **API & Web UI:** `http://localhost:8000/`


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 🏗️ Architecture

The pipeline consists of a sequential processing backend coupled with a real-time retrieval system.

1.  **Ingestion & Processing:** Raw `PDF` files are extracted to clear text, split into semantic chunks, vectorized using models like `all-MiniLM-L6-v2`, and finally stored inside `Pinecone`.
2.  **Hybrid Retrieval:** User queries undergo search strategies combining Dense Vectors inside Pinecone alongside a Local BM25 Lexical Index.
3.  **Merge & Rerank:** Retrieved contexts are subsequently merged using `Reciprocal Rank Fusion`. Query Coverage thresholds and specialized Numeric Signals boost document accuracy significantly.
4.  **Generative Output:** The curated chunks securely instruct the respective LLM endpoint (such as `gpt-4o-mini` or `llama-3.1`) generating concise Spanish responses mapped exclusively onto strict, source-cited references.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 🚀 Usage

### 1. Ingestion Pipeline
Place your target PDF files into `data/raw/`, then trigger the automated ingestion:
```bash
python scripts/ingest_all.py
```
*(Tip: To append new files to an existing index without wiping it clear, use `python scripts/index_pinecone.py --append`)*

### 2. Asking Questions (API)
Send a simple POST request to the application endpoints pointing to your question to receive an analytically reasoned answer directly corresponding with source locations.
```bash
curl -X POST http://localhost:8010/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Que dice el informe sobre inflacion?", "top_k": 5}'
```

The integrated Web UI (available dynamically on `/`) provisions an intuitive dashboard designed to seamlessly upload files, trigger pipeline runs, and execute index queries highlighting visual previews linked with document citations.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 🗺️ Roadmap

- [x] Initial FastAPI Backend implementation
- [x] Hybrid Dense/Lexical Retrieval Engine
- [x] Web UI & Streamlit Integrations
- [ ] Add support for Multi-modal inputs (Images/Tables)
- [ ] Incorporate Agentic chunking methods
- [ ] Cloud Deployment guides (AWS/GCP/Vercel)


## 📫 Contact

Amin El Alam Tizniti -  - amin.elalamtizniti1@gmail.com

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[Python-shield]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/
[FastAPI-shield]: https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white
[FastAPI-url]: https://fastapi.tiangolo.com/
[LangChain-shield]: https://img.shields.io/badge/LangChain-FFFFFF?style=for-the-badge&logo=langchain&logoColor=black
[LangChain-url]: https://langchain.com/
[Pinecone-shield]: https://img.shields.io/badge/Pinecone-000000?style=for-the-badge&logo=pinecone&logoColor=white
[Pinecone-url]: https://www.pinecone.io/
[Streamlit-shield]: https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white
[Streamlit-url]: https://streamlit.io/
[Docker-shield]: https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white
[Docker-url]: https://www.docker.com/
