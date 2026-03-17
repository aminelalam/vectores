import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

VALID_METRICS = {"cosine", "dotproduct", "euclidean"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate embeddings and upload to Pinecone.")
    parser.add_argument("--chunks_file", type=Path, default=Path("data/chunks/chunks.jsonl"))
    parser.add_argument("--index_name", type=str, default=os.getenv("PINECONE_INDEX", "rag-semantic-search"))
    parser.add_argument("--namespace", type=str, default=os.getenv("PINECONE_NAMESPACE", "default"))
    parser.add_argument("--metric", type=str, default="cosine")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embedding_model", type=str, default=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    parser.add_argument("--cloud", type=str, default=os.getenv("PINECONE_CLOUD", "aws"))
    parser.add_argument("--region", type=str, default=os.getenv("PINECONE_REGION", "us-east-1"))
    parser.add_argument(
        "--append",
        action="store_true",
        help="Do not clear existing vectors in the namespace before indexing.",
    )
    return parser.parse_args()


def load_chunks(archivo_fragmentos: Path) -> list[dict]:
    fragmentos: list[dict] = []
    with archivo_fragmentos.open("r", encoding="utf-8") as archivo:
        for numero_linea, linea in enumerate(archivo, start=1):
            linea = linea.strip()
            if not linea:
                continue
            try:
                fragmentos.append(json.loads(linea))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {archivo_fragmentos}:{numero_linea}") from exc
    return fragmentos


def batch_iter(elementos: list[dict], tamano_lote: int) -> Iterable[list[dict]]:
    for indice in range(0, len(elementos), tamano_lote):
        yield elementos[indice : indice + tamano_lote]


def _pick(valor: Any, clave: str, predeterminado: Any) -> Any:
    if isinstance(valor, dict):
        return valor.get(clave, predeterminado)
    return getattr(valor, clave, predeterminado)


def get_index_names(cliente: Pinecone) -> set[str]:
    respuesta = cliente.list_indexes()
    nombres: set[str] = set()

    if hasattr(respuesta, "names"):
        return set(respuesta.names())

    for elemento in respuesta:
        nombres.add(str(_pick(elemento, "name", "")).strip())
    nombres.discard("")
    return nombres


def wait_for_index_ready(cliente: Pinecone, nombre_indice: str, timeout_seconds: int = 120) -> None:
    inicio = time.time()
    while True:
        descripcion = cliente.describe_index(nombre_indice)
        estado = _pick(descripcion, "status", {})
        listo = _pick(estado, "ready", False)
        if listo:
            return

        if time.time() - inicio > timeout_seconds:
            raise TimeoutError(f"Index {nombre_indice} was not ready after {timeout_seconds}s.")
        time.sleep(2)


def _describe_index_dimension(cliente: Pinecone, nombre_indice: str) -> int | None:
    descripcion = cliente.describe_index(nombre_indice)
    dimension_sin_procesar = _pick(descripcion, "dimension", None)
    if dimension_sin_procesar is None:
        return None
    try:
        return int(dimension_sin_procesar)
    except (TypeError, ValueError):
        return None


def ensure_index(
    client: Pinecone,
    index_name: str,
    dimension: int,
    metric: str,
    cloud: str,
    region: str,
) -> None:
    existing_indexes = get_index_names(client)
    if index_name not in existing_indexes:
        client.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        wait_for_index_ready(client, index_name)
        return

    existing_dimension = _describe_index_dimension(client, index_name)
    if existing_dimension is not None and existing_dimension != dimension:
        raise ValueError(
            f"Pinecone index dimension mismatch for {index_name}: "
            f"index={existing_dimension}, model={dimension}."
        )


def main() -> None:
    load_dotenv()
    args = parse_args()

    if args.metric not in VALID_METRICS:
        raise ValueError(f"Invalid metric {args.metric}. Valid values: {sorted(VALID_METRICS)}")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
    if not pinecone_api_key:
        raise ValueError("Missing PINECONE_API_KEY in environment.")

    if not args.chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {args.chunks_file.resolve()}")

    fragmentos = load_chunks(args.chunks_file)
    if not fragmentos:
        raise ValueError(f"No chunks found in {args.chunks_file.resolve()}")

    modelo = SentenceTransformer(args.embedding_model)
    dimension = modelo.get_sentence_embedding_dimension()
    cliente_pinecone = Pinecone(api_key=pinecone_api_key)

    ensure_index(
        client=cliente_pinecone,
        index_name=args.index_name,
        dimension=dimension,
        metric=args.metric,
        cloud=args.cloud,
        region=args.region,
    )
    indice = cliente_pinecone.Index(args.index_name)

    if not args.append:
        indice.delete(delete_all=True, namespace=args.namespace)
        print(f"Cleared namespace '{args.namespace}' in index '{args.index_name}'.")

    total_insertados = 0
    total_lotes = (len(fragmentos) + args.batch_size - 1) // args.batch_size
    for lote in tqdm(batch_iter(fragmentos, args.batch_size), total=total_lotes, desc="Upserting batches"):
        textos = [str(elemento.get("text", "")) for elemento in lote]
        embeddings = modelo.encode(textos, normalize_embeddings=True).tolist()

        vectores: list[dict] = []
        for elemento, embedding in zip(lote, embeddings):
            identificador = str(elemento.get("id", "")).strip()
            if not identificador:
                continue

            metadatos = dict(elemento.get("metadata", {}))
            metadatos["text"] = str(elemento.get("text", ""))
            vectores.append({"id": identificador, "values": embedding, "metadata": metadatos})

        if not vectores:
            continue

        indice.upsert(vectors=vectores, namespace=args.namespace)
        total_insertados += len(vectores)

    print(f"Indexed {total_insertados} chunks into {args.index_name} (namespace={args.namespace}).")


if __name__ == "__main__":
    main()
