import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PageRecord:
    source: str
    page: int
    text: str


def parse_page(linea: str, ruta_jsonl: Path, numero_linea: int) -> PageRecord:
    try:
        carga = json.loads(linea)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {ruta_jsonl}:{numero_linea}") from exc

    fuente = str(carga.get("source", "unknown"))
    try:
        pagina = int(carga.get("page", -1))
    except (TypeError, ValueError):
        pagina = -1
    texto = str(carga.get("text", ""))
    return PageRecord(source=fuente, page=pagina, text=texto)


def load_pages(ruta_jsonl: Path) -> list[PageRecord]:
    paginas: list[PageRecord] = []
    with ruta_jsonl.open("r", encoding="utf-8") as archivo:
        for numero_linea, linea in enumerate(archivo, start=1):
            linea = linea.strip()
            if not linea:
                continue
            paginas.append(parse_page(linea=linea, ruta_jsonl=ruta_jsonl, numero_linea=numero_linea))
    return paginas


def make_chunk_id(source: str, page: int, position: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    stem = Path(source).stem
    return f"{stem}-p{page}-c{position}-{digest}"


def build_chunks(paginas: list[PageRecord], separador) -> list[dict]:
    fragmentos: list[dict] = []
    contador_fragmentos: dict[tuple[str, int], int] = {}

    for pagina in paginas:
        texto = pagina.text.strip()
        if not texto:
            continue

        documentos = separador.create_documents(
            texts=[texto],
            metadatas=[{"source": pagina.source, "page": pagina.page}],
        )
        clave_pagina = (pagina.source, pagina.page)
        contador_fragmentos.setdefault(clave_pagina, 0)

        for documento in documentos:
            contador_fragmentos[clave_pagina] += 1
            numero_fragmento = contador_fragmentos[clave_pagina]
            texto_fragmento = documento.page_content.strip()
            if not texto_fragmento:
                continue

            fragmentos.append(
                {
                    "id": make_chunk_id(pagina.source, pagina.page, numero_fragmento, texto_fragmento),
                    "text": texto_fragmento,
                    "metadata": {
                        "source": pagina.source,
                        "page": pagina.page,
                        "chunk": numero_fragmento,
                    },
                }
            )

    return fragmentos


def save_jsonl(registros: list[dict], archivo_salida: Path) -> None:
    archivo_salida.parent.mkdir(parents=True, exist_ok=True)
    with archivo_salida.open("w", encoding="utf-8") as archivo:
        for registro in registros:
            archivo.write(json.dumps(registro, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split processed pages into chunks.")
    parser.add_argument("--input_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output_file", type=Path, default=Path("data/chunks/chunks.jsonl"))
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--chunk_overlap", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {args.input_dir.resolve()}")
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if args.chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if args.chunk_overlap >= args.chunk_size:
        raise ValueError("chunk_overlap must be lower than chunk_size")

    separador = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    todos_fragmentos: list[dict] = []
    archivos_jsonl = sorted(args.input_dir.glob("*.jsonl"))
    if not archivos_jsonl:
        print(f"No processed .jsonl files found in {args.input_dir.resolve()}")
        return

    for archivo_jsonl in archivos_jsonl:
        paginas = load_pages(archivo_jsonl)
        fragmentos = build_chunks(paginas, separador)
        todos_fragmentos.extend(fragmentos)
        print(f"Chunked {archivo_jsonl.name}: {len(paginas)} pages -> {len(fragmentos)} chunks")

    save_jsonl(todos_fragmentos, args.output_file)
    print(f"Saved {len(todos_fragmentos)} chunks to {args.output_file.resolve()}")


if __name__ == "__main__":
    main()
