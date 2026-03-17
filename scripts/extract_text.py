import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class PageRecord:
    source: str
    page: int
    text: str


def repair_mojibake(texto: str) -> str:
    if "Ã" not in texto and "Â" not in texto:
        return texto
    try:
        texto_arreglado = texto.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return texto

    caracteres_erroneos = texto.count("Ã") + texto.count("Â")
    caracteres_reparados = texto_arreglado.count("Ã") + texto_arreglado.count("Â")
    if caracteres_reparados < caracteres_erroneos:
        return texto_arreglado
    return texto


def clean_text(texto: str) -> str:
    texto = repair_mojibake(texto)
    texto = texto.replace("-\n", "")
    texto = re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", " ", texto)
    texto = re.sub(r"[ \t]+\n", "\n", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    texto = re.sub(r"[ \t]{2,}", " ", texto)
    return texto.strip()


def extract_pdf(ruta_pdf: Path) -> list[PageRecord]:
    lector_pdf = PdfReader(str(ruta_pdf))
    paginas: list[PageRecord] = []

    for numero_pagina, pagina in enumerate(lector_pdf.pages, start=1):
        texto_bruto = pagina.extract_text() or ""
        paginas.append(
            PageRecord(
                source=ruta_pdf.name,
                page=numero_pagina,
                text=clean_text(texto_bruto),
            )
        )

    return paginas


def save_processed(paginas: list[PageRecord], directorio_salida: Path, nombre_fuente: str) -> tuple[Path, Path]:
    nombre_base = Path(nombre_fuente).stem
    ruta_txt = directorio_salida / f"{nombre_base}.txt"
    ruta_jsonl = directorio_salida / f"{nombre_base}.jsonl"

    with ruta_txt.open("w", encoding="utf-8") as archivo_txt:
        for pagina in paginas:
            archivo_txt.write(f"=== Page {pagina.page} ===\n")
            archivo_txt.write(pagina.text)
            archivo_txt.write("\n\n")

    with ruta_jsonl.open("w", encoding="utf-8") as archivo_jsonl:
        for pagina in paginas:
            archivo_jsonl.write(json.dumps(asdict(pagina), ensure_ascii=False) + "\n")

    return ruta_txt, ruta_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and clean text from PDFs.")
    parser.add_argument("--input_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--pattern", type=str, default="*.pdf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {args.input_dir.resolve()}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archivos_pdf = sorted(args.input_dir.glob(args.pattern))

    if not archivos_pdf:
        print(f"No PDF files found in {args.input_dir.resolve()}")
        return

    for archivo_pdf in archivos_pdf:
        paginas = extract_pdf(archivo_pdf)
        ruta_txt, ruta_jsonl = save_processed(paginas, args.output_dir, archivo_pdf.name)
        paginas_con_texto = sum(1 for pagina in paginas if pagina.text)
        print(
            f"Processed {archivo_pdf.name}: {len(paginas)} pages, "
            f"{paginas_con_texto} with text -> {ruta_txt.name}, {ruta_jsonl.name}"
        )


if __name__ == "__main__":
    main()
