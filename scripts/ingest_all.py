import argparse
import subprocess
import sys
from pathlib import Path


def run_step(comando: list[str], nombre_paso: str, directorio_trabajo: Path) -> None:
    print(f"\n[{nombre_paso}] Running: {' '.join(comando)}")
    resultado = subprocess.run(comando, cwd=directorio_trabajo, text=True, capture_output=True, check=False)
    if resultado.returncode != 0:
        salida_error = (resultado.stderr or "").strip()
        salida_estandar = (resultado.stdout or "").strip()
        detalles = salida_error or salida_estandar or "No output."
        raise RuntimeError(f"{nombre_paso} failed with exit code {resultado.returncode}. {detalles}")
    if resultado.stdout:
        print(resultado.stdout.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full ingestion pipeline.")
    parser.add_argument("--input_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--chunks_file", type=Path, default=Path("data/chunks/chunks.jsonl"))
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--chunk_overlap", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if args.chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if args.chunk_overlap >= args.chunk_size:
        raise ValueError("chunk_overlap must be lower than chunk_size")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    directorio_proyecto = Path(__file__).resolve().parents[1]
    ejecutable_python = sys.executable

    run_step(
        [
            ejecutable_python,
            "scripts/extract_text.py",
            "--input_dir",
            str(args.input_dir),
            "--output_dir",
            str(args.processed_dir),
        ],
        "extract_text",
        directorio_trabajo=directorio_proyecto,
    )
    run_step(
        [
            ejecutable_python,
            "scripts/chunk_text.py",
            "--input_dir",
            str(args.processed_dir),
            "--output_file",
            str(args.chunks_file),
            "--chunk_size",
            str(args.chunk_size),
            "--chunk_overlap",
            str(args.chunk_overlap),
        ],
        "chunk_text",
        directorio_trabajo=directorio_proyecto,
    )
    run_step(
        [
            ejecutable_python,
            "scripts/index_pinecone.py",
            "--chunks_file",
            str(args.chunks_file),
            "--batch_size",
            str(args.batch_size),
        ],
        "index_pinecone",
        directorio_trabajo=directorio_proyecto,
    )

    print("\nIngestion pipeline completed successfully.")


if __name__ == "__main__":
    main()
