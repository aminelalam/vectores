import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, UploadFile

from api.config import Settings

if TYPE_CHECKING:
    from api.rag import RagEngine, RetrievedChunk


@dataclass
class AppService:
    ajustes: Settings
    motor_rag: Any = None
    error_motor: str | None = None
    bloqueo_ingesta: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.ajustes.data_raw_dir.mkdir(parents=True, exist_ok=True)
        self.ajustes.data_processed_dir.mkdir(parents=True, exist_ok=True)
        self.ajustes.data_chunks_file.parent.mkdir(parents=True, exist_ok=True)
        self.ajustes.static_dir.mkdir(parents=True, exist_ok=True)

    def inicializar_motor(self) -> None:
        try:
            from api.rag import RagEngine

            self.motor_rag = RagEngine(self.ajustes)
            self.error_motor = None
        except Exception as excepcion:
            self.motor_rag = None
            self.error_motor = str(excepcion)

    def obtener_estado_salud(self) -> dict:
        if self.motor_rag is None:
            return {"status": "degraded", "details": self.error_motor, "index": None}
        return {"status": "ok", "details": None, "index": self.ajustes.pinecone_index}

    def recargar_motor(self) -> dict:
        self.inicializar_motor()
        return self.obtener_estado_salud()

    def listar_documentos(self) -> list[str]:
        return sorted(ruta_pdf.name for ruta_pdf in self.ajustes.data_raw_dir.glob("*.pdf"))

    @staticmethod
    def _validar_nombre_archivo(nombre_archivo: str) -> str:
        nombre_seguro = Path(nombre_archivo).name.strip()
        if not nombre_seguro:
            raise HTTPException(status_code=400, detail="Invalid filename.")
        if not nombre_seguro.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400, detail=f"Only PDF files are allowed: {nombre_seguro}"
            )
        return nombre_seguro

    async def guardar_subidas(self, archivos: list[UploadFile]) -> list[str]:
        if not archivos:
            raise HTTPException(status_code=400, detail="No files received.")

        maximo_bytes = self.ajustes.upload_max_size_mb * 1024 * 1024
        archivos_guardados: list[str] = []

        for archivo in archivos:
            nombre_archivo = self._validar_nombre_archivo(archivo.filename or "")
            destino = self.ajustes.data_raw_dir / nombre_archivo
            bytes_acumulados = 0

            try:
                with destino.open("wb") as archivo_salida:
                    while True:
                        fragmento = await archivo.read(1024 * 1024)
                        if not fragmento:
                            break
                        bytes_acumulados += len(fragmento)
                        if bytes_acumulados > maximo_bytes:
                            destino.unlink(missing_ok=True)
                            raise HTTPException(
                                status_code=413,
                                detail=(
                                    f"File {nombre_archivo} exceeds max size "
                                    f"({self.ajustes.upload_max_size_mb} MB)."
                                ),
                            )
                        archivo_salida.write(fragmento)
            finally:
                await archivo.close()

            if bytes_acumulados == 0:
                destino.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=400, detail=f"File is empty: {nombre_archivo}"
                )

            archivos_guardados.append(nombre_archivo)

        return archivos_guardados

    def _ejecutar_paso(self, comando: list[str], nombre_paso: str) -> str:
        resultado = subprocess.run(
            comando,
            cwd=self.ajustes.project_root,
            text=True,
            capture_output=True,
            check=False,
        )

        salida_estandar = (resultado.stdout or "").strip()
        salida_error = (resultado.stderr or "").strip()
        if resultado.returncode != 0:
            detalle = f"{nombre_paso} failed with code {resultado.returncode}."
            if salida_error:
                detalle = f"{detalle} {salida_error}"
            raise HTTPException(status_code=500, detail=detalle)

        salida_formateada = [f"[{nombre_paso}] OK"]
        if salida_estandar:
            salida_formateada.append(salida_estandar)
        return "\n".join(salida_formateada)

    def ingestar(self) -> dict:
        if self.bloqueo_ingesta.locked():
            raise HTTPException(status_code=409, detail="Ingestion already running.")

        archivos_pdf = self.listar_documentos()
        if not archivos_pdf:
            raise HTTPException(status_code=400, detail="No PDFs found in data/raw.")

        with self.bloqueo_ingesta:
            ejecutable_python = sys.executable
            registros = [
                self._ejecutar_paso(
                    [
                        ejecutable_python,
                        "scripts/extract_text.py",
                        "--input_dir",
                        str(self.ajustes.data_raw_dir),
                        "--output_dir",
                        str(self.ajustes.data_processed_dir),
                    ],
                    "extract_text",
                ),
                self._ejecutar_paso(
                    [
                        ejecutable_python,
                        "scripts/chunk_text.py",
                        "--input_dir",
                        str(self.ajustes.data_processed_dir),
                        "--output_file",
                        str(self.ajustes.data_chunks_file),
                        "--chunk_size",
                        str(self.ajustes.ingest_chunk_size),
                        "--chunk_overlap",
                        str(self.ajustes.ingest_chunk_overlap),
                    ],
                    "chunk_text",
                ),
                self._ejecutar_paso(
                    [
                        ejecutable_python,
                        "scripts/index_pinecone.py",
                        "--chunks_file",
                        str(self.ajustes.data_chunks_file),
                        "--batch_size",
                        str(self.ajustes.ingest_batch_size),
                    ],
                    "index_pinecone",
                ),
            ]

            self.inicializar_motor()
            return {
                "message": "Ingestion completed successfully.",
                "indexed_files": archivos_pdf,
                "engine_ready": self.motor_rag is not None,
                "logs": registros,
            }

    def consultar(self, pregunta: str, cantidad_fragmentos: int) -> tuple[str, list["RetrievedChunk"]]:
        if self.motor_rag is None:
            raise HTTPException(
                status_code=500, detail=f"RAG engine unavailable: {self.error_motor}"
            )
        return self.motor_rag.ask(pregunta=pregunta, cantidad_fragmentos=cantidad_fragmentos)
