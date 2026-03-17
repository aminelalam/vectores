import json
import logging
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from api.config import Settings

logger = logging.getLogger(__name__)

STOPWORDS = {
    "de",
    "del",
    "la",
    "el",
    "los",
    "las",
    "que",
    "con",
    "para",
    "por",
    "una",
    "uno",
    "sobre",
    "como",
    "cual",
    "cuales",
    "hablame",
    "dime",
    "quiero",
    "saber",
    "va",
    "cuanto",
    "cuantos",
    "cuanta",
    "cuantas",
    "tiene",
    "tienen",
    "hay",
}

TERM_EXPANSIONS = {
    "baloncesto": ("basquet", "basket"),
    "basquet": ("baloncesto", "basket"),
    "basket": ("basquet",),
    "equipo": ("equip",),
    "equipos": ("equips", "equip"),
    "partido": ("partit",),
    "partidos": ("partits", "partit"),
    "regla": ("regla", "regles"),
    "reglas": ("regles", "reglament"),
    "reglamento": ("reglament", "regles"),
    "tiempo": ("temps", "minut"),
    "tiempos": ("temps", "minuts"),
    "muerto": ("mort",),
    "muertos": ("morts", "mort"),
    "prorroga": ("prorroga",),
    "prorrogas": ("prorrogues", "prorroga"),
    "entrenador": ("entrenador", "coach"),
    "jugador": ("jugador", "jugadors"),
    "jugadores": ("jugadors", "jugador"),
    "falta": ("falta", "faltes"),
    "faltas": ("faltes", "falta"),
}


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    source: str
    page: int
    score: float
    text: str

    def source_label(self) -> str:
        return f"{self.source} page {self.page}"


@dataclass(frozen=True)
class LocalChunk:
    id: str
    source: str
    page: int
    text: str
    text_norm: str
    has_digits: bool
    source_terms: tuple[str, ...]


class RagEngine:
    _BM25_K1 = 1.4
    _BM25_B = 0.75
    _RRF_K = 60
    _DENSE_WEIGHT = 1.0
    _LEXICAL_WEIGHT = 0.95
    _FUZZY_MIN_SIMILARITY = 0.45
    _FUZZY_MAX_EXPANSIONS = 3
    _COVERAGE_BOOST = 0.40
    _NUMERIC_BOOST = 0.07

    def __init__(self, ajustes: Settings) -> None:
        if not ajustes.pinecone_api_key:
            raise ValueError("Missing PINECONE_API_KEY.")

        self.ajustes = ajustes
        self.embedder = SentenceTransformer(ajustes.embedding_model)
        self.embedding_dimension = self.embedder.get_sentence_embedding_dimension()
        self.pinecone = Pinecone(api_key=ajustes.pinecone_api_key)
        self.index = self.pinecone.Index(ajustes.pinecone_index)

        self._openai_client: OpenAI | None = None
        if ajustes.openai_api_key:
            self._openai_client = OpenAI(api_key=ajustes.openai_api_key)

        self._http = requests.Session()
        self._lex_chunks: list[LocalChunk] = []
        self._lex_doc_len: list[int] = []
        self._lex_postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._lex_doc_freq: dict[str, int] = {}
        self._lex_avg_doc_len = 1.0
        self._lex_term_ngrams: dict[str, set[str]] = {}
        self._ngram_to_terms: dict[str, set[str]] = defaultdict(set)
        self._chunk_text_norm_by_id: dict[str, str] = {}

        self._build_local_lexical_index()

    @staticmethod
    def _pick(value: dict[str, Any] | Any, key: str, default: Any) -> Any:
        if isinstance(value, dict):
            return value.get(key, default)
        return getattr(value, key, default)

    @staticmethod
    def _normalize(texto: str) -> str:
        texto = texto.lower().strip()
        texto = unicodedata.normalize("NFD", texto)
        texto = "".join(caracter for caracter in texto if unicodedata.category(caracter) != "Mn")
        texto = re.sub(r"[^a-z0-9._\s-]+", " ", texto)
        return re.sub(r"\s+", " ", texto).strip()

    def _keywords(self, consulta: str) -> list[str]:
        tokens = self._normalize(consulta).split()
        return [token for token in tokens if len(token) > 2 and token not in STOPWORDS]

    @staticmethod
    def _char_ngrams(palabra: str, n: int = 3) -> set[str]:
        if not palabra:
            return set()
        if len(palabra) < n:
            return {palabra}
        return {palabra[i : i + n] for i in range(len(palabra) - n + 1)}

    @staticmethod
    def _jaccard_similarity(conjunto_izquierdo: set[str], conjunto_derecho: set[str]) -> float:
        if not conjunto_izquierdo or not conjunto_derecho:
            return 0.0
        union_conjuntos = conjunto_izquierdo.union(conjunto_derecho)
        if not union_conjuntos:
            return 0.0
        return len(conjunto_izquierdo.intersection(conjunto_derecho)) / len(union_conjuntos)

    def _is_quantity_query(self, consulta: str) -> bool:
        consulta_normalizada = self._normalize(consulta)
        return any(
            termino in consulta_normalizada
            for termino in ("cuanto", "cuantos", "cuanta", "cuantas", "numero", "cantidad")
        )

    def _expand_query_terms(self, terminos: list[str]) -> list[str]:
        terminos_expandidos: list[str] = []
        for termino in terminos:
            terminos_expandidos.append(termino)
            if termino.endswith("es") and len(termino) > 4:
                terminos_expandidos.append(termino[:-2])
            elif termino.endswith("s") and len(termino) > 3:
                terminos_expandidos.append(termino[:-1])

            terminos_relacionados = TERM_EXPANSIONS.get(termino, ())
            terminos_expandidos.extend(terminos_relacionados)

        vistos: set[str] = set()
        terminos_unicos: list[str] = []
        for termino in terminos_expandidos:
            if termino in vistos:
                continue
            vistos.add(termino)
            terminos_unicos.append(termino)
        return terminos_unicos

    def _find_similar_terms(self, termino: str) -> list[tuple[str, float]]:
        if termino in self._lex_postings:
            return []

        ngramas_termino = self._char_ngrams(termino)
        if not ngramas_termino:
            return []

        terminos_candidatos: set[str] = set()
        for ngrama in ngramas_termino:
            terminos_candidatos.update(self._ngram_to_terms.get(ngrama, set()))

        if not terminos_candidatos:
            return []

        terminos_puntuados: list[tuple[str, float]] = []
        for candidato in terminos_candidatos:
            ngramas_candidato = self._lex_term_ngrams.get(candidato)
            if not ngramas_candidato:
                continue
            similitud = self._jaccard_similarity(ngramas_termino, ngramas_candidato)
            if similitud >= self._FUZZY_MIN_SIMILARITY:
                terminos_puntuados.append((candidato, similitud))

        terminos_puntuados.sort(key=lambda par: par[1], reverse=True)
        return terminos_puntuados[: self._FUZZY_MAX_EXPANSIONS]

    def _weighted_query_terms(self, consulta: str) -> dict[str, float]:
        terminos_base = self._keywords(consulta)
        if not terminos_base:
            return {}

        terminos_ponderados: dict[str, float] = {}

        def agregar(termino: str, peso: float) -> None:
            peso_actual = terminos_ponderados.get(termino, 0.0)
            if peso > peso_actual:
                terminos_ponderados[termino] = peso

        for termino in terminos_base:
            agregar(termino, 1.0)

            if termino.endswith("es") and len(termino) > 4:
                agregar(termino[:-2], 0.9)
            elif termino.endswith("s") and len(termino) > 3:
                agregar(termino[:-1], 0.9)

            for termino_relacionado in TERM_EXPANSIONS.get(termino, ()):
                agregar(termino_relacionado, 0.92)

        for termino in list(terminos_ponderados):
            for termino_difuso, similitud in self._find_similar_terms(termino):
                agregar(termino_difuso, 0.72 * similitud)

        return terminos_ponderados

    @staticmethod
    def _chunk_key(fragmento: RetrievedChunk) -> str:
        if fragmento.id:
            return fragmento.id
        digest = hash((fragmento.source, fragmento.page, fragmento.text[:120]))
        return f"{fragmento.source}:{fragmento.page}:{digest}"

    @staticmethod
    def _safe_page(valor: Any) -> int:
        try:
            return int(valor)
        except (TypeError, ValueError):
            return -1

    @staticmethod
    def _llm_error_hint(excepcion: Exception) -> str:
        mensaje = str(excepcion).lower()
        if "api key" in mensaje or "unauthorized" in mensaje or "401" in mensaje:
            return "Revisa credenciales del proveedor LLM."
        if "connection" in mensaje or "timed out" in mensaje or "timeout" in mensaje:
            return "No hay conectividad con el proveedor LLM."
        return "El proveedor LLM no respondio correctamente."

    def _build_local_lexical_index(self) -> None:
        archivo_fragmentos = self.ajustes.data_chunks_file
        if not archivo_fragmentos.exists():
            logger.info("Chunks file not found for lexical index: %s", archivo_fragmentos)
            return

        longitud_total = 0

        with archivo_fragmentos.open("r", encoding="utf-8") as archivo:
            for numero_linea, linea in enumerate(archivo, start=1):
                fila = linea.strip()
                if not fila:
                    continue
                try:
                    carga = json.loads(fila)
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON in %s at line %d", archivo_fragmentos, numero_linea)
                    continue

                identificador = str(carga.get("id", "")).strip()
                texto_fragmento = str(carga.get("text", "")).strip()
                metadatos = carga.get("metadata", {}) or {}
                if not isinstance(metadatos, dict):
                    metadatos = {}

                nombre_fuente = str(metadatos.get("source", "unknown")).strip() or "unknown"
                pagina = self._safe_page(metadatos.get("page"))
                terminos_fuente = self._keywords(Path(nombre_fuente).stem.replace(".", " "))
                texto_normalizado = self._normalize(texto_fragmento)
                terminos_texto = [
                    token for token in texto_normalizado.split() if len(token) > 2 and token not in STOPWORDS
                ]
                terminos_totales = terminos_texto + terminos_fuente

                if not identificador or not texto_fragmento or not terminos_totales:
                    continue

                indice_documento = len(self._lex_chunks)
                self._lex_chunks.append(
                    LocalChunk(
                        id=identificador,
                        source=nombre_fuente,
                        page=pagina,
                        text=texto_fragmento,
                        text_norm=texto_normalizado,
                        has_digits=any(caracter.isdigit() for caracter in texto_fragmento),
                        source_terms=tuple(sorted(set(terminos_fuente))),
                    )
                )
                self._chunk_text_norm_by_id[identificador] = texto_normalizado

                conteo_terminos = Counter(terminos_totales)
                longitud_documento = sum(conteo_terminos.values())
                longitud_total += longitud_documento
                self._lex_doc_len.append(longitud_documento)

                for termino, frecuencia in conteo_terminos.items():
                    self._lex_postings[termino].append((indice_documento, frecuencia))
                for termino in conteo_terminos:
                    self._lex_doc_freq[termino] = self._lex_doc_freq.get(termino, 0) + 1

        if self._lex_chunks:
            for termino in self._lex_postings:
                ngramas = self._char_ngrams(termino)
                self._lex_term_ngrams[termino] = ngramas
                for ngrama in ngramas:
                    self._ngram_to_terms[ngrama].add(termino)

            self._lex_avg_doc_len = max(1.0, longitud_total / len(self._lex_chunks))
            logger.info(
                "Loaded lexical index with %d chunks and %d terms.",
                len(self._lex_chunks),
                len(self._lex_postings),
            )

    def _build_dense_queries(self, consulta: str) -> list[str]:
        consultas_densas = [consulta]
        terminos_ponderados = self._weighted_query_terms(consulta)
        if terminos_ponderados:
            terminos_relevantes = sorted(
                terminos_ponderados.items(), key=lambda par: par[1], reverse=True
            )[:10]
            texto_expandido = " ".join(termino for termino, _ in terminos_relevantes)
            if texto_expandido and texto_expandido not in consultas_densas:
                consultas_densas.append(texto_expandido)
        return consultas_densas

    def _query_dense_once(self, consulta: str, cantidad_resultados: int) -> list[RetrievedChunk]:
        embedding_consulta = self.embedder.encode(consulta, normalize_embeddings=True).tolist()
        respuesta = self.index.query(
            vector=embedding_consulta,
            top_k=cantidad_resultados,
            include_metadata=True,
            namespace=self.ajustes.pinecone_namespace,
        )

        coincidencias = self._pick(respuesta, "matches", [])
        fragmentos_recuperados: list[RetrievedChunk] = []
        identificadores_vistos: set[str] = set()

        for coincidencia in coincidencias:
            metadatos = self._pick(coincidencia, "metadata", {}) or {}
            if not isinstance(metadatos, dict):
                metadatos = {}

            texto_fragmento = str(metadatos.get("text", "")).strip()
            if not texto_fragmento:
                continue

            fragmento = RetrievedChunk(
                id=str(self._pick(coincidencia, "id", "")).strip(),
                source=str(metadatos.get("source", "unknown")),
                page=self._safe_page(metadatos.get("page")),
                score=float(self._pick(coincidencia, "score", 0.0) or 0.0),
                text=texto_fragmento,
            )
            llave_fragmento = self._chunk_key(fragmento)
            if llave_fragmento in identificadores_vistos:
                continue
            identificadores_vistos.add(llave_fragmento)
            fragmentos_recuperados.append(fragmento)
        return fragmentos_recuperados

    def _retrieve_dense(self, consulta: str, cantidad_resultados: int) -> list[RetrievedChunk]:
        consultas_densas = self._build_dense_queries(consulta)
        if not consultas_densas:
            return []

        puntuaciones_fusionadas: dict[str, float] = defaultdict(float)
        fragmentos_representativos: dict[str, RetrievedChunk] = {}

        for consulta_densa in consultas_densas:
            fragmentos_ordenados = self._query_dense_once(
                consulta=consulta_densa, cantidad_resultados=cantidad_resultados
            )
            for posicion, fragmento in enumerate(fragmentos_ordenados, start=1):
                llave_fragmento = self._chunk_key(fragmento)
                puntuaciones_fusionadas[llave_fragmento] += 1.0 / (40 + posicion)
                if (
                    llave_fragmento not in fragmentos_representativos
                    or fragmento.score > fragmentos_representativos[llave_fragmento].score
                ):
                    fragmentos_representativos[llave_fragmento] = fragmento

        if not puntuaciones_fusionadas:
            return []

        maximo_rrf = 1.0 / 41.0
        llaves_ordenadas = sorted(
            puntuaciones_fusionadas.keys(), key=lambda llave: puntuaciones_fusionadas[llave], reverse=True
        )
        fragmentos_densos: list[RetrievedChunk] = []
        for llave in llaves_ordenadas[:cantidad_resultados]:
            fragmento = fragmentos_representativos[llave]
            fragmentos_densos.append(
                RetrievedChunk(
                    id=fragmento.id,
                    source=fragmento.source,
                    page=fragmento.page,
                    score=puntuaciones_fusionadas[llave] / maximo_rrf,
                    text=fragmento.text,
                )
            )
        return fragmentos_densos

    def _retrieve_lexical(self, consulta: str, cantidad_resultados: int) -> list[RetrievedChunk]:
        if not self._lex_chunks:
            return []

        terminos_ponderados = self._weighted_query_terms(consulta)
        if not terminos_ponderados:
            return []

        cantidad_documentos = len(self._lex_chunks)
        puntuaciones: dict[int, float] = defaultdict(float)
        terminos_consulta = set(terminos_ponderados)

        for termino, peso_consulta in terminos_ponderados.items():
            apariciones = self._lex_postings.get(termino)
            if not apariciones:
                continue

            frecuencia_documentos = self._lex_doc_freq.get(termino, 0)
            if frecuencia_documentos <= 0:
                continue

            idf = math.log(1.0 + ((cantidad_documentos - frecuencia_documentos + 0.5) / (frecuencia_documentos + 0.5)))
            for indice_documento, frecuencia in apariciones:
                longitud_documento = self._lex_doc_len[indice_documento]
                denominador = frecuencia + self._BM25_K1 * (
                    1 - self._BM25_B + self._BM25_B * (longitud_documento / self._lex_avg_doc_len)
                )
                puntuaciones[indice_documento] += (
                    peso_consulta * idf * ((frecuencia * (self._BM25_K1 + 1)) / max(denominador, 1e-9))
                )

        if not puntuaciones:
            return []

        for indice_documento in list(puntuaciones.keys()):
            coincidencia_fuente = len(
                terminos_consulta.intersection(self._lex_chunks[indice_documento].source_terms)
            )
            if coincidencia_fuente:
                puntuaciones[indice_documento] += 0.6 * coincidencia_fuente

        ordenados = sorted(puntuaciones.items(), key=lambda par: par[1], reverse=True)[:cantidad_resultados]
        return [
            RetrievedChunk(
                id=self._lex_chunks[indice_documento].id,
                source=self._lex_chunks[indice_documento].source,
                page=self._lex_chunks[indice_documento].page,
                score=float(puntuacion),
                text=self._lex_chunks[indice_documento].text,
            )
            for indice_documento, puntuacion in ordenados
        ]

    def _fuse_rankings(
        self,
        fragmentos_densos: list[RetrievedChunk],
        fragmentos_lexicos: list[RetrievedChunk],
        cantidad_resultados: int,
    ) -> list[RetrievedChunk]:
        if not fragmentos_densos:
            return fragmentos_lexicos[:cantidad_resultados]
        if not fragmentos_lexicos:
            return fragmentos_densos[:cantidad_resultados]

        puntuaciones_combinadas: dict[str, float] = defaultdict(float)
        fragmentos_representativos: dict[str, RetrievedChunk] = {}

        for posicion, fragmento in enumerate(fragmentos_densos, start=1):
            llave_fragmento = self._chunk_key(fragmento)
            puntuaciones_combinadas[llave_fragmento] += self._DENSE_WEIGHT / (self._RRF_K + posicion)
            fragmentos_representativos.setdefault(llave_fragmento, fragmento)

        for posicion, fragmento in enumerate(fragmentos_lexicos, start=1):
            llave_fragmento = self._chunk_key(fragmento)
            puntuaciones_combinadas[llave_fragmento] += self._LEXICAL_WEIGHT / (self._RRF_K + posicion)
            if (
                llave_fragmento not in fragmentos_representativos
                or len(fragmento.text) > len(fragmentos_representativos[llave_fragmento].text)
            ):
                fragmentos_representativos[llave_fragmento] = fragmento

        maximo_rrf = (self._DENSE_WEIGHT + self._LEXICAL_WEIGHT) / (self._RRF_K + 1)
        llaves_ordenadas = sorted(
            puntuaciones_combinadas.keys(), key=lambda llave: puntuaciones_combinadas[llave], reverse=True
        )

        fragmentos_fusionados: list[RetrievedChunk] = []
        for llave in llaves_ordenadas[:cantidad_resultados]:
            fragmento = fragmentos_representativos[llave]
            puntaje = puntuaciones_combinadas[llave] / maximo_rrf if maximo_rrf > 0 else puntuaciones_combinadas[llave]
            fragmentos_fusionados.append(
                RetrievedChunk(
                    id=fragmento.id,
                    source=fragmento.source,
                    page=fragmento.page,
                    score=float(puntaje),
                    text=fragmento.text,
                )
            )
        return fragmentos_fusionados

    def _chunk_text_norm(self, fragmento: RetrievedChunk) -> str:
        if fragmento.id and fragmento.id in self._chunk_text_norm_by_id:
            return self._chunk_text_norm_by_id[fragmento.id]
        texto_normalizado = self._normalize(fragmento.text)
        if fragmento.id:
            self._chunk_text_norm_by_id[fragmento.id] = texto_normalizado
        return texto_normalizado

    def _query_coverage(self, fragmento: RetrievedChunk, terminos_ponderados: dict[str, float]) -> float:
        if not terminos_ponderados:
            return 0.0

        fuente_normalizada = self._normalize(fragmento.source)
        texto_normalizado = self._chunk_text_norm(fragmento)
        peso_cubierto = 0.0
        peso_total = sum(terminos_ponderados.values())
        if peso_total <= 0:
            return 0.0

        for termino, peso in terminos_ponderados.items():
            if termino in texto_normalizado or termino in fuente_normalizada:
                peso_cubierto += peso
        return peso_cubierto / peso_total

    def _rerank_by_query_coverage(
        self, fragmentos: list[RetrievedChunk], consulta: str, cantidad_resultados: int
    ) -> list[RetrievedChunk]:
        terminos_ponderados = self._weighted_query_terms(consulta)
        consulta_cuantitativa = self._is_quantity_query(consulta)

        fragmentos_recalculados: list[tuple[float, float, RetrievedChunk]] = []
        for fragmento in fragmentos:
            cobertura = self._query_coverage(fragmento, terminos_ponderados)
            puntaje_ajustado = fragmento.score + (cobertura * self._COVERAGE_BOOST)
            if consulta_cuantitativa and any(caracter.isdigit() for caracter in fragmento.text):
                puntaje_ajustado += self._NUMERIC_BOOST
            fragmentos_recalculados.append((puntaje_ajustado, cobertura, fragmento))

        fragmentos_recalculados.sort(key=lambda fila: (fila[0], fila[1]), reverse=True)
        return [
            RetrievedChunk(
                id=fragmento.id,
                source=fragmento.source,
                page=fragmento.page,
                score=float(puntaje),
                text=fragmento.text,
            )
            for puntaje, _, fragmento in fragmentos_recalculados[:cantidad_resultados]
        ]

    def retrieve(self, consulta: str, cantidad_resultados: int) -> list[RetrievedChunk]:
        cantidad_final = max(1, cantidad_resultados)
        cantidad_local = max(1, len(self._lex_chunks))
        lote_denso = min(max(cantidad_final * 18, 80), min(400, cantidad_local))
        lote_lexico = min(max(cantidad_final * 36, 160), min(1200, cantidad_local))

        fragmentos_densos = self._retrieve_dense(consulta=consulta, cantidad_resultados=lote_denso)
        fragmentos_lexicos = self._retrieve_lexical(consulta=consulta, cantidad_resultados=lote_lexico)
        fragmentos_fusionados = self._fuse_rankings(
            fragmentos_densos=fragmentos_densos,
            fragmentos_lexicos=fragmentos_lexicos,
            cantidad_resultados=max(cantidad_final * 2, cantidad_final),
        )
        return self._rerank_by_query_coverage(fragmentos=fragmentos_fusionados, consulta=consulta, cantidad_resultados=cantidad_final)

    def _build_context(self, fragmentos: list[RetrievedChunk]) -> str:
        caracteres_restantes = self.ajustes.rag_max_context_chars
        partes_contexto: list[str] = []
        for indice, fragmento in enumerate(fragmentos, start=1):
            if caracteres_restantes <= 0:
                break
            encabezado = f"[Source {indice}] {fragmento.source_label()}\n"
            cuerpo = fragmento.text.strip()
            permitido = max(0, caracteres_restantes - len(encabezado))
            cuerpo = cuerpo[:permitido]
            partes_contexto.append(encabezado + cuerpo)
            caracteres_restantes -= len(encabezado) + len(cuerpo) + 2
        return "\n\n".join(partes_contexto)

    @staticmethod
    def _build_prompt(pregunta: str, contexto: str) -> str:
        return (
            "You are a document analysis assistant.\n"
            "Use only the provided context to answer.\n"
            "If context is partial, answer with available facts and point out what is missing.\n"
            "Do not invent facts that are not present in context.\n"
            "Context may include Spanish or Catalan; interpret both correctly.\n"
            "Before saying you do not know, check for semantically equivalent expressions.\n"
            "Example: 'tiempo muerto' and 'temps mort' refer to the same concept.\n"
            "Cite sources in the format [Source N] when possible.\n"
            "Answer in Spanish.\n\n"
            f"Context:\n{contexto}\n\n"
            f"Question:\n{pregunta}\n"
        )

    @staticmethod
    def _answer_looks_uncertain(respuesta: str) -> bool:
        texto_minusculas = respuesta.lower()
        return any(
            patron in texto_minusculas
            for patron in (
                "no se",
                "no sé",
                "no puedo responder",
                "no dispongo",
                "informacion insuficiente",
                "información insuficiente",
                "contexto proporcionado no incluye",
            )
        )

    def _should_retry_extraction(self, consulta: str, fragmentos: list[RetrievedChunk]) -> bool:
        terminos_ponderados = self._weighted_query_terms(consulta)
        if not terminos_ponderados:
            return False

        consulta_cuantitativa = self._is_quantity_query(consulta)
        for fragmento in fragmentos[:6]:
            cobertura = self._query_coverage(fragmento, terminos_ponderados)
            if cobertura >= 0.22:
                return True
            if consulta_cuantitativa and any(caracter.isdigit() for caracter in fragmento.text):
                return True
        return False

    @staticmethod
    def _build_rescue_prompt(consulta: str, contexto: str) -> str:
        return (
            "Re-evaluate the context carefully and extract the answer directly.\n"
            "The context may be in Spanish or Catalan.\n"
            "If the user asks for a quantity, prioritize explicit numeric rules in the context.\n"
            "If an answer exists, provide it with concise explanation and sources.\n"
            "Only if no supporting evidence exists, answer 'No se'.\n\n"
            f"Context:\n{contexto}\n\n"
            f"Question:\n{consulta}\n"
        )

    def _generate_openai(self, instruccion: str) -> str:
        if self._openai_client is None:
            raise ValueError("Missing OPENAI_API_KEY for provider openai.")

        completion = self._openai_client.chat.completions.create(
            model=self.ajustes.openai_model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": "Be concise, accurate, and cite provided context."},
                {"role": "user", "content": instruccion},
            ],
            timeout=self.ajustes.llm_timeout_seconds,
        )
        return (completion.choices[0].message.content or "").strip()

    def _generate_ollama(self, instruccion: str) -> str:
        respuesta = self._http.post(
            f"{self.ajustes.ollama_base_url.rstrip('/')}/api/generate",
            json={
                "model": self.ajustes.ollama_model,
                "prompt": instruccion,
                "stream": False,
            },
            timeout=self.ajustes.llm_timeout_seconds,
        )
        respuesta.raise_for_status()
        return str(respuesta.json().get("response", "")).strip()

    def _generate_answer(self, instruccion: str) -> str:
        if self.ajustes.llm_provider == "openai":
            return self._generate_openai(instruccion)
        if self.ajustes.llm_provider == "ollama":
            return self._generate_ollama(instruccion)
        raise ValueError(f"Unsupported LLM provider: {self.ajustes.llm_provider}")

    def _fallback_answer(self, motivo: str) -> str:
        return (
            f"No pude generar respuesta con el modelo ({self.ajustes.llm_provider}). "
            f"{motivo} Te muestro abajo las fuentes recuperadas."
        )

    def ask(self, pregunta: str, cantidad_fragmentos: int | None = None) -> tuple[str, list[RetrievedChunk]]:
        cantidad_solicitada = cantidad_fragmentos or self.ajustes.rag_top_k
        fragmentos = self.retrieve(consulta=pregunta, cantidad_resultados=max(1, cantidad_solicitada))

        if not fragmentos:
            return "No encontre informacion relevante en los documentos indexados.", []

        contexto = self._build_context(fragmentos)
        instruccion = self._build_prompt(pregunta, contexto)

        try:
            respuesta = self._generate_answer(instruccion)
            if self._answer_looks_uncertain(respuesta) and self._should_retry_extraction(pregunta, fragmentos):
                instruccion_rescate = self._build_rescue_prompt(consulta=pregunta, contexto=contexto)
                respuesta_rescate = self._generate_answer(instruccion_rescate)
                if respuesta_rescate:
                    respuesta = respuesta_rescate
        except Exception as excepcion:
            logger.exception("LLM generation failed; returning retrieval fallback: %s", excepcion)
            respuesta = self._fallback_answer(motivo=self._llm_error_hint(excepcion))

        if not respuesta:
            respuesta = "No se pudo generar una respuesta con el contexto disponible."

        return respuesta, fragmentos
