from api.rag import RagEngine, RetrievedChunk


def make_chunk(
    chunk_id: str,
    source: str,
    page: int,
    score: float,
    text: str,
) -> RetrievedChunk:
    return RetrievedChunk(id=chunk_id, source=source, page=page, score=score, text=text)


def test_keywords_strip_stopwords_and_accents() -> None:
    engine = RagEngine.__new__(RagEngine)
    tokens = engine._keywords("HablamE del lab1, con que dataset se entrena?")
    assert "lab1" in tokens
    assert "dataset" in tokens
    assert "entrena" in tokens
    assert "del" not in tokens
    assert "con" not in tokens


def test_expand_query_terms_adds_plural_and_multilingual_variants() -> None:
    engine = RagEngine.__new__(RagEngine)
    terms = engine._expand_query_terms(["tiempos", "muertos", "equipo"])
    assert "tiempo" in terms
    assert "temps" in terms
    assert "morts" in terms
    assert "equip" in terms


def test_weighted_query_terms_adds_semantic_and_fuzzy_variants() -> None:
    engine = RagEngine.__new__(RagEngine)
    engine._lex_postings = {"equip": [(0, 1)], "temps": [(0, 1)], "morts": [(0, 1)]}
    engine._lex_term_ngrams = {term: engine._char_ngrams(term) for term in engine._lex_postings}
    engine._ngram_to_terms = {}
    for term, ngrams in engine._lex_term_ngrams.items():
        for ngram in ngrams:
            engine._ngram_to_terms.setdefault(ngram, set()).add(term)

    weighted = engine._weighted_query_terms("cuantos tiempos muertos tiene un equipoo")
    assert "temps" in weighted
    assert "morts" in weighted
    assert "equip" in weighted


def test_fuse_rankings_combines_dense_and_lexical() -> None:
    engine = RagEngine.__new__(RagEngine)
    dense = [
        make_chunk("a", "doc-a.pdf", 1, 0.91, "dense a"),
        make_chunk("b", "doc-b.pdf", 2, 0.88, "dense b"),
    ]
    lexical = [
        make_chunk("c", "doc-c.pdf", 3, 3.4, "lex c"),
        make_chunk("a", "doc-a.pdf", 1, 2.9, "lex a longer text"),
    ]

    fused = engine._fuse_rankings(
        fragmentos_densos=dense, fragmentos_lexicos=lexical, cantidad_resultados=3
    )

    assert len(fused) == 3
    assert {chunk.id for chunk in fused} == {"a", "b", "c"}
    chunk_a = next(chunk for chunk in fused if chunk.id == "a")
    assert "longer" in chunk_a.text


def test_fuse_rankings_falls_back_when_one_side_empty() -> None:
    engine = RagEngine.__new__(RagEngine)
    dense = [make_chunk("a", "doc-a.pdf", 1, 0.91, "dense a")]
    lexical = [make_chunk("b", "doc-b.pdf", 1, 3.1, "lex b")]

    assert (
        engine._fuse_rankings(
            fragmentos_densos=dense, fragmentos_lexicos=[], cantidad_resultados=1
        )
        == dense
    )
    assert (
        engine._fuse_rankings(
            fragmentos_densos=[], fragmentos_lexicos=lexical, cantidad_resultados=1
        )
        == lexical
    )


def test_rerank_by_query_coverage_prioritizes_matching_numeric_chunk() -> None:
    engine = RagEngine.__new__(RagEngine)
    engine._chunk_text_norm_by_id = {}
    engine._lex_postings = {"temps": [(0, 1)], "morts": [(0, 1)], "equip": [(0, 1)]}
    engine._lex_term_ngrams = {term: engine._char_ngrams(term) for term in engine._lex_postings}
    engine._ngram_to_terms = {}
    for term, ngrams in engine._lex_term_ngrams.items():
        for ngram in ngrams:
            engine._ngram_to_terms.setdefault(ngram, set()).add(term)

    chunks = [
        make_chunk("a", "doc-a.pdf", 1, 0.9, "Texto general sin el dato."),
        make_chunk("b", "doc-b.pdf", 2, 0.9, "Cada equip te 2 temps morts per partit."),
    ]
    reranked = engine._rerank_by_query_coverage(
        fragmentos=chunks,
        consulta="cuantos tiempos muertos tiene un equipo",
        cantidad_resultados=2,
    )
    assert reranked[0].id == "b"


def test_answer_looks_uncertain_detects_no_se_patterns() -> None:
    assert RagEngine._answer_looks_uncertain("No sé.")
    assert RagEngine._answer_looks_uncertain("El contexto proporcionado no incluye ese dato.")
    assert not RagEngine._answer_looks_uncertain("La norma indica 2 tiempos muertos por equipo.")
