import pytest
from pathlib import Path

from scripts.chunk_text import parse_page


def test_parse_page_handles_valid_payload() -> None:
    record = parse_page(
        linea='{"source":"demo.pdf","page":"3","text":"abc"}',
        ruta_jsonl=Path("data/processed/demo.jsonl"),
        numero_linea=1,
    )
    assert record.source == "demo.pdf"
    assert record.page == 3
    assert record.text == "abc"


def test_parse_page_raises_on_invalid_json() -> None:
    with pytest.raises(ValueError):
        parse_page("{bad json}", Path("data/processed/demo.jsonl"), 9)
