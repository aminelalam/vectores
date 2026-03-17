from scripts.extract_text import clean_text


def test_clean_text_removes_extra_spaces_and_controls() -> None:
    raw = "Hello\x01  world \n\n\nLine-\nbreak"
    cleaned = clean_text(raw)
    assert cleaned == "Hello world\n\nLinebreak"


def test_clean_text_repairs_common_mojibake() -> None:
    raw = "CÃ³digo ensamblador"
    cleaned = clean_text(raw)
    assert cleaned == "C\u00f3digo ensamblador"
