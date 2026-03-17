import pytest
from fastapi import HTTPException

from api.service import AppService


@pytest.mark.parametrize("filename", ["report.pdf", "a-b_c.pdf", "X.PDF"])
def test_validate_filename_accepts_pdf(filename: str) -> None:
    assert AppService._validar_nombre_archivo(filename).lower().endswith(".pdf")


@pytest.mark.parametrize("filename", ["", " ", "report.txt", "virus.exe"])
def test_validate_filename_rejects_invalid_files(filename: str) -> None:
    with pytest.raises(HTTPException):
        AppService._validar_nombre_archivo(filename)
