import pytest
from src.document_processor import process_pdf

def test_process_pdf_valid_file():
    docs = process_pdf("test_data/sample.pdf")
    assert len(docs) > 0
    assert all(isinstance(doc, Document) for doc in docs)

def test_process_pdf_invalid_file():
    with pytest.raises(FileNotFoundError):
        process_pdf("nonexistent.pdf")