import pytest
from pydantic_ai import RunContext
from app.services.tools.document_retrieval_tool import get_document_title

def test_get_document_title_strips_timestamp():
    # Arrange
    doc_name = "Introducing Semantics_20260503_010549"
    
    # Act
    title = get_document_title(None, doc_name) # type: ignore
    
    # Assert
    assert title == "Introducing Semantics"

def test_get_document_title_no_timestamp():
    # Arrange
    doc_name = "My Simple Document"
    
    # Act
    title = get_document_title(None, doc_name) # type: ignore
    
    # Assert
    assert title == "My Simple Document"

def test_get_document_title_partial_timestamp_not_stripped():
    # Arrange
    doc_name = "Document_20260503"
    
    # Act
    title = get_document_title(None, doc_name) # type: ignore
    
    # Assert
    assert title == "Document_20260503"
