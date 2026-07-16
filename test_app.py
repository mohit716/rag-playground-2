from unittest.mock import MagicMock, patch

from app import (
    summarizePdfText,
    extractTextFromPdf,
    convert_to_embedding,
    storeVector,
    retrieveDocument,
    break_text,
    vector_store,
)


def test_extract_text_from_pdf():
    pdf_path = r"C:\Users\mohit\rag-playground-2\Employment Agreement Excerpt.pdf"

    text = extractTextFromPdf(pdf_path)

    assert isinstance(text, str)
    assert len(text) > 0



@patch("app.client")
def test_summarize_pdf_text(mock_client):
    mock_response = MagicMock()
    mock_response.output_text = "This document describes a Legal RAG roadmap."

    mock_client.responses.create.return_value = mock_response

    result = summarizePdfText("Sample PDF text")

    assert result == "This document describes a Legal RAG roadmap."

    mock_client.responses.create.assert_called_once()

    call_arguments = mock_client.responses.create.call_args.kwargs

    assert call_arguments["model"] == "gpt-5-nano"
    assert call_arguments["input"][0]["role"] == "user"
    assert "Sample PDF text" in call_arguments["input"][0]["content"]





@patch("app.client")
def test_convert_to_embedding(mock_client):
    mock_response = MagicMock()
    mock_response.data[0].embedding = [0.1, 0.2, 0.3]

    mock_client.embeddings.create.return_value = mock_response

    result = convert_to_embedding("Sample text")

    assert result == [0.1, 0.2, 0.3]

    mock_client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input="Sample text"
    )




def test_store_vector():
    vector_store.clear()

    text = "Sample document text"
    embedding = [0.1, 0.2, 0.3]

    result = storeVector(text, embedding)

    assert result == {
        "text": "Sample document text",
        "embedding": [0.1, 0.2, 0.3]
    }

    assert len(vector_store) == 1
    assert vector_store[0] == result




def test_retrieve_document_when_store_is_empty():
    vector_store.clear()

    result = retrieveDocument()

    assert result is None


def test_retrieve_document_returns_first_record():
    vector_store.clear()

    first_record = storeVector(
        "First document",
        [0.1, 0.2, 0.3]
    )

    storeVector(
        "Second document",
        [0.4, 0.5, 0.6]
    )

    result = retrieveDocument()

    assert result == first_record
    assert result["text"] == "First document"




def test_break_text_into_chunks():
    text = "abcdefghij"

    chunks = break_text(text, chunk_size=4)

    assert chunks == [
        "abcd",
        "efgh",
        "ij"
    ]


def test_break_text_when_text_is_smaller_than_chunk_size():
    text = "hello"

    chunks = break_text(text, chunk_size=10)

    assert chunks == ["hello"]


def test_break_text_with_empty_text():
    chunks = break_text("", chunk_size=10)

    assert chunks == []


def test_break_text_chunks_do_not_exceed_chunk_size():
    text = "This is some sample text for chunking."

    chunks = break_text(text, chunk_size=10)

    assert all(len(chunk) <= 10 for chunk in chunks)
