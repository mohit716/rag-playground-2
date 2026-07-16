from unittest.mock import MagicMock, patch
import pytest


from app import (
    cosine_similarity,
    search_right_chunks,
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


def test_cosine_similarity_same_vectors():
    vector_a = [1.0, 2.0, 3.0]
    vector_b = [1.0, 2.0, 3.0]

    result = cosine_similarity(vector_a, vector_b)

    assert result == pytest.approx(1.0)



def test_cosine_similarity_orthogonal_vectors():
    vector_a = [1.0, 0.0]
    vector_b = [0.0, 1.0]

    result = cosine_similarity(vector_a, vector_b)

    assert result == pytest.approx(0.0)


def test_cosine_similarity_with_zero_vector():
    result = cosine_similarity(
        [0.0, 0.0],
        [1.0, 2.0]
    )

    assert result == 0.0



@patch("app.convert_to_embedding")
def test_search_right_chunks_returns_most_similar_chunk(
    mock_convert_to_embedding
):
    vector_store.clear()

    # Fake embedding for the user's query
    mock_convert_to_embedding.return_value = [1.0, 0.0]

    storeVector(
        text="This section explains termination.",
        embedding=[0.9, 0.1]
    )

    storeVector(
        text="This section explains employee benefits.",
        embedding=[0.0, 1.0]
    )

    results = search_right_chunks(
        query="What does the agreement say about termination?",
        top_k=1
    )

    assert len(results) == 1
    assert results[0]["text"] == "This section explains termination."
    assert "score" in results[0]

    mock_convert_to_embedding.assert_called_once_with(
        "What does the agreement say about termination?"
    )


@patch("app.convert_to_embedding")
def test_search_right_chunks_when_store_is_empty(
    mock_convert_to_embedding
):
    vector_store.clear()

    results = search_right_chunks(
        query="What does the agreement say?",
        top_k=2
    )

    assert results == []
    mock_convert_to_embedding.assert_not_called()





@patch("app.convert_to_embedding")
def test_search_right_chunks_respects_top_k(
    mock_convert_to_embedding
):
    vector_store.clear()

    mock_convert_to_embedding.return_value = [1.0, 0.0]

    storeVector("Chunk one", [1.0, 0.0])
    storeVector("Chunk two", [0.8, 0.2])
    storeVector("Chunk three", [0.0, 1.0])

    results = search_right_chunks(
        query="Sample question",
        top_k=2
    )

    assert len(results) == 2
    assert results[0]["text"] == "Chunk one"
    assert results[1]["text"] == "Chunk two"