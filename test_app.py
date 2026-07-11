from unittest.mock import MagicMock, patch

from app import summarizePdfText

from app import extractTextFromPdf

from app import convert_to_embedding

from app import storeVector, vector_store


def test_extract_text_from_pdf():
    pdf_path = r"C:\Users\mohit\Downloads\project-plan.pdf"

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
