from unittest.mock import MagicMock, patch

from app import summarizePdfText

from app import extractTextFromPdf


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
