from app import extractTextFromPdf


def test_extract_text_from_pdf():
    pdf_path = r"C:\Users\mohit\Downloads\project-plan.pdf"

    text = extractTextFromPdf(pdf_path)

    assert isinstance(text, str)
    assert len(text) > 0
