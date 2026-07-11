

from openai import OpenAI
from dotenv import load_dotenv
import os
from pypdf import PdfReader

# Load API key from .env
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
vector_store = []


# Send message to LLM
def sayHiToLLM():
    response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[
        {"role": "user", "content": "Hi"}
    ]
    )
    # Print LLM reply
    print(response.choices[0].message.content)



def extractTextFromPdf(pdf_path):

    # Open and read the PDF
    reader = PdfReader(pdf_path)

    pdf_text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            pdf_text += page_text + "\n"

    return pdf_text


def summarizePdfText(pdf_text):
    response = client.responses.create(
        model="gpt-5-nano",
        input=[
            {
                "role": "user",
                "content": f"""
In 3-4 sentences of plain prose, tell me what this document is about and the overall approach it takes. Do not use bullet points and do not list the phases.

{pdf_text}
"""
            }
        ]
    )

    return response.output_text

def convert_to_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    embedding = response.data[0].embedding

    return embedding

def storeVector(text, embedding):
    vector_record = {
        "text": text,
        "embedding": embedding
    }

    vector_store.append(vector_record)

    return vector_record


if __name__=="__main__":

    pdf_path = r"C:\Users\mohit\Downloads\project-plan.pdf"

    # Step 1: Extract text from PDF
    pdf_text= extractTextFromPdf(pdf_path)

    # Step 2: Convert text into an embedding
    document_embedding = convert_to_embedding(pdf_text)

    # Step 3: Store text and embedding
    stored_document = storeVector(
        text=pdf_text,
        embedding=document_embedding
    )

    print("Document stored successfully.")
    print("Number of stored documents:", len(vector_store))
    print("Number of embedding values:", len(stored_document["embedding"]))
