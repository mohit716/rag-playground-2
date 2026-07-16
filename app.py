

from openai import OpenAI
from dotenv import load_dotenv
import os
from pypdf import PdfReader
import math

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

def break_text(text, chunk_size=500):
    chunks = []

    for start in range(0, len(text), chunk_size):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)

    return chunks



def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)



def search_right_chunks(query, top_k=1):
    if not vector_store:
        return []

    query_embedding = convert_to_embedding(query)

    scored_chunks = []

    for record in vector_store:
        similarity_score = cosine_similarity(
            query_embedding,
            record["embedding"]
        )

        scored_chunks.append({
            "text": record["text"],
            "embedding": record["embedding"],
            "score": similarity_score
        })

    scored_chunks.sort(
        key=lambda record: record["score"],
        reverse=True
    )

    return scored_chunks[:top_k]



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


def retrieveDocument():
    if not vector_store:
        return None

    return vector_store[0]


if __name__=="__main__":

    # Step 0: Load the PDF
    pdf_path = r"C:\Users\mohit\rag-playground-2\Employment Agreement Excerpt.pdf"

    # Step 1: Extract text from PDF
    pdf_text= extractTextFromPdf(pdf_path)

     # Step 2: Break text into chunks
    chunks = break_text(pdf_text)
    print("Number of chunks:", len(chunks))
    print()

    # Step 3: Convert each chunk into an embedding and store it
    for chunk in chunks:
        chunk_embedding = convert_to_embedding(chunk)
        storeVector(
            text=chunk,
            embedding=chunk_embedding
        )


        
    print("Stored records:", len(vector_store))
    print()

    # Preview the first stored chunk
    print("First stored chunk:")
    print(vector_store[0]["text"])
    print()

    print("First embedding preview:")
    print(vector_store[0]["embedding"][:10])
    print()

  
     # Step 4: Ask a question
    query = "Under what circumstances can the employment agreement be terminated immediately, and how much notice is normally required?"

    # Step 5: Search for the most relevant chunks
    retrieved_chunks = search_right_chunks(
        query=query,
        top_k=2
    )

    if retrieved_chunks:
        print("Relevant chunks found:")
        print()

        for index, result in enumerate(retrieved_chunks):
            print(f"Result {index + 1}:")
            print("Similarity score:", result["score"])
            print("Retrieved text:")
            print(result["text"])
            print()
            print()
    else:
        print("No relevant chunks found.")

    