import re
import chromadb
import ollama
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from pypdf import PdfReader
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)

INPUT_FILE = "samhita_faq.txt"
OUTPUT_FILE = "samhita_chunks.txt"


def load_text():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        return file.read()

def load_pdf(pdf_path):
    print(f"\n=== Loading PDF: {pdf_path} ===")

    reader = PdfReader(pdf_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    full_text = "\n".join(pages)

    print(f"Pages: {len(reader.pages)}")
    print(f"Characters extracted: {len(full_text)}")

    return full_text

def create_pdf_chunks(text, chunk_size=500, overlap=100):
    words = text.split()

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

def create_faq_chunks(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    chunks = []
    current_question = None
    current_answer = []

    for line in lines:

        if line.endswith("?"):
            if current_question and current_answer:
                chunks.append(
                    f"Question: {current_question}\n"
                    f"Answer: {' '.join(current_answer)}"
                )

            current_question = line
            current_answer = []

        elif current_question:
            current_answer.append(line)

    if current_question and current_answer:
        chunks.append(
            f"Question: {current_question}\n"
            f"Answer: {' '.join(current_answer)}"
        )

    return chunks


def clean_chunks(chunks):
    cleaned = []

    for chunk in chunks:
        if "View Treatment Pricing" in chunk:
            continue

        if "Book an Appointment" in chunk:
            continue

        cleaned.append(chunk)

    return cleaned

class NomicEmbeddingFunction(EmbeddingFunction):

    def __init__(self, model_name="nomic-embed-text"):
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        response = ollama.embed(
            model=self.model_name,
            input=input
        )

        return response["embeddings"]

def store_in_chromadb(faq_chunks, pdf_chunks):

    print("\n=== Creating Week 3 ChromaDB ===")

    chroma_client = chromadb.PersistentClient(
        path="./chroma_db_week3"
    )

    try:
        chroma_client.delete_collection("samhita_dental")
    except Exception:
        pass

    nomic_ef = NomicEmbeddingFunction()

    collection = chroma_client.create_collection(
        name="samhita_dental",
        embedding_function=nomic_ef,
        metadata={"hnsw:space": "cosine"}
    )
    collection._chroma_client = chroma_client

    documents = []
    ids = []
    metadatas = []

    # Add FAQ chunks
    for i, chunk in enumerate(faq_chunks, 1):

        documents.append(chunk)

        ids.append(f"faq_chunk_{i}")

        metadatas.append({
            "source": "Samhita Dental FAQ",
            "type": "faq",
            "chunk_id": str(i)
        })

    # Add PDF chunks
    for i, chunk in enumerate(pdf_chunks, 1):

        documents.append(chunk["text"])

        ids.append(f"pdf_chunk_{i}")

        metadatas.append({
            "source": chunk["source"],
            "type": "pdf",
            "chunk_id": chunk["chunk_id"]
        })

    print(f"\nTotal chunks to embed: {len(documents)}")

    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )

    print("All chunks successfully embedded and stored!")

    print(f"ChromaDB collection count: {collection.count()}")

    return collection

def test_retrieval(collection):

    print("\n=== Week 3 Retrieval Test ===\n")

    test_questions = [
        "How much does a dental implant cost?",
        "How long does a root canal take?",
        "What are the clinic's working hours?",
        "How much do clear aligners cost?",
        "How do I book an appointment?"
    ]

    for question in test_questions:

        print(f"Question: {question}")

        results = collection.query(
            query_texts=[question],
            n_results=3
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        print("\nTop 3 Retrieved Chunks:")

        for i, (document, metadata) in enumerate(
            zip(documents, metadatas), 1
        ):
            print(f"\n{i}. Source: {metadata.get('source')}")
            print(f"   Type: {metadata.get('type')}")
            print(f"   Chunk ID: {metadata.get('chunk_id')}")
            print(f"   Content: {document}")

        print("\n" + "-" * 70)

def evaluate_knowledge_base(collection):

    print("\n=== Week 3: 25-Question Evaluation ===\n")

    questions = [
        "What are the clinic's working hours?",
        "How do I book an appointment?",
        "How much does a dental implant cost at Samhita Dental?",
        "How much do clear aligners cost?",
        "How long does a root canal take?",
        "What should I bring to my first appointment?",
        "How do I care for clear aligners?",
        "How long is recovery after dental implant surgery?",
        "Does the clinic provide wheelchair accessibility?",
        "How often should I have routine dental checkups?",

        "What is a dental implant?",
        "What are the advantages of dental implants?",
        "What factors determine whether someone is suitable for dental implants?",
        "How long does it take for a dental implant to heal?",

        "What is root canal treatment?",
        "Why might someone need root canal treatment?",
        "What happens during root canal treatment?",
        "What should a patient expect after root canal treatment?",

        "What is fluoride used for?",
        "How does fluoride help prevent dental problems?",
        "What role does brushing play in oral health?",
        "How can dental visits help prevent oral health problems?",

        "Does Samhita Dental provide braces?",
        "Does Samhita Dental perform wisdom tooth surgery?",
        "Does Samhita Dental provide treatment for sleep apnea?"
    ]

    for number, question in enumerate(questions, 1):

        print("=" * 70)
        print(f"QUESTION {number}/25")
        print(question)

        results = collection.query(
            query_texts=[question],
            n_results=3
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        answer = generate_answer(question, documents)

        print("\n=== Generated Answer ===")
        print(answer)

        print("\nTop 3 Retrieved Chunks:")

        for i, (document, metadata, distance) in enumerate(
            zip(documents, metadatas, distances), 1
        ):

            print(f"\n{i}. Source: {metadata.get('source')}")
            print(f"   Type: {metadata.get('type')}")
            print(f"   Chunk ID: {metadata.get('chunk_id')}")
            print(f"   Distance: {distance:.4f}")
            print(f"   Content: {document[:700]}")

        print()

def generate_answer(question, documents):

    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )

    context = "\n\n".join(documents)

    prompt = f"""
You are a dental clinic information assistant.

Answer the user's question using ONLY the information provided
in the context below.

If the context does not contain enough information to answer
the question, say:
"I'm sorry, I don't have that information in the available
clinic knowledge base."

Do not invent facts.
Do not use outside knowledge.
Keep the answer clear and concise.

CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1
    )

    return response.choices[0].message.content
def main():

    # ==============================
    # LOAD SAMHITA FAQ
    # ==============================

    print("=== Loading Samhita Dental FAQ ===")

    text = load_text()

    print(f"Characters loaded: {len(text)}")

    faq_chunks = create_faq_chunks(text)

    print(f"FAQ chunks created: {len(faq_chunks)}")

    faq_chunks = clean_chunks(faq_chunks)

    print(f"Useful FAQ chunks after cleaning: {len(faq_chunks)}")


    # ==============================
    # SAVE FAQ CHUNKS
    # ==============================

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        for i, chunk in enumerate(faq_chunks, 1):

            file.write(f"--- CHUNK {i} ---\n")
            file.write(chunk)
            file.write("\n\n")

    print(f"\nChunks saved to: {OUTPUT_FILE}")


    # ==============================
    # SHOW FAQ CHUNKS
    # ==============================

    print("\n=== First 3 FAQ Chunks ===\n")

    for i, chunk in enumerate(faq_chunks[:3], 1):

        print(f"CHUNK {i}")
        print(chunk)
        print("-" * 60)


    print("\n=== Last 5 FAQ Chunks ===\n")

    for i, chunk in enumerate(
        faq_chunks[-5:],
        len(faq_chunks) - 4
    ):

        print(f"CHUNK {i}")
        print(chunk)
        print("-" * 60)


    # ==============================
    # LOAD PDFS
    # ==============================

    pdf_files = [
        "LN001941.pdf",
        "LN006096.pdf",
        "FTDP_March2007_2.pdf"
    ]

    all_pdf_chunks = []

    print("\n=== Processing PDFs ===")

    for pdf_file in pdf_files:

        pdf_text = load_pdf(pdf_file)

        pdf_chunks = create_pdf_chunks(pdf_text)

        print(f"PDF chunks created: {len(pdf_chunks)}")

        for i, chunk in enumerate(pdf_chunks, 1):

            all_pdf_chunks.append({
                "text": chunk,
                "source": pdf_file,
                "type": "pdf",
                "chunk_id": str(i)
            })


    # ==============================
    # PDF SUMMARY
    # ==============================

    print("\n=== PDF CHUNK SUMMARY ===")

    print(f"Total PDF chunks: {len(all_pdf_chunks)}")

    for item in all_pdf_chunks[:3]:

        print("\nSource:", item["source"])
        print("Chunk ID:", item["chunk_id"])
        print("Content:", item["text"][:500])


    # ==============================
    # STORE EVERYTHING
    # ==============================

    collection = store_in_chromadb(
        faq_chunks,
        all_pdf_chunks
    )
    evaluate_knowledge_base(collection)


    # ==============================
    # TEST RETRIEVAL
    # ==============================

    test_retrieval(collection)


if __name__ == "__main__":
    main()