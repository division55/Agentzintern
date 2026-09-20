import ollama
import chromadb

from chromadb.api.types import EmbeddingFunction, Documents, Embeddings


class NomicEmbeddingFunction(EmbeddingFunction):

    def __init__(self, model_name="nomic-embed-text"):
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        response = ollama.embed(
            model=self.model_name,
            input=input
        )

        return response["embeddings"]


# Connect to the existing Week 3 database
chroma_client = chromadb.PersistentClient(
    path="./chroma_db_week3"
)

nomic_ef = NomicEmbeddingFunction()

collection = chroma_client.get_collection(
    name="samhita_dental",
    embedding_function=nomic_ef
)


def query(question, top_k=3):

    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return documents, metadatas, distances

if __name__ == "__main__":

    question = "How much does a dental implant cost?"

    documents, metadatas, distances = query(question)

    print("\nQUESTION:")
    print(question)

    print("\nRETRIEVED CHUNKS:")

    for i, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances), 1
    ):

        print(f"\n--- RESULT {i} ---")
        print("Source:", metadata["source"])
        print("Type:", metadata["type"])
        print("Distance:", distance)
        print("Content:", document)