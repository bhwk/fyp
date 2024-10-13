import os
import chromadb
from chromadb.utils import embedding_functions
from pprint import pprint
import ollama

SENTENCE_TRANSFORMER = embedding_functions.SentenceTransformerEmbeddingFunction(  # pyright: ignore [reportAttributeAccessIssue]
    model_name="BAAI/bge-small-en-v1.5"
)

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
"""


def load_chroma_collection(path: str, name: str):
    chroma_client = chromadb.PersistentClient(path=path)
    return chroma_client.get_collection(
        name=name, embedding_function=SENTENCE_TRANSFORMER
    )


def get_relevant_document(query: str, db, n_results: int):
    results = db.query(query_texts=[query], n_results=n_results)
    return results["documents"][0]


def main():
    db_folder = "chroma_db"
    if not os.path.exists(db_folder):
        os.makedirs(db_folder)

    db_path = os.path.join(os.getcwd(), db_folder)
    db = load_chroma_collection(db_path, "patient_db")

    query = "List records of interest for patient Ferdinand55"

    results = get_relevant_document(query, db, 10)

    context_text = "\n\n - - \n\n".join(results)

    prompt = f"Use the following context as reference when you answer the query: {context_text}- -Answer the question based on the above context: {query}"

    response = ollama.chat(
        model="openhermes", messages=[{"role": "system", "content": prompt}]
    )

    print(response["message"]["content"])


if __name__ == "__main__":
    main()
