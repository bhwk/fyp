from importlib.metadata import metadata
import os
import chromadb
from chromadb.utils import embedding_functions
from pprint import pprint
import ollama

SENTENCE_TRANSFORMER = embedding_functions.SentenceTransformerEmbeddingFunction(  # pyright: ignore [reportAttributeAccessIssue]
    model_name="BAAI/bge-small-en-v1.5"
)

PROMPT_TEMPLATE = """
Use the following context to enhance your answer to the query:
{context}
 - -
Answer the query based on the above context: {query}
"""


def load_chroma_collection(path: str, name: str):
    chroma_client = chromadb.PersistentClient(path=path)
    return chroma_client.get_collection(
        name=name,
        embedding_function=SENTENCE_TRANSFORMER,
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

    index = "Arnold"
    query = "What can you tell me about Arnold?"

    results = get_relevant_document(index, db, 5)
    pprint(results)

    context_text = "\n\n - - \n\n".join(results)

    prompt = f"Use the following context to help your answer to the query: {context_text}- -Query: {query}"

    response = ollama.chat(
        model="openhermes", messages=[{"role": "system", "content": prompt}]
    )

    print(response["message"]["content"])


if __name__ == "__main__":
    main()
