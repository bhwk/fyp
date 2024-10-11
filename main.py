import asyncio
import aiofiles
from typing import List
from chromadb.utils import embedding_functions
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
import chromadb
from collections import deque


sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(  # pyright: ignore [reportAttributeAccessIssue]
    model_name="all-MiniLM-L6-v2"
)


def create_chroma_db(documents: List[str], path: str, name: str):
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.create_collection(
        name=name, embedding_function=sentence_transformer_ef
    )
    for i, d in enumerate(documents):
        db.add(documents=[d], ids=[str(i)])
    return db


def load_chroma_collection(path: str, name: str):
    chroma_client = chromadb.PersistentClient(path=path)
    return chroma_client.get_collection(
        name=name, embedding_function=sentence_transformer_ef
    )


def get_relevant_document(query: str, db, n_results: int):
    results = db.query(query_texts=[query], n_results=n_results)
    return [doc[0] for doc in results["documents"]]


def read_file(path: Path):
    with path.open() as f:
        return f.read()


async def load_file(filename):
    async with aiofiles.open(filename, mode="r") as f:
        return await f.read()


async def process_batch(batch):
    tasks = [load_file(filename) for filename in batch]
    return await asyncio.gather(*tasks)


async def load_files_async(dir_path, batch_size=1000):
    filenames = [file for file in dir_path.iterdir()]
    results = []

    file_queue = deque(filenames)

    while file_queue:
        batch = [file_queue.popleft() for _ in range(min(batch_size, len(file_queue)))]
        batch_results = await process_batch(batch)
        results.extend(batch_results)

        print(f"Processed {len(results)} files out of {len(filenames)}")

    return results


def load_documents(dir_path: Path) -> List[str]:
    text_files = []

    with ThreadPoolExecutor() as executor:
        results = executor.map(read_file, dir_path.iterdir())

        text_files = list(results)
    return text_files


async def main():
    # check if folder for db exists
    db_folder = "chroma_db"
    if not os.path.exists(db_folder):
        os.makedirs(db_folder)

    ## Comment this when I don't have to create the db from scratch
    # Load documents to create db
    FLAT_FILE_PATH = "./temp/flat"
    text_path = Path(FLAT_FILE_PATH)
    print("Loading documents into DB...please wait.")
    text_files = await load_files_async(text_path)
    db_path = os.path.join(os.getcwd(), db_folder)
    db = create_chroma_db(text_files, db_path, "patient_db")
    print("Creation complete.")


if __name__ == "__main__":
    asyncio.run(main())
