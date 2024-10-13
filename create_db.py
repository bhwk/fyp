import asyncio
import aiofiles
from typing import List
from chromadb.utils import embedding_functions
from pathlib import Path
import os
import chromadb
from collections import deque
import time


SENTENCE_TRANSFORMER_EF = embedding_functions.SentenceTransformerEmbeddingFunction(  # pyright: ignore [reportAttributeAccessIssue]
    model_name="BAAI/bge-small-en-v1.5"
)


def create_chroma_db(documents: List[str], path: str, name: str):
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.create_collection(
        name=name, embedding_function=SENTENCE_TRANSFORMER_EF
    )
    for i, d in enumerate(documents):
        db.add(documents=[d], ids=[str(i)])
    return db


def load_chroma_collection(path: str, name: str):
    chroma_client = chromadb.PersistentClient(path=path)
    return chroma_client.get_collection(
        name=name, embedding_function=SENTENCE_TRANSFORMER_EF
    )


async def load_file(filename):
    async with aiofiles.open(filename, mode="r") as f:
        return await f.read()


async def process_batch(batch):
    tasks = [load_file(filename) for filename in batch]
    return await asyncio.gather(*tasks)


async def load_files_async(dir_path: Path, batch_size=100):
    dirs: list[Path] = [dir for dir in dir_path.iterdir() if dir.is_dir()]
    filenames = []
    # set to 100 patients for testing
    for dir in dirs[:100]:
        files = [file for file in dir.iterdir()]
        filenames.extend(files)

    results = []
    start_time = time.time()
    file_queue = deque(filenames)

    while file_queue:
        batch = [file_queue.popleft() for _ in range(min(batch_size, len(file_queue)))]
        batch_results = await process_batch(batch)
        results.extend(batch_results)

        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / len(results)) * len(filenames)
        remaining_time = estimated_total_time - elapsed_time
        print(
            f"Processed {len(results)}/{len(filenames)} "
            f"({len(results)/len(filenames)*100:.2f}%) "
            f"| Elapsed: {elapsed_time:.2f}s "
            f"| Remaining: {remaining_time:.2f}s "
        )

        # try not to overload HDD
        await asyncio.sleep(0.01)

    return results


async def main():
    # check if folder for db exists
    db_folder = "chroma_db"
    if not os.path.exists(db_folder):
        os.makedirs(db_folder)

    ## Comment this when I don't have to create the db from scratch
    # Load documents to create db
    FLAT_FILE_PATH = "./temp/flat"
    text_path = Path(FLAT_FILE_PATH)
    text_files = await load_files_async(text_path)
    db_path = os.path.join(os.getcwd(), db_folder)
    db = create_chroma_db(text_files, db_path, "patient_db")
    print("Creation complete.")


if __name__ == "__main__":
    asyncio.run(main())
