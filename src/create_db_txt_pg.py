from json import load
import logging
import sys
import os
import nltk
from llama_index.core import (
    VectorStoreIndex,
    KeywordTableIndex,
    Settings,
    StorageContext,
    SimpleDirectoryReader,
)
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import pathlib
from llama_index.vector_stores.postgres import PGVectorStore
import psycopg2
import psycopg2.pool
from sqlalchemy import make_url

# Uncomment to see debug logs
# logging.basicConfig(stream=sys.stdout.flush(), level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout.flush()))

file_path = os.path.dirname(os.path.abspath(__file__))

FILE_DIR = os.path.join(file_path, "temp", "flat")


# FILE_DIR = pathlib.Path("./temp/flat/")
#


async def create_db():
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-en-v1.5",
        embed_batch_size=100,
    )

    documents = SimpleDirectoryReader(input_dir=FILE_DIR, recursive=True).load_data(
        num_workers=10, show_progress=True
    )

    connection_string = "postgresql://postgres:password@postgres:5432"
    conn = psycopg2.connect(connection_string)
    conn.autocommit = True

    connection_string = os.environ.get("DATABASE_URL")
    url = make_url(connection_string)  # type: ignore

    with conn.cursor() as c:
        c.execute(f"DROP DATABASE IF EXISTS {url.database}")
        c.execute(f"CREATE DATABASE {url.database}")

    vector_store = PGVectorStore.from_params(
        database=url.database,
        host=url.host,
        password=url.password,
        port=url.port,  # type: ignore
        user=url.username,
        table_name="patient_records",
        embed_dim=768,  # openai embedding dimension
        hybrid_search=True,
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
    )

    vector_index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
        use_async=True,
    )

    keyword_index = KeywordTableIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
        use_async=True,
    )

    storage_context.persist("./index/")


if __name__ == "__main__":
    import asyncio
    import nest_asyncio

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
    # Set to local llm

    Settings.llm = Ollama(
        model="qwen2.5:32b",
        base_url=os.environ.get("OLLAMA_URL"),  # type: ignore
        request_timeout=1000,
        context_window=8000,
    )

    nest_asyncio.apply()

    asyncio.run(create_db())
