import logging
import sys
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    SimpleDirectoryReader,
)
from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import pathlib
from llama_index.vector_stores.postgres import PGVectorStore
import psycopg2
from sqlalchemy import make_url

# Uncomment to see debug logs
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

FILE_DIR = pathlib.Path("./temp/flat")


def create_db(callback_manager):
    documents = SimpleDirectoryReader(
        input_dir="./temp/flat", recursive=True
    ).load_data(show_progress=True)

    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-en-v1.5",
        parallel_process=True,
        callback_manager=callback_manager,
        embed_batch_size=100,
    )

    connection_string = "postgresql://postgres:password@localhost:5432"
    db_name = "vector_db"
    conn = psycopg2.connect(connection_string)
    conn.autocommit = True

    with conn.cursor() as c:
        c.execute(f"DROP DATABASE IF EXISTS {db_name}")
        c.execute(f"CREATE DATABASE {db_name}")

    url = make_url(connection_string)
    vector_store = PGVectorStore.from_params(
        database=db_name,
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

    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-en-v1.5",
        parallel_process=True,
        embed_batch_size=100,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    return index


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    # Set to local llm
    Settings.llm = Ollama(model="openhermes", request_timeout=500)

    index = create_db(callback_manager)
