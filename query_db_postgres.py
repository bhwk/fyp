import logging
import sys
from llama_index.core import VectorStoreIndex, Settings, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url

# Uncomment to see debug logs
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def get_db():
    connection_string = "postgresql://postgres:password@localhost:5432"
    db_name = "vector_db"
    url = make_url(connection_string)

    vector_store = PGVectorStore.from_params(
        database=db_name,
        host=url.host,
        password=url.password,
        port=url.port,  # type: ignore
        user=url.username,
        table_name="patient_records",
        hybrid_search=True,
        embed_dim=768,
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )

    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-en-v1.5",
        callback_manager=callback_manager,
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )
    return index


if __name__ == "__main__":
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    # Set to local llm
    Settings.llm = Ollama(model="openhermes", request_timeout=500)

    index = get_db()

    query_engine = index.as_query_engine(
        similarity_top_k=2,
        sparse_top_k=12,
        vector_store_query_mode="hybrid",
    )

    response = query_engine.query("Which patients have a history of cardiac arrest?")

    print(response)
