import logging
import sys
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
)
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool
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
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )
    return index


if __name__ == "__main__":
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    # Set to local llm
    Settings.llm = Ollama(model="mistral-nemo", request_timeout=500)

    index = get_db()
    query_engine = index.as_query_engine()
    query_tool = QueryEngineTool.from_defaults(
        query_engine,
        name="patient_records",
        description=(
            "A RAG engine with patient observations sorted by date."
            "There is a master record tagged with patient_info that contains information about the patient's conditions and medications"
            "Use a detailed plain text question as input to the tool."
        ),
    )

    agent = ReActAgent.from_tools([query_tool], verbose=True)

    response = agent.chat(
        "Which patients have a history of diabetes? Suggest a care plan for each patient that is still alive."
    )
    print(response)
