import llama_index.core
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    get_response_synthesizer,
)
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb
import logging
import sys
from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
)

llama_index.core.set_global_handler("simple")


def get_db(callback_manager):
    db = chromadb.PersistentClient("./chroma_db/")
    chroma_collection = db.get_collection(
        "patient_db",
    )
    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection,
    )
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        callback_manager=callback_manager,
    )
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    return index


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    # Set to local llm
    Settings.llm = Ollama(model="openhermes", request_timeout=500)

    # uncomment on first run
    # index = create_db(callback_manager)

    # retrieves ALREADY created db
    index = get_db(callback_manager)
    query_engine = index.as_query_engine()

    while True:
        user_input = input("Enter query: ")
        if not user_input:
            continue
        else:
            response = query_engine.query(f"{user_input}")
            print(response)
