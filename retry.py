from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
)
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


def create_db(callback_manager):
    documents = SimpleDirectoryReader(
        input_dir="./temp/flat", recursive=True
    ).load_data(show_progress=True)

    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        parallel_process=True,
        callback_manager=callback_manager,
        embed_batch_size=100,
    )

    db = chromadb.PersistentClient("./chroma_db/")
    chroma_collection = db.get_or_create_collection(
        "patient_db",
        metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 600,
            "hnsw:search_ef": 1000,
            "hnsw:M": 60,
        },
    )
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        callback_manager=callback_manager,
        show_progress=True,
    )

    return index


def get_db():
    db = chromadb.PersistentClient("./chroma_db/")
    chroma_collection = db.get_collection(
        "patient_db",
    )
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    return index


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    # Set to local llm
    Settings.llm = Ollama(model="openhermes", request_timeout=120.0)

    # index = create_db(callback_manager)
    index = get_db()
    query_engine = index.as_query_engine()

    while True:
        user_input = input("Enter query: ")
        if not user_input:
            continue
        else:
            response = query_engine.query(f"{user_input}")
            print(response)
