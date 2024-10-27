from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    Document,
)
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb
import logging
import sys
import json
import pathlib
from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
)

FILE_DIR = pathlib.Path("./temp/flat")


def load_json(file_path):
    with open(file_path, mode="r") as f:
        content = f.read()
        return json.loads(content)


def create_db(callback_manager):
    patient_objects = [load_json(file) for file in FILE_DIR.iterdir()]

    documents = [
        Document(doc_id=object["id"], text=object["text"], metadata=object["metadata"])  # type: ignore
        for object in patient_objects
    ]

    nodes = [
        TextNode(id_=object["id"], text=object["text"], metadata=object["metadata"])  # type: ignore
        for object in patient_objects
    ]

    # documents = SimpleDirectoryReader(
    #     input_dir="./temp/flat", recursive=True
    # ).load_data(show_progress=True)

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

    index = VectorStoreIndex(
        nodes=nodes,
        embed_model=embed_model,
        storage_context=storage_context,
        show_progress=True,
    )

    # index = VectorStoreIndex.from_documents(
    #     documents,
    #     storage_context=storage_context,
    #     embed_model=embed_model,
    #     callback_manager=callback_manager,
    #     show_progress=True,
    # )

    return index


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    # Set to local llm
    Settings.llm = Ollama(model="openhermes", request_timeout=500)

    index = create_db(callback_manager)
