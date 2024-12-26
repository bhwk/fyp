import logging
import sys
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    KeywordTableIndex,
    Settings,
    PromptTemplate,
    StorageContext,
    load_index_from_storage,
    get_response_synthesizer,
)
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
)
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url
from typing import List

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
    index_store = SimpleIndexStore.from_persist_dir("./index/")
    document_store = SimpleDocumentStore.from_persist_dir("./index/")
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, index_store=index_store, docstore=document_store
    )

    keyword_index = load_index_from_storage(
        storage_context=storage_context, index_id="f3a8b382-58da-4cf0-b93e-a35a18f54e55"
    )

    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-en-v1.5",
    )
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )
    return (vector_index, keyword_index)


def retrieve_context(index: VectorStoreIndex, query: str) -> List[NodeWithScore]:
    vector_retriever = index.as_retriever(
        vector_store_query_mode="default",
        similarity_top_k=5,
    )
    text_retriever = index.as_retriever(
        vector_store_query_mode="sparse",
        sparse_top_k=5,  # interchangeable with sparse_top_k in this context
    )
    retriever = QueryFusionRetriever(
        [vector_retriever, text_retriever],
        similarity_top_k=5,
        num_queries=1,  # set this to 1 to disable query generation
        mode="relative_score",  # type: ignore
        use_async=False,
    )
    nodes = retriever.retrieve(str_or_query_bundle=query)

    response_synthesizer = CompactAndRefine()
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )
    print(query_engine.query(query))

    return nodes


def query_as_engine(index: VectorStoreIndex, query):
    query_engine = index.as_query_engine(similarity_top_k=5)
    response = query_engine.query(query)

    print(response)


def determine_nodes_context(nodes: List[NodeWithScore], query: str) -> str:
    """
    Given an input query and a node, determine if the node contains information that will answer the query,
    and extract only the relevant information from this node
    """
    qa_prompt_tmpl = (
        "You will be provided with a chunk of information."
        "Given the context information and prior knowledge, "
        "evalute the context information against the query and determine if there is information that can answer the query."
        "IF so, generate a response that contains only the relevant information from this chunk of text "
        "that will HELP answer the query."
        "OTHERWISE, reply with NONE"
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_prompt = PromptTemplate(qa_prompt_tmpl)

    response_synthesizer = get_response_synthesizer(
        text_qa_template=qa_prompt, verbose=True
    )
    response = response_synthesizer.synthesize(query, nodes=nodes)
    return str(response)


if __name__ == "__main__":
    # Set to local llm
    Settings.llm = Ollama(
        model="mistral-nemo", request_timeout=3600, context_window=10000
    )
    # Settings.llm = Gemini(
    #     model="models/gemini-1.5-flash",
    #     api_key="AIzaSyBoOaUIrtSIemKQROi7IFijhG-2CDN-AIA",
    # )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-en-v1.5",
    )

    vector_index, keyword_index = get_db()

    vector_retriever = vector_index.as_retriever(
        vector_store_query_mode="hybrid", sparse_top_k=2, verbose=True
    )
    keyword_retriever = keyword_index.as_retriever(verbose=True)

    query = "Which patients have hypertension?"
    keyword_query = Settings.llm.complete(
        f"Based on the following query, extract the keywords and output as plaintext, using commas to separate multiple keywords. Query: {query}"
    ).text

    keyword_nodes = keyword_retriever.retrieve(keyword_query)
    print(keyword_nodes)
    vector_nodes = vector_retriever.retrieve(query)
    print(vector_nodes)

    print(
        "Extracted Keyword nodes context:\n",
        determine_nodes_context(keyword_nodes, query),
    )
    print(
        "Extracted Vector nodes context:\n",
        determine_nodes_context(vector_nodes, query),
    )

    # nodes = retrieve_context(index, query)
    #
    # print(determine_context(query, nodes))

    # synthetic_context = []
    # for node in nodes:
    #     print(node)
    #     synthetic_context.append(determine_node_context(node, query))
    #
    # synthetic_context = " ".join(synthetic_context)
    # print(synthetic_context)
    #
    # response = Settings.llm.complete(
    #     f"Based off the following context information, answer the query.CONTEXT:\n{synthetic_context}\nQUERY:\n{query}\nANSWER:\n"
    # )

    # print(response)
