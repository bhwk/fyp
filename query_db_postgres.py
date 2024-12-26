import logging
import sys
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    PromptTemplate,
    StorageContext,
    load_index_from_storage,
    get_response_synthesizer,
)
from llama_index.core.indices.base import BaseIndex
from llama_index.core.storage.docstore.simple_docstore import DocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
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
    docstore = DocumentStore.from_persist_dir("./index/")
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, index_store=index_store, docstore=docstore
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


def determine_nodes_context(nodes: List[NodeWithScore], query: str) -> str:
    """
    Given an input query and a node, determine if the node contains information that will answer the query,
    and extract only the relevant information from this node
    """
    qa_prompt_tmpl = (
        "You will be provided with a chunk of information."
        "Given the context information and prior knowledge, "
        "evalute the context information against the query and determine if there is information that can answer the query."
        "IF so, generate a response that contains the relevant information structured such that it answers the query"
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


def retrieve_context(
    vector_index: VectorStoreIndex, keyword_index: BaseIndex, query: str
) -> List[NodeWithScore]:
    """
    Given an input query, run both semantic and keyword search over the query, returning the most relevant nodes.
    """

    # create node retrievers
    vector_retriever = vector_index.as_retriever(
        vector_store_query_mode="hybrid", sparse_top_k=2, verbose=True
    )
    keyword_retriever = keyword_index.as_retriever(verbose=True)

    # use llm to extract keywords from query for better performance
    extracted_keywords = Settings.llm.complete(
        f"Based on the following query, extract the keywords and output as plaintext, using commas to separate multiple keywords. Query: {query}"
    )

    # retrieve nodes
    vector_nodes: List[NodeWithScore] = vector_retriever.retrieve(query)
    keyword_nodes: List[NodeWithScore] = keyword_retriever.retrieve(
        extracted_keywords.text
    )

    # Return combined list
    return vector_nodes + keyword_nodes


def synthesize_context(query: str, relevant_text: str) -> str:
    """Given the input context and query, generate synthetic context and query that removes PII"""

    response = Settings.llm.complete(
        f"""
        Given the following context information, perform the following operations:

        Anonymization: The patient’s name and any identifying information must be removed or replaced with placeholders (e.g., "[Anonymized]").
        Clinical Data Requirements:
            Summarize relevant vitals (e.g., blood pressure, BMI, glucose levels) with appropriate medical context.
            Round all values.
        Tone and Clarity:
            Use formal and professional language. Avoid abbreviations unless they are common medical terms (e.g., "BP" for blood pressure).
            Write in full sentences, ensuring clarity for medical professionals reviewing the report.

        Formatting:
            Preserve original form of information presented

        CONTEXT INFORMATION:
        {relevant_text}
        QUERY:
        {query}
    """
    )

    return str(response)


if __name__ == "__main__":
    # Set to local llm
    # Settings.llm = Ollama(
    #     model="mistral-nemo", request_timeout=3600, context_window=10000
    # )
    Settings.llm = Gemini(
        model="models/gemini-1.5-flash",
        api_key="AIzaSyBoOaUIrtSIemKQROi7IFijhG-2CDN-AIA",
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-en-v1.5",
    )

    vector_index, keyword_index = get_db()

    query = "Show me blood pressure readings for Monserrate4_Mills423"

    context_nodes = retrieve_context(vector_index, keyword_index, query)

    relevant_text = determine_nodes_context(context_nodes, query)
    print("Extracted text:\n", relevant_text)

    # response = Settings.llm.complete(
    #     f"RETRIEVED INFORMATION:{relevant_text}\nQUERY:{query}"
    # )
    #
    # print(str(response))

    synth_context = synthesize_context(query, relevant_text)
    print("Synth context:\n", synth_context)
    synth_response = Settings.llm.complete(f"CONTEXT:{synth_context}\nQUERY:{query}")

    print(str(synth_response))
