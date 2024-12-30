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
from llama_index.core.postprocessor import (
    KeywordNodePostprocessor,
    SimilarityPostprocessor,
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
        storage_context=storage_context, index_id="af9d3fe5-e4fb-48f6-bc54-f9b5cab1454d"
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
    Given an input query and a list of nodes, extract only the relevant information from each node that will answer the query.
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


def retrieve_context(query: str) -> List[NodeWithScore]:
    """
    Given an input query, run both semantic and keyword search over the query, returning the most relevant nodes.
    """

    # create node retrievers
    vector_retriever = VECTOR_INDEX.as_retriever(
        vector_store_query_mode="hybrid", sparse_top_k=2, verbose=True
    )
    keyword_retriever = KEYWORD_INDEX.as_retriever(verbose=True)

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


def search_for_patient_observations(patient_name: str, query: str):
    """
    Takes in an input query to search for observations (e.g. blood pressure, glucose levels, etc.). Matches retrieved nodes to patient_name, and returns nodes that contain the patient's name.
    """

    vector_retriever = VECTOR_INDEX.as_retriever(
        vector_store_query_mode="hybrid", sparse_top_k="25", verbose=True
    )

    nodes: List[NodeWithScore] = vector_retriever.retrieve(query)

    filtered_nodes = list(filter(lambda node: patient_name in node.text, nodes))

    postprocessor = SimilarityPostprocessor(similarity_cutoff=0.5)

    return postprocessor.postprocess_nodes(filtered_nodes)


def synthesize_content(query: str, relevant_text: str) -> str:
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

        You are to also generate a synthetic query that can be answered by the generated context.
        Output in the format:

        "Synthetic Query": GENERATED QUERY,
        "Synthetic Context": GENERATED CONTEXT

        CONTEXT INFORMATION:
        {relevant_text}
        QUERY:
        {query}
    """
    )

    return str(response)


def search_for_patients_with_specified_condition(condition: str) -> List[NodeWithScore]:
    """
    Takes in an input string of specific condition name to look up. Returns a list of 10 patient nodes that contain the condition name.
    """
    keyword_retriever = KEYWORD_INDEX.as_retriever(
        verbose=True, retriever_mode="Simple"
    )
    nodes = keyword_retriever.retrieve(condition)

    print(nodes)

    filtered_nodes = list(filter(lambda node: condition in node.text, nodes))

    return filtered_nodes


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

    VECTOR_INDEX, KEYWORD_INDEX = get_db()

    query = "What do blood pressure readings for Monserrate4_Mills423 look like?"

    condition_nodes = search_for_patients_with_specified_condition("hypertension")
    print(len(condition_nodes))
    print(condition_nodes)

    # llm_planning = Settings.llm.complete(f"""
    # Given the following query, think step by step and break it down into steps on how you would answer it.
    #
    # You have access to the following tools:
    # - search_for_condition(condition: str) -> List[NodeWithScore]: Takes in an input string of condition name to look up. Returns a list of patient nodes that contain the condition name.
    # - retrieve_context: Given an input query, run both semantic and keyword search over the query, returning the most relevant nodes. May not return all nodes that contain the right information.
    # - determine_nodes_context(nodes: List[NodeWithScore], query: str): Given an input query and a list of nodes, extract only the relevant information from each node that will answer the query.
    #
    # Make use of all tools you have to answer the query.
    #
    # Do not explain your steps. Output in the following format:
    # Step 1: ---
    # Step 2: ---
    # ... and so on.
    #
    # Query: {query}
    # """)

    # print(llm_planning.text)
