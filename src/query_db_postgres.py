import logging
import sys
import os
from custom_retriever import CustomRetriever
from llama_index.core.query_engine import RetrieverQueryEngine, SubQuestionQueryEngine
from llama_index.core.tools import QueryPlanTool
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    PromptTemplate,
    StorageContext,
    load_index_from_storage,
    get_response_synthesizer,
)
from llama_index.core.storage.docstore.simple_docstore import DocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url
from typing import List
import json

from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent, StructuredPlannerAgent, FunctionCallingAgentWorker, ReActAgentWorker

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def get_db():
    connection_string = os.environ.get("DATABASE_URL")
    url = make_url(connection_string)  # pyright: ignore[]

    vector_store = PGVectorStore.from_params(
        database=url.database,
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

    keyword_index_id = ""
    with open("./index/index_store.json") as f:
        data = json.load(f)
        for key in data["index_store/data"].keys():
            if data["index_store/data"][key]["__type__"] == "keyword_table":
                keyword_index_id = key

    keyword_index = load_index_from_storage(
        storage_context=storage_context, index_id=keyword_index_id
    )
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store
    )
    return (vector_index, keyword_index)


def determine_nodes_context(nodes: List[NodeWithScore], query: str) -> str:
    """
    nodes: List[NodeWithScore]
    query: str

    Function takes in a list of NodeWithScore, and an input query.
    Using the input query, function will extract information that will answer the query from nodes, and collate them into a response.
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
    patient_name: Name of the patient.
    query: Information to search for. Include the patient's name in the query.

    Takes in an input query to search for observations (e.g. blood pressure, glucose levels, etc.). Matches 25 retrieved nodes to patient_name, and returns nodes that contain the patient's name. Include the patient's name in the query.
    Will call determine_node_context, and returns the response of the function.
    """

    vector_retriever = VECTOR_INDEX.as_retriever(
        vector_store_query_mode="hybrid", sparse_top_k="25", verbose=True
    )

    nodes: List[NodeWithScore] = vector_retriever.retrieve(query)

    filtered_nodes = list(filter(lambda node: patient_name in node.text, nodes))

    context_nodes = determine_nodes_context(filtered_nodes, query)

    return context_nodes


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


def search_for_patients_with_specified_condition(query: str, condition: str) -> str:
    """
    Takes in an input string of specific condition name to look up. 
    """
    keyword_retriever = KEYWORD_INDEX.as_retriever(
        verbose=True, retriever_mode="simple"
    )
    nodes = keyword_retriever.retrieve(condition)

    print(nodes)

    filtered_nodes = list(filter(lambda node: condition in node.text, nodes))

    response_synth = get_response_synthesizer()

    response = response_synth.synthesize(query_str=query, nodes=filtered_nodes)

    return str(response)



if __name__ == "__main__":
    # Set to local llm
    Settings.llm = Ollama(
        model="qwq",
        request_timeout=3600,
        context_window=32000,
        base_url=os.environ.get("OLLAMA_URL"),  # pyright: ignore[]
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="./bge-base-en-v1.5",
    )

    VECTOR_INDEX, KEYWORD_INDEX = get_db()


    search_for_patients_with_condition_tool = FunctionTool.from_defaults(
        fn=search_for_patients_with_specified_condition
    )
    search_for_patient_observations_tool = FunctionTool.from_defaults(
        fn=search_for_patient_observations
    )

    determine_node_context_tool = FunctionTool.from_defaults(fn=determine_nodes_context)

    vector_retriever = VectorIndexRetriever(index=VECTOR_INDEX, vector_store_query_mode="hybrid", sparse_top_k=10)
    keyword_retriever = KeywordTableSimpleRetriever(index=KEYWORD_INDEX)




    custom_retriever = CustomRetriever(vector_retriever=vector_retriever, keyword_retriever=keyword_retriever, mode="OR")

    response_synth = get_response_synthesizer()

    custom_query_engine = RetrieverQueryEngine(retriever=custom_retriever, response_synthesizer=response_synth)
    condition_query_engine = RetrieverQueryEngine(retriever=keyword_retriever, response_synthesizer=response_synth)

    retrieve_tool = QueryEngineTool(query_engine=custom_query_engine, 
                               metadata= ToolMetadata(
                                         name="retrieve_patient_information",
                                         description="""A tool for running keyword and semantic search for information about patients.
                                         Only contains patient information on a local database."""))

    search_condition_tool = QueryEngineTool(query_engine=condition_query_engine,
                                            metadata = ToolMetadata(
                                                name ="search_medical_condition",
                                                description="""A tool to search for patients with the specified medical condition."""
                                            ))
    

    query_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=[retrieve_tool, search_condition_tool])
    search_tool = QueryEngineTool(query_engine=query_engine,
                                  metadata=ToolMetadata(
                                      name="complex_query_search",
                                      description="""A tool to break down queries in sub-queries and solve step-by-step. 
                                      Has access to other query engines that contain patient information."""
                                  ))

    agent = ReActAgent.from_tools(tools=[search_condition_tool, retrieve_tool], verbose =True)
    query = "Which patients have hypertension? What do their blood pressure readings look like?"
    response = agent.chat(query)

    # print(str(response))
