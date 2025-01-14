from llama_index.core import (
    Settings,
    PromptTemplate,
    get_response_synthesizer,
)
from llama_index.core.schema import NodeWithScore
from typing import List

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