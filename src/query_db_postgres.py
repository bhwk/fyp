import os
from custom_retriever import CustomRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
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
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.agent import ReActAgent
from sqlalchemy import make_url
import json

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


if __name__ == "__main__":
    # Set to local llm
    Settings.llm = Ollama(
        model="qwen2.5:32b",
        request_timeout=3600,
        context_window=32000,
        base_url=os.environ.get("OLLAMA_URL"),  # pyright: ignore[]
        temperature=0.7,
        additional_kwargs={"top_k": 20, "top_p": 0.8, "min_p": 0.05}
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="./bge-base-en-v1.5",
    )

    VECTOR_INDEX, KEYWORD_INDEX = get_db()


    vector_retriever = VectorIndexRetriever(index=VECTOR_INDEX, vector_store_query_mode="hybrid", sparse_top_k=10)
    keyword_retriever = KeywordTableSimpleRetriever(index=KEYWORD_INDEX)

    qa_prompt_template = (
        """Context information is below.
        -------
        {context_str}
        -------
        Given the context information and not prior knowledge,
        answer the query.
        Ensure that your answer is concise and includes the information needed to answer the query.
        Query: {query_str}
        Answer: """
        )
    qa_prompt = PromptTemplate(qa_prompt_template)

    refine_prompt_template = (
        """The original query is as follows: {query_str}
        We have provided an existing answer. {existing_answer}
        We have the opportunity to refine the existing answer (if needed) with more context below.
        ------
        {context_msg}
        ------
        Given the new context, refine the existing answer to better answer the query.
        If the context is not useful, return the original answer.
        Ensure that your answer is concise.
        Refined answer: """
    )

    refine_prompt = PromptTemplate(refine_prompt_template)

    custom_synth = get_response_synthesizer(text_qa_template=qa_prompt, refine_template=refine_prompt)

    readings_query_engine = RetrieverQueryEngine(retriever = vector_retriever, response_synthesizer=custom_synth)
    condition_query_engine = RetrieverQueryEngine(retriever=keyword_retriever, response_synthesizer=custom_synth)

    retrieve_tool = QueryEngineTool(query_engine=readings_query_engine, 
                               metadata= ToolMetadata(
                                         name="retrieve_medical_readings_for_patient",
                                         description="""A tool for running semantic search for information related to a patient.
                                         Only contains patient information on a local database.
                                         Information consists of medical observations.
                                         Necessary to specify the patient's name in the form ([information to search] for [patient name])."""))

    search_condition_tool = QueryEngineTool(query_engine=condition_query_engine,
                                            metadata = ToolMetadata(
                                                name ="search_for_patients_with_medical_condition",
                                                description="""A tool to search for patients with the specified medical condition."""
                                            ))
    
    search_agent = ReActAgent.from_tools(tools=[search_condition_tool, retrieve_tool],
                                         verbose=True, 
                                         context="You are an expert planner that breaks queries down into easy-to-follow\
                                         steps. Think step-by-step on how you will execute your plan.")


    query_engine_tools = [QueryEngineTool(query_engine=search_agent, metadata=ToolMetadata(
        name="search_agent",
        description="Agent used to search for information related to patients."
    ))]
    agent = ReActAgent.from_tools(tools=query_engine_tools, verbose=True, context="You are an expert AI that understands how to make use of your tools effectively. You will check your answers\
                                  and make sure that they satisfy the query at all times.")

    query = input("ENTER QUERY: ")
    response = agent.chat(query)
    print(f"RESPONSE FROM RETRIEVAL/INFERENCE AGENT:\n{str(response)}")

    synth_agent_prompt = f"""
        You are an expert data anonymization and summarization agent. Your task is to identify and replace all Personally Identifiable Information (PII) in the given text and query.
        You do not need to make use of any tools to answer.
        Follow these rules:

        - Anonymization:
            - The patient's name must be removed and replaced with pseudonyms.
            - Replace specific locations (e.g, cities, countries, landmarks) with placeholders.
            - Unless asked in query, replace specific dates with placeholders.
            - Replace phone numbers, email addresses, and postal addresses with "[CONTACT]".
        - Medical Reporting:
            - Replace mentions of medication dosages with "[DOSAGE]".
            - Summarize and round all vitals with appropriate medical context.
            - If there are multiple readings of the same type for a patient, summarize it into a range of values.
            - Replace values with rounded values for lab results and vital signs. Use approximate ranges if values fluctuate.
        - Summarization:
            - Extract the key points from the text and summarize it.
            - When possible, rewrite your answer such that it omits any PII only if it doesn't affect the original meaning of the answer.
        - Query:
            - Based on the anonymized text, alter the provided query such that it can be answered by the anonymized text, while retaining its original meaning.

        Output in the format:
        Generated Query: [GENERATED QUERY]
        Context: [ANONYMIZED TEXT]

        Ensure your answer follows the format provided.
    """

    synthesis_agent = ReActAgent.from_tools(verbose=True, context=synth_agent_prompt)

    synth_response = synthesis_agent.chat(f"""
    Original Query: {query}
    ------
    TEXT:
    {str(response)}""")
    print(f"RESPONSE FROM SYNTHESIS AGENT:\n{synth_response}")

