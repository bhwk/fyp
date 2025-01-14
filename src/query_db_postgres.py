import os
from custom_retriever import CustomRetriever
from llama_index.core.query_engine import RetrieverQueryEngine, SubQuestionQueryEngine
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




if __name__ == "__main__":
    # Set to local llm
    Settings.llm = Ollama(
        model="qwen2.5:32b",
        request_timeout=3600,
        context_window=32000,
        base_url=os.environ.get("OLLAMA_URL"),  # pyright: ignore[]
        temperature=0.7,
        additional_kwargs={"top_k": 20, "top_p": 0.8}
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="./bge-base-en-v1.5",
    )

    VECTOR_INDEX, KEYWORD_INDEX = get_db()


    vector_retriever = VectorIndexRetriever(index=VECTOR_INDEX, vector_store_query_mode="hybrid", sparse_top_k=10)
    keyword_retriever = KeywordTableSimpleRetriever(index=KEYWORD_INDEX)

    custom_retriever = CustomRetriever(vector_retriever=vector_retriever, keyword_retriever=keyword_retriever, mode="OR")

    response_synth = get_response_synthesizer()

    qa_prompt_template = (
        """Context information is below.
        -------
        {context_str}
        -------
        Given the context information and not prior knowledge,
        answer the query.
        Ensure that your answer is concise, and does not include explanations.
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
                                         name="retrieve_patient_medical_readings",
                                         description="""A tool for running semantic search for information about patients.
                                         Only contains patient information on a local database.
                                         Information consists of medical observations.
                                         Specifying a patient's name is optional. (e.g. glucose readings for [patient])"""))

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
    query = "Which patients have hypertension? What are their blood pressure readings?"
    response = agent.chat(query)

    # print(str(response))
