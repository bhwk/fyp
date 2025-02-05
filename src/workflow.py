import os
import sys
from llama_index.core.tools import ToolMetadata, QueryEngineTool, FunctionTool
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent, ReActAgent
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)
from llama_index.core import PromptTemplate
from llama_index.core.workflow import Context
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.openai_like import OpenAILike
from query_db_postgres import get_db

import logging
import logging.config

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

llm = Ollama(
    model="qwen2.5:32b",
    request_timeout=3600,
    context_window=32000,
    base_url=os.environ.get("OLLAMA_URL"),  # pyright: ignore
    temperature=0.7,
    additional_kwargs={"top_k": 20, "top_p": 0.8, "min_p": 0.05},
    is_function_calling_model=True,
    # json_mode=True,
)

VECTOR_INDEX, KEYWORD_INDEX = get_db()

readings_query_engine = VECTOR_INDEX.as_query_engine()
condition_query_engine = KEYWORD_INDEX.as_query_engine()

vector_retriever = VectorIndexRetriever(
    index=VECTOR_INDEX,
    vector_store_query_mode="hybrid",  # type: ignore
    sparse_top_k=5,
)
keyword_retriever = KeywordTableSimpleRetriever(index=KEYWORD_INDEX)  # type: ignore

qa_prompt_template = """Context information is below.
    -------
    {context_str}
    -------
    Given the context information and not prior knowledge,
    answer the query.
    Ensure that your answer is concise and includes the information needed to answer the query.
    Query: {query_str}
    Answer: """
qa_prompt = PromptTemplate(qa_prompt_template)

refine_prompt_template = """The original query is as follows: {query_str}
    We have provided an existing answer. {existing_answer}
    We have the opportunity to refine the existing answer (if needed) with more context below.
    ------
    {context_msg}
    ------
    Given the new context, refine the existing answer to better answer the query.
    If the context is not useful, return the original answer.
    Ensure that your answer is concise.
    Refined answer: """

refine_prompt = PromptTemplate(refine_prompt_template)

custom_synth = get_response_synthesizer(
    text_qa_template=qa_prompt, refine_template=refine_prompt
)

readings_query_engine = RetrieverQueryEngine(
    retriever=vector_retriever, response_synthesizer=custom_synth
)
condition_query_engine = RetrieverQueryEngine(
    retriever=keyword_retriever, response_synthesizer=custom_synth
)


retrieve_tool = QueryEngineTool(
    query_engine=readings_query_engine,
    metadata=ToolMetadata(
        name="retrieve_medical_readings_for_patient",
        description="""A tool for running semantic search for information related to a patient.
                                        Only contains patient information on a local database.
                                        Information consists of medical observations.
                                        Necessary to specify the patient's name in the form ([information to search] for [patient name]).""",
    ),
)

search_condition_tool = QueryEngineTool(
    query_engine=condition_query_engine,
    metadata=ToolMetadata(
        name="search_for_patients_with_medical_condition",
        description="""A tool to search for patients with the specified medical condition.""",
    ),
)


async def main():
    from llama_index.core.agent.workflow import (
        AgentInput,
        AgentOutput,
        ToolCall,
        ToolCallResult,
        AgentStream,
    )

    async def record_information(ctx: Context, information: str) -> str:
        """Useful for recording information for a given query. Your input should be information written in plain text."""
        current_state = await ctx.get("state")
        if "information" not in current_state:
            current_state["information"] = []
        current_state["information"].append(information)
        await ctx.set("state", current_state)

        return "Information recorded."

    record_information_tool = FunctionTool.from_defaults(
        async_fn=record_information, description="Record information"
    )
    search_agent = ReActAgent(
        name="SearchAgent",
        description="Searches and records information of a patient from a local database.",
        llm=llm,
        system_prompt=(
            "You are the SearchAgent that can search a local database for information about patients and record it"
            "Identify and carry out the necessary steps to retrieve the right information needed."
            "You must make use of the tools assigned to you."
            "Record the information you receive using the record_information_tool."
            "You must handoff to the WriteAgent."
        ),
        tools=[record_information_tool, retrieve_tool, search_condition_tool],
        can_handoff_to=["WriteAgent"],
    )

    async def write_response(ctx: Context, response_content: str) -> str:
        """Useful for writing a response to a query. Your input should be concise and in markdown format."""
        current_state = ctx.get("state")
        current_state["response_content"] = response_content  # type: ignore
        await ctx.set("state", current_state)

        return "Response written"

    write_response_tool = FunctionTool.from_defaults(
        async_fn=write_response, description="Write a response"
    )

    write_agent = ReActAgent(
        name="WriteAgent",
        description="Useful for writing a response.",
        system_prompt=(
            "You are the WriteAgent that can write a response to a given query."
            "Generate a concise response with the write_response_tool using the recorded information stored."
        ),
        llm=llm,
        can_handoff_to=["SearchAgent"],
        tools=[write_response_tool],
    )

    workflow = AgentWorkflow(
        agents=[write_agent, search_agent],
        root_agent=write_agent.name,
    )

    handler = workflow.run("Show me blood pressure readings for hypertension patients.")

    async for event in handler.stream_events():
        if isinstance(event, AgentInput):
            print(f"Agent {event.current_agent_name}: ")
        if isinstance(event, AgentStream):
            print(f"{event.delta}", end="")
        elif isinstance(event, ToolCallResult):
            print(f"Tool called: {event.tool_name} -> {event.tool_output}")

    state = await handler.ctx.get("state")  # type: ignore
    print(state)


if __name__ == "__main__":
    import nest_asyncio
    import asyncio

    nest_asyncio.apply()
    asyncio.run(main())
