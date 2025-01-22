import os
from llama_index.core.tools import ToolMetadata, QueryEngineTool, FunctionTool
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent, ReActAgent
from llama_index.core.retrievers import VectorIndexRetriever, KeywordTableSimpleRetriever
from llama_index.core import PromptTemplate
from llama_index.core.workflow import Context
from llama_index.core.response_synthesizers import get_response_synthesizer
from query_db_postgres import get_db

llm = Ollama(
        model="qwen2.5:32b",
        request_timeout=3600,
        context_window=32000,
        base_url=os.environ.get("OLLAMA_URL"),  # pyright: ignore[]
        temperature=0.7,
        additional_kwargs={"top_k": 20, "top_p": 0.8, "min_p": 0.05}
    )

VECTOR_INDEX, KEYWORD_INDEX = get_db()

readings_query_engine = VECTOR_INDEX.as_query_engine()
condition_query_engine = KEYWORD_INDEX.as_query_engine()

vector_retriever = VectorIndexRetriever(index=VECTOR_INDEX, vector_store_query_mode="hybrid", sparse_top_k=5)
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

# async def retrieve_medical_readings_for_patient(ctx: Context, query: str) -> str:
#     """Useful for running semantic search for information related to a patient.
#     Specify patient's name in the form ([information to search] for [patient name])."""
#     response = await readings_query_engine.aquery(query)
#     return(str(response))

# async def search_for_patients_with_medical_condition(ctx: Context, query: str) -> str:
#     """Useful for searching for patients with the specified medical condition"""
#     response = await condition_query_engine.aquery(query)
#     return(str(response))


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


async def record_information(ctx: Context, information: str) -> str:
    """Useful for recording information received."""

    current_state = await ctx.get("state")
    if "information" not in current_state:
        current_state["information"] = []
    current_state["information"].append(information)
    await ctx.set("state", current_state)
    return "information recorded"

async def write_response(ctx: Context, response_content: str) -> str:
    """Useful for writing a response to a query."""
    current_state = await ctx.get("state")
    current_state["response_content"] = response_content
    await ctx.set("state", current_state)
    return "Response written"

async def review_response(ctx: Context, review: str) -> str:
    """Useful for reviewing a response and providing feedback."""
    current_state = await ctx.get("state")
    current_state["review"] = review
    await ctx.set("state", current_state)
    return "Response reviewed"


search_agent = ReActAgent(
    name="SearchAgent",
    description="Searches for information related to a patient on a local database.",  
    tools=[retrieve_tool, search_condition_tool],
    llm=llm,
    system_prompt=(
        "You are the SearchAgent that can search the local database for information related to a patient."
        "Break down the search into simpler steps that you can execute."
        "Once the information is recorded and you are satisfied, you should hand off control to the ReviewAgent for reviewing."
    ),
    can_handoff_to=["ReviewAgent"]
    
)

review_agent = ReActAgent(
    name="ReviewAgent",
    description="Useful for reviewing a response and providing feedback.",
    system_prompt=(
        "You are the ReviewAgent that can review a response and provide feedback."
        "The information contained in the response should be anonymized. Names of patients should be replaced with placeholders."
        "Your feedback should either approve the current response or request changes for the GenerateAgent to implement."
    ),
    tools = [FunctionTool.from_defaults(review_response)],
    can_handoff_to=["GenerateAgent"]
)
                                       
generate_agent = ReActAgent(
    name="GenerateAgent",
    description="Generates an appropriate response to the user's query.",
    llm=llm,
    system_prompt=(
        "You are the GenerateAgent that can generate a response to the user's query."
        "The content in your response should be grounded in information received."
        "Once your response is written, you must get feedback from the ReviewAgent until your response is satisfactory."
        ),
    tools=[FunctionTool.from_defaults(fn) for fn in [write_response, record_information]],
    can_handoff_to=["SearchAgent", "ReviewAgent"]
    )

async def main():
    from llama_index.core.agent.workflow import (
        AgentInput,
        AgentOutput,
        ToolCall,
        ToolCallResult,
        AgentStream
    )

    workflow = AgentWorkflow(
        agents= [
            generate_agent,
            search_agent,
            #review_agent
            ],
        root_agent= generate_agent.name,
        initial_state={
            "information": [],
            "response_content": "Not written yet.",
            "review": "Review required."
        },
        num_concurrent_runs = 1
    )

    ctx = Context(workflow)

    handler = workflow.run("Show me blood pressure readings for hypertension patients.",ctx=ctx)

    current_agent = None
    current_tool_calls = ""
    async for event in handler.stream_events():
        if (
            hasattr(event, "current_agent_name")
            and event.current_agent_name != current_agent
        ):
            current_agent = event.current_agent_name
            print(f"\n{'='*50}")
            print(f"🤖 Agent: {current_agent}")
            print(f"{'='*50}\n")

        # if isinstance(event, AgentStream):
        #     if event.delta:
        #         print(event.delta, end="", flush=True)
        # elif isinstance(event, AgentInput):
        #     print("📥 Input:", event.input)
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("📤 Output:", event.response.content)
            if event.tool_calls:
                print(
                    "🛠️  Planning to use tools:",
                    [call.tool_name for call in event.tool_calls],
                )
        elif isinstance(event, ToolCallResult):
            print(f"🔧 Tool Result ({event.tool_name}):")
            print(f"  Arguments: {event.tool_kwargs}")
            print(f"  Output: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"🔨 Calling Tool: {event.tool_name}")
            print(f"  With arguments: {event.tool_kwargs}")
    state = await handler.ctx.get("state")
    print(state["response_content"])

if __name__ == "__main__":
    import nest_asyncio
    import asyncio
    nest_asyncio.apply()
    asyncio.run(main())