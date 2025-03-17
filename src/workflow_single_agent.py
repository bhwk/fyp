import csv
import aiofiles
import asyncio
import json
import os
from llama_index.core.agent.workflow import (
    AgentInput,
    ToolCallResult,
    AgentStream,
)
from llama_index.core.schema import NodeWithScore
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.workflow import Context
from query_db_postgres import get_db
from rag import RAGWorkflow


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

JSON_DIR = "single_agent_results"
BATCH_SIZE = 10

semaphore = asyncio.Semaphore(10)

os.makedirs(JSON_DIR, exist_ok=True)


async def save_batch(batch, batch_index):
    """Save the current batch to a JSON file."""
    json_file = os.path.join(JSON_DIR, f"batch_{batch_index}.json")
    async with aiofiles.open(json_file, "w", encoding="utf-8") as f:
        await f.write(json.dumps(batch, indent=4))
    print(f"Saved batch {batch_index} to {json_file}")


async def process_question(
    file_path,
    question,
    llm,
    progress,
    total,
    batch,
    batch_index,
    workflow,
    context,
):
    async with semaphore:
        handler = workflow.run(question, ctx=context)

        print(f"Starting processing: {question}")
        async for event in handler.stream_events():
            if isinstance(event, ToolCallResult):
                print(f"Tool called: {event.tool_name} -> {event.tool_output}")

        state = await handler.ctx.get("state")  # type: ignore

        response = await llm.acomplete(
            f"Based only on the retrieved information and the query, answer the query."
            f"\nQuery: {question}.\nInformation: {state.get('synthesized_information', '')}\nAnswer:"
        )
        progress[0] += 1
        print(f"Progress: {progress[0]}/{total} questions completed.")

        # Append the current result to the batch
        batch.append(
            {
                "file": str(file_path),
                "query": str(question),
                "information": state.get("information", []),
                "nodes": state.get("nodes", []),
                "response": str(response),
            }
        )

        # Save the batch to JSON if the batch size is reached
        if len(batch) >= BATCH_SIZE:
            await save_batch(batch, batch_index[0])
            batch.clear()  # Clear the batch after saving
            batch_index[0] += 1  # Increment the batch index

        print(f"Completed processing: {question}")


async def record_information(ctx: Context, information: str) -> str:
    """Useful for recording information for a given query. Your input should be information written in plain text."""
    current_state = await ctx.get("state")
    if "information" not in current_state:
        current_state["information"] = []
    current_state["information"].append(information)
    await ctx.set("state", current_state)

    return "Information recorded."


async def record_nodes(ctx: Context, nodes: list[NodeWithScore]) -> str:
    """Useful for recording the nodes retrieved from a search. Your input should be the list of nodes retrieved"""
    current_state = await ctx.get("state")
    if "nodes" not in current_state:
        current_state["nodes"] = []
    current_state["nodes"].extend(nodes)

    await ctx.set("state", current_state)
    return "Nodes recorded"


async def main():
    print("Starting main process...")
    rag = RAGWorkflow(verbose=True, timeout=120.0)

    async def retrieve_medical_readings_for_patient(
        query: str,
    ):
        """A tool for running semantic search for information related to a patient.
        Only contains patient information on a local database.
        Information consists of medical observations.
        Necessary to specify the patient's name in the form ([information to search] for [patient name])."""

        result = await rag.run(
            query=query,
            mode="semantic",
            vector_index=VECTOR_INDEX,
            keyword_index=KEYWORD_INDEX,
            llm=llm,
        )
        return result

    async def search_for_patients_with_medical_condition(
        query: str,
    ):
        """A tool to search for patients with the specified medical condition."""
        result = await rag.run(
            query=query,
            mode="keyword",
            vector_index=VECTOR_INDEX,
            keyword_index=KEYWORD_INDEX,
            llm=llm,
        )
        return result

    agent = FunctionAgent(
        name="Agent",
        description="General purpose agent for queries",
        llm=llm,
        system_prompt=(
            "You are a general purpose agent."
            "You must make use of all your tools to answer the query."
            "You have the ability to search a patient database for any information you may need."
            "You must record any of the information that you retrieve, alongside the nodes retrieved."
            "Extract the key points from the text and summarize it."
        ),
        tools=[
            record_information,
            record_nodes,
            retrieve_medical_readings_for_patient,
            search_for_patients_with_medical_condition,
        ],  # type: ignore
    )

    with open("questions.json") as f:
        obj = json.load(f)

    total_questions = sum(len(file["questions"]) for file in obj["files"])
    progress = [0]
    batch = []
    batch_index = [0]
    print(f"Total questions to process: {total_questions}")
    tasks = []
    for file in obj["files"]:
        for i in range(0, len(file["questions"]), BATCH_SIZE):
            batch_questions = file["questions"][i : i + BATCH_SIZE]
            for q in batch_questions:
                workflow = AgentWorkflow(
                    agents=[agent],
                    root_agent=agent.name,
                )
                context = Context(workflow)
                tasks.append(
                    process_question(
                        file["file"],
                        q,
                        llm,
                        progress,
                        total_questions,
                        batch,
                        batch_index,
                        workflow,
                        context,
                    )
                )

    await asyncio.gather(*tasks)

    if batch:
        await save_batch(batch, batch_index[0])
    print("Processing complete.")


if __name__ == "__main__":
    import nest_asyncio
    import asyncio

    nest_asyncio.apply()
    asyncio.run(main())
