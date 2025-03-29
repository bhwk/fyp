import random
import time
from collections import deque
import asyncio
import aiofiles
import json
import os
from llama_index.core.agent.workflow import (
    ToolCallResult,
)
from llama_index.core.schema import NodeWithScore
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.workflow import Context
from query_db_postgres import get_db
from rag import RAGWorkflow

rag = RAGWorkflow(
    # verbose=True,
    timeout=300.0
)

llm = Ollama(
    model="qwen2.5:32b",
    request_timeout=3600,
    context_window=16000,
    base_url=os.environ.get("OLLAMA_URL"),  # pyright: ignore
    temperature=0.7,
    additional_kwargs={"top_k": 20, "top_p": 0.8, "min_p": 0.05},
    is_function_calling_model=True,
    # json_mode=True,
)

VECTOR_INDEX, KEYWORD_INDEX = get_db()

JSON_DIR = "single_agent_results"

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
    workflow,
    context,
):
    async with semaphore:
        handler = workflow.run(question, ctx=context)

        print(f"Starting processing: {question}")
        async for event in handler.stream_events():
            if isinstance(event, ToolCallResult):
                print(f"Tool called: {event.tool_name}")

        state = await handler.ctx.get("state")  # type: ignore

        # Append the current result to the batch
        return {
            "file": str(file_path),
            "query": str(question),
            "information": state.get("information", []),
            "nodes": state.get("nodes", []),
        }


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


async def search_for_information(
    query: str,
):
    """A tool to search for patient information on a database.
    Only contains patient information on a local database."""
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
    ),
    tools=[record_information, record_nodes, search_for_information],  # type: ignore
)


async def process_batch(batch):
    tasks = []
    for file in batch:
        workflow = AgentWorkflow(
            agents=[agent],
            root_agent=agent.name,
        )
        context = Context(workflow)
        tasks.append(
            process_question(
                file["file"],
                file["questions"][0],
                workflow,
                context,
            )
        )
    return await asyncio.gather(*tasks)


async def load_and_process_questions(batch_size=10):
    with open("questions.json") as f:
        obj = json.load(f)

    batch = []
    files = random.sample(obj["files"], k=100)
    file_queue = deque(files)

    results = []
    start_time = time.time()

    while file_queue:
        batch = [file_queue.popleft() for _ in range(min(batch_size, len(file_queue)))]
        batch_results = await process_batch(batch)
        results.extend(batch_results)

        # Write results in larger batches to reduce I/O overhead
        if len(results) % 100 == 0 or not file_queue:
            json_file = os.path.join(JSON_DIR, f"batch_{len(results)}.json")
            async with aiofiles.open(json_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(results, indent=4))

        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / len(results)) * len(files)
        remaining_time = estimated_total_time - elapsed_time

        print(
            f"Processed {len(results)}/{len(files)} "
            f"({len(results)/len(files)*100:.2f}%) "
            f"| Elapsed: {elapsed_time:.2f}s "
            f"| Remaining: {remaining_time:.2f}s "
        )
    return results


async def main():
    print("Starting main process...")

    results = await load_and_process_questions()

    with open(os.path.join(JSON_DIR, "results.json"), "w") as fp:
        json.dump(results, fp, indent=4, sort_keys=True)

    print("Processing complete.")


if __name__ == "__main__":
    import nest_asyncio
    import asyncio

    nest_asyncio.apply()
    asyncio.run(main())
