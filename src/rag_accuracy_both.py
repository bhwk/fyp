import random
import argparse
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
    timeout=None
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


async def process_question(file_path, question, k):
    async with semaphore:
        print(f"Starting processing: {question}")
        result = await search_for_information(question, k)

        # Append the current result to the batch
        return {
            "file": str(file_path),
            "query": str(question),
            "information": result.get("information", []),
            "nodes": result.get("nodes", []),
        }


async def search_for_information(
    query: str,
    k: int,
):
    """A tool to search for patient information on a database.
    Only contains patient information on a local database."""
    keyword_result = await rag.run(
        query=query,
        mode="keyword",
        vector_index=VECTOR_INDEX,
        keyword_index=KEYWORD_INDEX,
        llm=llm,
        k=k,
    )
    semantic_result = await rag.run(
        query=query,
        mode="semantic",
        vector_index=VECTOR_INDEX,
        keyword_index=KEYWORD_INDEX,
        llm=llm,
        k=k,
    )

    nodes = []

    nodes.extend(keyword_result["nodes"])
    nodes.extend(semantic_result["nodes"])

    result = {
        "information": keyword_result["information"] + semantic_result["information"],
        "nodes": nodes,
    }
    return result


async def process_batch(batch, k):
    tasks = []
    for file in batch:
        tasks.append(process_question(file["file"], file["questions"][0], k))
    return await asyncio.gather(*tasks)


async def load_and_process_questions(k=3, batch_size=10):
    with open("questions.json") as f:
        obj = json.load(f)

    batch = []
    files = random.sample(obj["files"], k=100)
    file_queue = deque(files)

    results = []
    start_time = time.time()

    while file_queue:
        batch = [file_queue.popleft() for _ in range(min(batch_size, len(file_queue)))]
        batch_results = await process_batch(batch, k)
        results.extend(batch_results)

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


async def main(k):
    print("Starting main process...")

    results = await load_and_process_questions(k=k)
    output_file = f"semantic_keyword_k{k}_{int(time.time())}.json"
    with open(output_file, "w") as fp:
        json.dump(results, fp, indent=4, sort_keys=True)

    print("Processing complete.")


if __name__ == "__main__":
    import nest_asyncio
    import asyncio

    nest_asyncio.apply()

    parser = argparse.ArgumentParser(
        description="Run RAG both accuracy with a specified k value."
    )
    parser.add_argument(
        "k", type=int, help="Top k results to fetch in search_for_information"
    )
    args = parser.parse_args()

    asyncio.run(main(args.k))
