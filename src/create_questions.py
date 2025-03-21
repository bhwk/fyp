from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import os
import json
import asyncio
import aiofiles
import pathlib
import time
from collections import deque
from llama_index.llms.ollama import Ollama

from fhir_select_1 import FLAT_FILE_PATH

dir_path = pathlib.Path(FLAT_FILE_PATH)

llm = Ollama(
    model="qwen2.5:32b",
    base_url=os.environ.get("OLLAMA_URL"),  # type: ignore
    request_timeout=1000,
    context_window=8000,
)


class Questions(BaseModel):
    questions: list[str] = Field(
        description="The question generated from the file content"
    )


async def load_file(file: pathlib.Path):
    async with aiofiles.open(file, mode="r") as f:
        content = await f.read()
        return str(content)


async def process_file(file: pathlib.Path):
    prompt = PromptTemplate(
        "Generate a single question about the following text. Avoid general queries such as marriage status/death. Text: {text}"
    )
    content = await load_file(file)
    response = await llm.astructured_predict(
        Questions,
        prompt=prompt,
        text=content,
    )
    json_output = response.model_dump_json()
    output = json.loads(json_output)

    obj = {"file": str(file), "questions": output["questions"]}
    return obj


async def process_batch(batch):
    tasks = [process_file(file) for file in batch if file.is_file()]
    return await asyncio.gather(*tasks)


async def load_and_process_files(dir_path: pathlib.Path, batch_size=500):
    filenames = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            filenames.append(pathlib.Path(os.path.join(root, file)))
    results = []
    start_time = time.time()

    file_queue = deque(filenames)

    while file_queue:
        batch = [file_queue.popleft() for _ in range(min(batch_size, len(file_queue)))]
        batch_results = await process_batch(batch)
        results.extend(batch_results)

        # Write results in larger batches to reduce I/O overhead
        if len(results) % (batch_size * 2) == 0 or not file_queue:
            with open(f"batch_{len(results)}.json", "w") as fp:
                obj = {"files": results}
                json.dump(obj, fp, indent=4, sort_keys=True)  # Formatted JSON

        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / len(results)) * len(filenames)
        remaining_time = estimated_total_time - elapsed_time
        print(
            f"Processed {len(results)}/{len(filenames)} "
            f"({len(results)/len(filenames)*100:.2f}%) "
            f"| Elapsed: {elapsed_time:.2f}s "
            f"| Remaining: {remaining_time:.2f}s "
        )

    return results


async def main():
    results = await load_and_process_files(dir_path)

    obj = {"files": results}

    with open("questions.json", "w") as fp:
        json.dump(obj, fp, indent=4, sort_keys=True)  # Formatted JSON


if __name__ == "__main__":
    asyncio.run(main())
