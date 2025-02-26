from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import os
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
        description="A list of questions generated from the file content"
    )


async def load_file(file: pathlib.Path):
    async with aiofiles.open(file, mode="r") as f:
        content = await f.read()
        return str(content)


async def process_file(file: pathlib.Path):
    prompt = PromptTemplate("Generate a series of 3 questions from the following text.")
    # load the content of the file
    # then we pass it to the LLM to generate a series of questions,
    # finally we write to individual files
    content = await load_file(file)
    # TODO: implement LLM generation
    # TODO: also implement writing contents

    response = await llm.astructured_predict(
        Questions,
        prompt=prompt,
        text=content,
    )
    json_output = response.model_dump_json()

    return json_output


async def process_batch(batch):
    tasks = [process_file(file) for file in batch if file.is_file()]
    return await asyncio.gather(*tasks)


async def load_and_process_files(dir_path: pathlib.Path, batch_size=100):
    filenames = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            filenames.append(pathlib.Path(os.path.join(root, file)))
    results = []
    start_time = time.time()

    file_queue = deque(filenames[:1])

    while file_queue:
        batch = [file_queue.popleft() for _ in range(min(batch_size, len(file_queue)))]
        batch_results = await process_batch(batch)
        results.extend(batch_results)

        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / len(results)) * len(filenames)
        remaining_time = estimated_total_time - elapsed_time
        print(
            f"Processed {len(results)}/{len(filenames)} "
            f"({len(results)/len(filenames)*100:.2f}%) "
            f"| Elapsed: {elapsed_time:.2f}s "
            f"| Remaining: {remaining_time:.2f}s "
        )

        await asyncio.sleep(0.01)
    return results


async def main():
    results = await load_and_process_files(dir_path)

    for result in results:
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
