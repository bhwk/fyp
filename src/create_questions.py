import os
import asyncio
import aiofiles
import pathlib
import time
from collections import deque
from llama_index.llms.ollama import Ollama

from fhir_select_1 import FLAT_FILE_PATH

dir_path = pathlib.Path(FLAT_FILE_PATH)


async def load_file(file: pathlib.Path):
    async with aiofiles.open(file, mode="r") as f:
        content = await f.read()
        return str(content)


async def process_file(file: pathlib.Path):
    # load the content of the file
    # then we pass it to the LLM to generate a series of questions,
    # finally we write to individual files
    content = await load_file(file)
    # TODO: implement LLM generation
    # TODO: also implement writing contents


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

    file_queue = deque(filenames)

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


async def main():
    await load_and_process_files(dir_path)


if __name__ == "__main__":
    asyncio.run(main())
