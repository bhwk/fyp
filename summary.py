import os
import pathlib
from llama_index.llms.ollama import Ollama


llm = Ollama(
    model="phi3.5",
    request_timeout=1000,
    context_window=10000,
    additional_kwargs={"top_k": 10},
)


def load_text_files_recursive(directory: pathlib.Path):
    for file_path in directory.rglob("*.txt"):
        text = ""
        print(f"Generating summary for {file_path.stem}")
        with file_path.open("r", encoding="utf-8") as f:
            text = f.read()

        response = llm.complete(f"Generate a summary of the following:\n{text}")

        with file_path.open("w", encoding="utf-8") as w:
            w.write(str(response))


load_text_files_recursive(pathlib.Path("./temp/flat/"))
