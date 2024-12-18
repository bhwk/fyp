import os
import pathlib
from llama_index.llms.ollama import Ollama


llm = Ollama(
    model="llama3",
    request_timeout=1000,
    context_window=5000,
)


def load_text_files_recursive(directory: pathlib.Path):
    for file_path in directory.rglob("*.txt"):
        text = ""
        print(f"Generating summary for {file_path.stem}")
        with file_path.open("r", encoding="utf-8") as f:
            text = f.read()

        response = llm.complete(
            f"Generate a summary that includes ALL information from the following:\n{text}\n"
        )

        if not os.path.exists(f"./temp/summary/{file_path.parent.stem}"):
            os.makedirs(f"./temp/summary/{file_path.parent.stem}")
        with open(
            f"./temp/summary/{file_path.parent.stem}/{file_path.stem}.txt", "w"
        ) as f:
            f.write(str(response))


load_text_files_recursive(pathlib.Path("./temp/flat/"))
