import asyncio
import os
from query_db_postgres import get_db
from rag import RAGWorkflow
from llama_index.llms.ollama import Ollama

rag = RAGWorkflow(verbose=True)

llm = Ollama(
    model="qwen2.5:32b",
    request_timeout=3600,
    context_window=20000,
    base_url=os.environ.get("OLLAMA_URL"),  # pyright: ignore
    temperature=0.7,
    additional_kwargs={"top_k": 20, "top_p": 0.8, "min_p": 0.05},
    is_function_calling_model=True,
    # json_mode=True,
)


async def main():
    vector_index, keyword_index = get_db()
    w = RAGWorkflow(verbose=True, timeout=120.0)
    result = await w.run(
        llm=llm,
        query="hypertension",
        vector_index=vector_index,
        keyword_index=keyword_index,
        mode="semantic",
    )
    print(result)


asyncio.run(main())
