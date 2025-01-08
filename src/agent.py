import logging
import sys
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
)
from llama_index.core import PromptTemplate
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url

# Uncomment to see debug logs
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


query_system_header_str = """\

You are an agent designed to send appropriate queries to a RAG system based on the input query provided to you.\
You are to determine from the context information retrieved the necessary information required to answer the query,
and generate a succint summary containing all the information that can answer the query.

## Tools
You have access to a wide variety of tools. You are responsible for using
the tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools
to complete each subtask.

You have access to the following tools:
{tool_desc}

## Output Format
To answer the question, please use the following format.

```
Thought: I need to use a tool to help me answer the question.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
```

Please ALWAYS start with a Thought.

Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

If this format is used, the user will respond in the following format:

```
Observation: tool response
```

You should keep repeating the above format until you have enough information
to answer the question without using any more tools. At that point, you MUST respond
in the one of the following two formats:

```
Thought: I can answer without using any more tools.
Answer: [your answer here]
```

```
Thought: I cannot answer the question with the provided tools.
Answer: Sorry, I cannot answer your query.
```

Determine if your answer can satisfy the query properly. If not, continue repeating the previous instructions.
Your answer should be in the form of a synthetic summary containing all the information necessary to answer the query.


"""
query_system_prompt = PromptTemplate(query_system_header_str)


def get_db():
    connection_string = "postgresql://postgres:password@localhost:5432"
    db_name = "vector_db"
    url = make_url(connection_string)

    vector_store = PGVectorStore.from_params(
        database=db_name,
        host=url.host,
        password=url.password,
        port=url.port,  # type: ignore
        user=url.username,
        table_name="patient_records",
        hybrid_search=True,
        embed_dim=768,
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )

    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-en-v1.5",
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )
    return index


if __name__ == "__main__":
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    # Set to local llm
    # Settings.llm = Ollama(
    #     model="mistral-nemo", request_timeout=3600, context_window=10000
    # )

    Settings.llm = Gemini(
        model="models/gemini-1.5-flash",
        api_key="AIzaSyBoOaUIrtSIemKQROi7IFijhG-2CDN-AIA",
    )

    index = get_db()
    query_engine = index.as_query_engine()
    query_tool = QueryEngineTool.from_defaults(
        query_engine,
        name="patient_records_RAG",
        description=(
            "Provides information about patient readings, medications, conditions, and allergies."
            "Provides information about patient observations and procedures."
            "All files are sorted by date."
            "Use a succinct plain text string as input."
            "Information that can be queried: Conditions, Medications, Observations, Procedures"
        ),
    )

    agent = ReActAgent.from_tools([query_tool], verbose=True)  # type: ignore
    agent.update_prompts({"agent_worker:system_prompt": query_system_prompt})

    response = agent.chat(
        "What are some common risk factors between patients with diabetes?"
    )
    print(response)
