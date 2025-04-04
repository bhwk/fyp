from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    PromptTemplate,
    get_response_synthesizer,
)
from llama_index.core.response_synthesizers import CompactAndRefine, ResponseMode
from llama_index.core.schema import NodeWithScore
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.storage.docstore.simple_docstore import DocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.workflow import (
    Event,
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)


class RetrieverEvent(Event):
    """Result of running retrieval"""

    nodes: list[NodeWithScore]


class SemanticSearchEvent(Event):
    """Event representing Semantic search"""

    query: str


class KeywordSearchEvent(Event):
    """Event representing Semantic search"""

    query: str


class RAGWorkflow(Workflow):
    @step
    async def query(
        self, ctx: Context, ev: StartEvent
    ) -> SemanticSearchEvent | KeywordSearchEvent | None:
        if ev.get("k") is not None:
            k = ev.get("k")
        else:
            k = 3

        llm = ev.get("llm")
        query = ev.get("query")
        mode = ev.get("mode")
        vector_index = ev.get("vector_index")
        keyword_index = ev.get("keyword_index")
        if llm is None:
            print("Pass in LLM")
            return None
        if keyword_index is None:
            print("Pass keyword index to workflow")
            return None
        if vector_index is None:
            print("Pass in vector index")
            return None
        # Set the indexes in global context
        await ctx.set("vector_index", vector_index)
        await ctx.set("keyword_index", keyword_index)
        await ctx.set("llm", llm)
        await ctx.set("k", k)

        if mode is None:
            print("Pass in retrieval mode")
            return None

        if mode == "semantic":
            return SemanticSearchEvent(query=query)

        if mode == "keyword":
            return KeywordSearchEvent(query=query)

    @step
    async def semantic_retrieve(
        self,
        ctx: Context,
        ev: SemanticSearchEvent,
    ) -> RetrieverEvent | None:
        query = ev.query
        k = await ctx.get("k")

        if not query:
            return None
        await ctx.set("query", query)

        vector_index = await ctx.get("vector_index")
        if not vector_index:
            print("vector_index not loaded")
        # print(f"Query vector index with: {query}")

        vector_retriever = VectorIndexRetriever(
            index=vector_index,
            vector_store_query_mode="hybrid",  # type: ignore
            sparse_top_k=k,
        )

        nodes = await vector_retriever.aretrieve(query)
        # print(f"Retrieved {len(nodes)} nodes")

        return RetrieverEvent(nodes=nodes)

    @step
    async def keyword_retrieve(
        self,
        ctx: Context,
        ev: KeywordSearchEvent,
    ) -> RetrieverEvent | None:
        query = ev.query

        if not query:
            return None
        await ctx.set("query", query)

        keyword_index = await ctx.get("keyword_index")
        if not keyword_index:
            print("keyword_index not loaded")
        # print(f"Query keyword index with {query}")

        keyword_retriever = KeywordTableSimpleRetriever(
            index=keyword_index, num_chunks_per_query=10
        )
        nodes = await keyword_retriever.aretrieve(
            query,
        )
        # print(f"Retrieved {len(nodes)} nodes")

        return RetrieverEvent(nodes=nodes)

    @step
    async def synthesize(self, ctx: Context, ev: RetrieverEvent) -> StopEvent:
        """Return only the relevant context from retrieved nodes"""

        qa_prompt_template = """Context information is below.
            -------
            {context_str}
            -------
            Given the context information and not prior knowledge,
            evaluate the context information and extract the necessary information to answer the query.
            Ensure that your answer presents the information in a format similar to the original, and that it only contains the relevant information for the query.
            Query: {query_str}
            Answer: """
        qa_prompt = PromptTemplate(qa_prompt_template)

        # get llm from global context
        llm: Ollama = await ctx.get("llm")

        query = await ctx.get("query", default=None)

        synthesizer = get_response_synthesizer(
            llm=llm,
            use_async=True,
            text_qa_template=qa_prompt,
            response_mode=ResponseMode.COMPACT,
        )

        # synthesizer = CompactAndRefine(
        #     llm=llm,
        #     verbose=True,
        #     text_qa_template=qa_prompt,
        #     refine_template=refine_prompt,
        # )

        response = await synthesizer.asynthesize(query, ev.nodes)

        result = {
            "information": str(response),
            "nodes": [node.metadata["file_name"] for node in ev.nodes],
        }

        return StopEvent(result=result)
