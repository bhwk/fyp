from llama_index.core import VectorStoreIndex, StorageContext, PromptTemplate
from llama_index.core.response_synthesizers import CompactAndRefine
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
    async def initialise(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
        llm = ev.get("llm")
        if llm is None:
            print("Pass in LLM")
            return None

        vector_index = ev.get("vector_index")
        keyword_index = ev.get("keyword_index")

        if vector_index or keyword_index is None:
            print("Pass keyword and vector indexes to workflow")
            return None

        # Set the indexes in global context
        await ctx.set("vector_index", vector_index)
        await ctx.set("keyword_index", keyword_index)
        await ctx.set("llm", llm)

        return StopEvent()

    @step
    async def query(
        self, ctx: Context, ev: StartEvent
    ) -> SemanticSearchEvent | KeywordSearchEvent | None:
        query = ev.get("query")
        mode = ev.get("mode")

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

        if not query:
            return None
        await ctx.set("query", query)

        vector_index = await ctx.get("vector_index")
        if not vector_index:
            print("vector_index not loaded")
        print(f"Query vector index with: {query}")

        vector_retriever = VectorIndexRetriever(
            index=vector_index,
            vector_store_query_mode="hybrid",  # type: ignore
            sparse_top_k=5,
        )

        nodes = vector_retriever.aretrieve(query)
        print(f"Retrieved {len(nodes)} nodes")

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
        print(f"Query keyword index with {query}")

        keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)
        nodes = keyword_retriever.aretrieve(query)
        print(f"Retrieved {len(nodes)} nodes")

        return RetrieverEvent(nodes=nodes)

    @step
    async def synthesize(self, ctx: Context, ev: RetrieverEvent) -> StopEvent:
        """Return only the relevant context from retrieved nodes"""

        # Custom prompt for response synthesizer
        qa_prompt_template = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "Retrieve only the information necessary to answer the query.\n"
            "Your answer should only contain the necessary context information."
            "Query: {query_str}\n"
            "Answer: "
        )
        qa_prompt = PromptTemplate(qa_prompt_template)

        refine_prompt_template = """The original query is as follows: {query_str}
            We have provided an existing answer. {existing_answer}
            We have the opportunity to refine the existing answer (if needed) with more context below.
            ------
            {context_msg}
            ------
            Given the new context, refine the existing answer to include better context information that will help answer the query.
            If the context is not useful, return the original answer.
            Ensure that your answer is concise, and only contains the necessary context information.
            Refined answer: """

        refine_prompt = PromptTemplate(refine_prompt_template)

        # get llm from global context
        llm: Ollama = await ctx.get("llm")

        query = await ctx.get("query", default=None)

        synthesizer = CompactAndRefine(
            llm=llm,
            verbose=True,
            text_qa_template=qa_prompt,
            refine_template=refine_prompt,
        )

        response = await synthesizer.asynthesize(query, ev.nodes)

        return StopEvent(result=response)
