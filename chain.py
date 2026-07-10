import os
import logging
from typing import List, Tuple
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_groq import ChatGroq

from index import IndexBundle
from retrieve import hybrid_search
from rerank import rerank_results
from context import build_context

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class RAGChain:
    """
    Encapsulates a QA generation chain structured with LCEL and featuring 
    an instance-level conversation memory for multi-turn dialogue.
    """

    def __init__(
        self, 
        index_bundle: IndexBundle, 
        model_name: str = None, 
        temperature: float = None,
        k: int = None,
        rrf_k: int = None,
        rerank_top_n: int = None,
        max_tokens: int = None
    ):
        self.index_bundle = index_bundle
        
        # Load hyperparameters from environment or default parameters
        self.model_name = model_name or os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
        temp_env = os.getenv("LLM_TEMPERATURE", "0.2")
        self.temperature = temperature if temperature is not None else float(temp_env)
        
        self.k = k if k is not None else int(os.getenv("RETRIEVAL_K", "5"))
        self.rrf_k = rrf_k if rrf_k is not None else int(os.getenv("RRF_K", "60"))
        self.rerank_top_n = rerank_top_n if rerank_top_n is not None else int(os.getenv("RERANK_TOP_N", "5"))
        self.max_tokens = max_tokens if max_tokens is not None else int(os.getenv("MAX_TOKENS", "2000"))

        # Instance-level chat history memory list of (user_message, assistant_response) tuples
        self.chat_history: List[Tuple[str, str]] = []

        # Build the LLM instance
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key or "your_groq_api_key" in groq_api_key:
            logger.warning("GROQ_API_KEY is not defined or is placeholder. Using FakeListLLM for verification.")
            from langchain_core.language_models.fake import FakeListLLM
            self.llm = FakeListLLM(responses=[
                "This is a fake response about photosynthesis. Green plants use sunlight.",
                "This is a second fake response about photosynthesis."
            ])
        else:
            logger.info(f"Initializing ChatGroq: {self.model_name} (temp={self.temperature})...")
            self.llm = ChatGroq(
                model=self.model_name, 
                temperature=self.temperature,
                groq_api_key=groq_api_key
            )

        # Build prompt template
        self.prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Use the following context and chat history to answer "
            "the user's question. If you do not know the answer, state that you do not know.\n\n"
            "Context:\n{context}\n\n"
            "Chat History:\n{chat_history_str}\n\n"
            "Question: {question}\n"
            "Answer:"
        )

        # Assemble the LCEL chain
        self.chain = self._build_lcel_chain()

    def _build_lcel_chain(self):
        """Assembles the retrieval-augmented generation chain in LCEL."""
        
        # 1. Retrieve candidates via hybrid search
        retrieve_step = RunnablePassthrough.assign(
            docs=RunnableLambda(
                lambda inputs: hybrid_search(
                    query=inputs["question"],
                    index_bundle=self.index_bundle,
                    k=self.k,
                    rrf_k=self.rrf_k
                )
            )
        )

        # 2. Rerank retrieve output candidate docs
        rerank_step = RunnablePassthrough.assign(
            reranked_docs=RunnableLambda(
                lambda inputs: rerank_results(
                    query=inputs["question"],
                    docs=[doc for doc, _ in inputs["docs"]],
                    top_n=self.rerank_top_n
                )
            )
        )

        # 3. Compile context string satisfying token budget guard limits
        context_step = RunnablePassthrough.assign(
            context=RunnableLambda(
                lambda inputs: build_context(
                    reranked_docs=inputs["reranked_docs"],
                    max_tokens=self.max_tokens
                )
            )
        )

        # Pipe components: fetch context -> compile prompt -> query LLM -> parse string output
        lcel_chain = (
            retrieve_step
            | rerank_step
            | context_step
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        return lcel_chain

    def query(self, question: str) -> str:
        """
        Queries the RAG chain on a question. Computes multi-turn chat history context,
        updates the instance history list, and returns the response.
        """
        logger.info(f"Received query: '{question}'")
        
        # 1. Format chat history from memories list to string
        history_lines = []
        for q, a in self.chat_history:
            history_lines.append(f"User: {q}")
            history_lines.append(f"Assistant: {a}")
        chat_history_str = "\n".join(history_lines) if history_lines else "No previous chat history."

        # 2. Prepare inputs for LCEL chain
        inputs = {
            "question": question,
            "chat_history_str": chat_history_str
        }

        # 3. Execute
        response = self.chain.invoke(inputs)

        # 4. Append dialogue session context to chat memory
        self.chat_history.append((question, response))
        logger.info("Dialogue appended to chat history.")

        return response

    def query_with_contexts(self, question: str) -> Tuple[str, List[Document]]:
        """
        Runs the query through the retrieval pipeline, returning both the final answer
        and the list of reranked candidate Document objects that were compiled into
        the final prompt.
        """
        logger.info(f"Received query with contexts: '{question}'")
        
        # Format chat history from memories list to string
        history_lines = []
        for q, a in self.chat_history:
            history_lines.append(f"User: {q}")
            history_lines.append(f"Assistant: {a}")
        chat_history_str = "\n".join(history_lines) if history_lines else "No previous chat history."

        # Compute intermediate retrieval steps manually to extract contexts
        docs_with_scores = hybrid_search(
            query=question,
            index_bundle=self.index_bundle,
            k=self.k,
            rrf_k=self.rrf_k
        )
        
        # Extract direct document elements from hybrid search tuples
        doc_objects = [doc for doc, _ in docs_with_scores]
        
        # Rerank
        reranked_docs_with_scores = rerank_results(
            query=question,
            docs=doc_objects,
            top_n=self.rerank_top_n
        )
        
        # Compile pruned context string based on token budget
        context_str = build_context(
            reranked_docs=reranked_docs_with_scores,
            max_tokens=self.max_tokens
        )
        
        # Invoke the prompt and model with compiled inputs
        prompt_val = self.prompt.format_prompt(
            context=context_str,
            chat_history_str=chat_history_str,
            question=question
        )
        
        response_msg = self.llm.invoke(prompt_val.to_messages())
        
        # Extract content
        if hasattr(response_msg, "content"):
            response = response_msg.content
        else:
            response = str(response_msg)

        # Update chat history
        self.chat_history.append((question, response))
        logger.info("Dialogue appended to chat history in query_with_contexts.")
        
        # Filter final set of documents to include only those that actually fit in context_str
        final_docs = []
        for _, doc in reranked_docs_with_scores:
            if doc.page_content in context_str:
                final_docs.append(doc)
                
        return response, final_docs
