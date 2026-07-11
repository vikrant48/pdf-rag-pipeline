import os
import time
import logging
import json
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

def retry_on_rate_limit(max_attempts=5, initial_wait=2.0, backoff_factor=2.0):
    """Exponential backoff decorator to retry on Groq rate limits (HTTP 429)."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            wait_time = initial_wait
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    err_msg = str(e).lower()
                    is_rate_limit = "rate limit" in err_msg or "429" in err_msg or "too many requests" in err_msg
                    if is_rate_limit and attempt < max_attempts:
                        logger.warning(
                            f"Groq API rate limit hit (attempt {attempt}/{max_attempts}). "
                            f"Retrying in {wait_time:.1f}s... Error: {e}"
                        )
                        time.sleep(wait_time)
                        wait_time *= backoff_factor
                    else:
                        raise e
        return wrapper
    return decorator

class RetryingChatGroq(ChatGroq):
    """Subclass of ChatGroq that wraps invoke and stream calls with rate-limit retry protection."""
    
    def invoke(self, *args, **kwargs):
        decorator = retry_on_rate_limit(max_attempts=5, initial_wait=2.0)
        return decorator(super().invoke)(*args, **kwargs)

    def stream(self, *args, **kwargs):
        decorator = retry_on_rate_limit(max_attempts=5, initial_wait=2.0)
        return decorator(super().stream)(*args, **kwargs)

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

        # JSON file to load/save chat history registry
        self.history_path = os.path.join("data", "chat_history.json")

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
            logger.info(f"Initializing RetryingChatGroq: {self.model_name} (temp={self.temperature})...")
            self.llm = RetryingChatGroq(
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
                    rrf_k=self.rrf_k,
                    metadata_filter=inputs.get("metadata_filter")
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

    def get_session_history(self, session_id: str) -> List[Tuple[str, str]]:
        """Returns the dialogue history list for a session_id from the database."""
        from db import db_get_chat_history
        return db_get_chat_history(session_id)

    def _append_session_dialogue(self, session_id: str, question: str, response: str):
        """Appends a new query-response dialogue pair to the database."""
        from db import db_save_chat
        db_save_chat(session_id, question, response)

    @property
    def chat_history(self) -> List[Tuple[str, str]]:
        """Legacy property mapping to 'default' session history. Reads/writes from DB."""
        return self.get_session_history("default")

    @chat_history.setter
    def chat_history(self, val):
        # Allow clearing 'default' for legacy compatibility
        if not val:
            from db import get_connection, is_postgres_conn
            try:
                conn = get_connection()
                cursor = conn.cursor()
                is_pg = is_postgres_conn(conn)
                if is_pg:
                    cursor.execute("DELETE FROM chat_history_logs WHERE session_id = %s;", ("default",))
                else:
                    cursor.execute("DELETE FROM chat_history_logs WHERE session_id = ?;", ("default",))
                conn.commit()
                cursor.close()
                conn.close()
            except Exception as e:
                logger.error(f"Failed to clear default session history: {e}")

    def query(self, question: str, metadata_filter: dict = None, session_id: str = "default") -> str:
        """
        Queries the RAG chain on a question. Computes multi-turn chat history context,
        updates the persistent session history, and returns the response.
        """
        logger.info(f"Received query: '{question}' for session: '{session_id}' with filter: {metadata_filter}")
        
        # 1. Format chat history from memories list to string
        session_history = self.get_session_history(session_id)
        history_lines = []
        for q, a in session_history:
            history_lines.append(f"User: {q}")
            history_lines.append(f"Assistant: {a}")
        chat_history_str = "\n".join(history_lines) if history_lines else "No previous chat history."

        # 2. Prepare inputs for LCEL chain
        inputs = {
            "question": question,
            "chat_history_str": chat_history_str,
            "metadata_filter": metadata_filter
        }

        # 3. Execute
        response = self.chain.invoke(inputs)

        # 4. Append dialogue session context to chat memory
        self._append_session_dialogue(session_id, question, response)

        return response

    def query_with_contexts(self, question: str, metadata_filter: dict = None, session_id: str = "default") -> Tuple[str, List[Document]]:
        """
        Runs the query through the retrieval pipeline, returning both the final answer
        and the list of reranked candidate Document objects that were compiled into
        the final prompt.
        """
        logger.info(f"Received query with contexts: '{question}' for session: '{session_id}' with filter: {metadata_filter}")
        
        # Format chat history from memories list to string
        session_history = self.get_session_history(session_id)
        history_lines = []
        for q, a in session_history:
            history_lines.append(f"User: {q}")
            history_lines.append(f"Assistant: {a}")
        chat_history_str = "\n".join(history_lines) if history_lines else "No previous chat history."

        # Compute intermediate retrieval steps manually to extract contexts
        docs_with_scores = hybrid_search(
            query=question,
            index_bundle=self.index_bundle,
            k=self.k,
            rrf_k=self.rrf_k,
            metadata_filter=metadata_filter
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
        
        # Invoke response through LLM
        response_msg = self.llm.invoke(prompt_val.to_messages())
        
        # Extract content
        if hasattr(response_msg, "content"):
            response = response_msg.content
        else:
            response = str(response_msg)

        # Update chat history
        self._append_session_dialogue(session_id, question, response)
        
        # Filter final set of documents to include only those that actually fit in context_str
        final_docs = []
        for _, doc in reranked_docs_with_scores:
            if doc.page_content in context_str:
                final_docs.append(doc)
                
        return response, final_docs

    def query_stream(self, question: str, metadata_filter: dict = None, session_id: str = "default"):
        """
        Queries the RAG chain on a question and yields chunks of the response in real time.
        Appends the consolidated response to the dialog session memory.
        """
        logger.info(f"Received query stream: '{question}' for session: '{session_id}' with filter: {metadata_filter}")
        
        # Format chat history from memories list to string
        session_history = self.get_session_history(session_id)
        history_lines = []
        for q, a in session_history:
            history_lines.append(f"User: {q}")
            history_lines.append(f"Assistant: {a}")
        chat_history_str = "\n".join(history_lines) if history_lines else "No previous chat history."

        inputs = {
            "question": question,
            "chat_history_str": chat_history_str,
            "metadata_filter": metadata_filter
        }

        full_response_parts = []
        for chunk in self.chain.stream(inputs):
            full_response_parts.append(chunk)
            yield chunk

        consolidated = "".join(full_response_parts)
        self._append_session_dialogue(session_id, question, consolidated)
        logger.info("Dialogue appended to chat history after streaming completed.")
