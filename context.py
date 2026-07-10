import os
import logging
import tiktoken
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

def build_context(
    reranked_docs: List[Tuple[float, Document]], 
    max_tokens: int = None, 
    encoding_name: str = "cl100k_base"
) -> str:
    """
    Builds a formatted LLM context string from reranked (score, doc) tuples.
    
    Format per chunk:
    [Source {i+1}: {filename}, page {page}] (Relevance: {score:.4f})
    {chunk_text}
    
    Includes a token-budget guard using tiktoken. If the total context string
    exceeds max_tokens, the lowest-ranked chunks (those at the end of the list)
    are discarded sequentially until it fits within the limit.
    """
    if max_tokens is None:
        max_tokens = int(os.getenv("MAX_TOKENS", "2000"))

    if not reranked_docs:
        logger.warning("Empty reranked document list, returning empty context.")
        return ""

    logger.info(f"Building context for {len(reranked_docs)} chunks with token limit: {max_tokens}")

    # Initialize encoding
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.warning(f"Failed to load encoding '{encoding_name}', falling back to 'cl100k_base': {e}")
        encoding = tiktoken.get_encoding("cl100k_base")

    # Clone the input list so we can modify it (pop from bottom)
    current_docs = list(reranked_docs)

    while current_docs:
        # Build the current context string representation
        blocks = []
        for i, (score, doc) in enumerate(current_docs):
            filename = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", 0)
            
            # Construct chunk text block to build context exactly as requested:
            # [Source {i+1}: {filename}, page {page}] (Relevance: {score:.4f})
            # {chunk_text}
            header = f"[Source {i+1}: {filename}, page {page}] (Relevance: {score:.4f})"
            block = f"{header}\n{doc.page_content}"
            blocks.append(block)
            
        context_str = "\n\n".join(blocks)
        
        # Tokenize and compute context length
        tokens = encoding.encode(context_str)
        num_tokens = len(tokens)
        
        if num_tokens <= max_tokens:
            logger.info(f"Context compiled successfully. Token count: {num_tokens}/{max_tokens}")
            return context_str
            
        # If budget exceeded, discard the lowest-ranked document chunk (always at the end)
        logger.info(f"Current token count ({num_tokens}) exceeds budget ({max_tokens}). Dropping the lowest-ranked chunk...")
        current_docs.pop()

    logger.warning("No documents could fit in the specified token budget.")
    return ""
