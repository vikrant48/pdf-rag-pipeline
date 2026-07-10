import os
import logging
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

def rerank_results(
    query: str, 
    docs: List[Document], 
    top_n: int = None
) -> List[Tuple[float, Document]]:
    """
    Reranks retrieving candidate Document list using CrossEncoder model.
    Returns the top_n results as a list of (score, doc) tuples sorted descending by score.
    """
    if not docs:
        logger.warning("Empty document list provided for reranking.")
        return []

    if top_n is None:
        top_n = int(os.getenv("RERANK_TOP_N", "5"))

    # Get reranker model from env, or default to ms-marco-MiniLM-L-6-v2
    model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    logger.info(f"Reranking {len(docs)} documents using CrossEncoder '{model_name}'...")
    
    # Initialize the cross-encoder on CPU to keep resource constraints light
    model = CrossEncoder(model_name, device="cpu")
    
    # Format input pairs
    pairs = [[query, doc.page_content] for doc in docs]
    
    # Predict relevance scores
    scores = model.predict(pairs)
    
    # Pair scores with docs and convert numpy floats to native Python floats
    scored_docs = [
        (float(score), doc) 
        for score, doc in zip(scores, docs)
    ]
    
    # Sort by score in descending order
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    logger.info(f"Reranking complete. Returning top {min(top_n, len(scored_docs))} documents.")
    return scored_docs[:top_n]
