import os
import numpy as np
import logging
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_core.documents import Document
from index import IndexBundle

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

def _get_chunk_key(doc: Document) -> Tuple[str, int, str]:
    """Helper to generate a stable, hashable identifier for a document chunk."""
    return (
        str(doc.metadata.get("doc_id", "unknown")),
        int(doc.metadata.get("page", 0)),
        doc.page_content
    )

def hybrid_search(
    query: str, 
    index_bundle: IndexBundle, 
    k: int = None, 
    rrf_k: int = None
) -> List[Tuple[Document, float]]:
    """
    Performs hybrid search combining BM25 (sparse) and Chroma (dense) retrieval
    using Reciprocal Rank Fusion (RRF).

    Reciprocal Rank Fusion (RRF) Formula:
        RRF_Score(d) = sum_{m in M} ( 1 / (rrf_k + r_m(d)) )
        
        where:
        - M is the set of retrievers (in our case, BM25 and Chroma).
        - r_m(d) is the 1-based rank position of document d in retriever m's output.
        - rrf_k is a constant parameter (default 60) that prevents high-ranking 
          documents from completely dominating the final score.

    Why Rank Fusion is used instead of combining raw scores directly:
        1. Scale Incompatibility: BM25 scores are unbounded positive logits representing 
           lexical term relevance, while Chroma vector search similarity scores are 
           distance metrics (e.g. cosine distance [0, 2] or L2 distance). They reside on 
           completely different scales, boundaries, and probability distributions.
        2. Calibration Sensitivity: Combining raw scores requires normalization rules 
           or scalar weights (e.g. w1*BM25_score + w2*Vector_score). Choosing these weights 
           is highly sensitive, text-domain dependent, and shifts easily when documents are 
           added or model weights update.
        3. Distribution Independency: RRF relies solely on relative rank sequences. Any 
           retriever and model rank distribution can be fused directly without scaling or 
           calibration, yielding a scoring logic that is extremely robust and generalizable.

    Parameters:
        query: The raw search string.
        index_bundle: The IndexBundle containing BM25, Chroma, and source chunks.
        k: The number of final fused documents to return.
        rrf_k: Reciprocal Rank Fusion constant parameter (defaults to 60).

    Returns:
        A list of (Document, float) tuples containing the top-k fused results
        sorted in descending order of their reciprocal rank fusion score.
    """
    if not index_bundle.chunks:
        logger.warning("Empty chunk list in IndexBundle, returning empty results.")
        return []

    if k is None:
        k = int(os.getenv("RETRIEVAL_K", "5"))
    if rrf_k is None:
        rrf_k = int(os.getenv("RRF_K", "60"))

    logger.info(f"Initiating hybrid search for query: '{query}' (k={k}, rrf_k={rrf_k})")

    # Build document identity maps to link back to the exact pristine chunk instances
    key_to_doc = {_get_chunk_key(doc): doc for doc in index_bundle.chunks}

    # --- 1. Sparse Ranker (BM25Okapi) ---
    tokenized_query = query.lower().split()
    bm25_scores = index_bundle.bm25.get_scores(tokenized_query)
    
    # Sort all chunk indices by BM25 score in descending order
    ranked_bm25_indices = np.argsort(bm25_scores)[::-1]
    
    bm25_ranks = {}
    for rank_idx, doc_idx in enumerate(ranked_bm25_indices):
        doc = index_bundle.chunks[doc_idx]
        key = _get_chunk_key(doc)
        # Record 1-based rank
        bm25_ranks[key] = rank_idx + 1

    # --- 2. Dense Ranker (Chroma Similarity Search) ---
    # Query Chroma for all documents in the bundle to get a complete rank order
    num_chunks = len(index_bundle.chunks)
    vector_docs = index_bundle.vectorstore.similarity_search(query, k=num_chunks)
    
    vector_ranks = {}
    for rank_idx, doc in enumerate(vector_docs):
        key = _get_chunk_key(doc)
        # Record 1-based rank
        vector_ranks[key] = rank_idx + 1

    # --- 3. Reciprocal Rank Fusion ---
    all_keys = set(bm25_ranks.keys()).union(set(vector_ranks.keys()))
    rrf_scores = {}
    
    for key in all_keys:
        score = 0.0
        if key in bm25_ranks:
            score += 1.0 / (rrf_k + bm25_ranks[key])
        if key in vector_ranks:
            score += 1.0 / (rrf_k + vector_ranks[key])
        rrf_scores[key] = score

    # Sort keys by RRF score descending
    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    # Reconstruct (Document, score) list from top-k keys
    fused_results = []
    for key in sorted_keys[:k]:
        original_doc = key_to_doc[key]
        fused_results.append((original_doc, rrf_scores[key]))

    logger.info(f"RRF retrieval completed. Returned {len(fused_results)} documents.")
    return fused_results
