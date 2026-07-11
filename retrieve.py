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

# Simple in-memory cache mapping (query, allowed_doc_ids_tuple) -> List[Tuple[Document, float]]
_retrieval_cache = {}

def clear_retrieval_cache():
    """Clears the global retrieval cache."""
    _retrieval_cache.clear()
    logger.info("Retrieval cache cleared.")

def hybrid_search(
    query: str, 
    index_bundle: IndexBundle, 
    k: int = None, 
    rrf_k: int = None,
    metadata_filter: dict = None
) -> List[Tuple[Document, float]]:
    """
    Performs hybrid search combining BM25 (sparse) and Chroma (dense) retrieval
    using Reciprocal Rank Fusion (RRF), supporting metadata pre-filtering on
    source/page/doc_id attributes.
    """
    if not index_bundle.chunks:
        logger.warning("Empty chunk list in IndexBundle, returning empty results.")
        return []

    if k is None:
        k = int(os.getenv("RETRIEVAL_K", "5"))
    if rrf_k is None:
        rrf_k = int(os.getenv("RRF_K", "60"))

    # Resolve document selection to build cache key
    allowed_doc_ids = None
    if metadata_filter and "doc_id" in metadata_filter:
        val = metadata_filter["doc_id"]
        if isinstance(val, list):
            allowed_doc_ids = tuple(sorted(val))
        elif isinstance(val, str):
            allowed_doc_ids = (val,)

    cache_key = (query.strip().lower(), allowed_doc_ids)
    if cache_key in _retrieval_cache:
        logger.info(f"Retrieval cache hit for query: '{query}' with doc scope: {allowed_doc_ids}")
        return _retrieval_cache[cache_key]

    # Resolve all matching chunk indices for BM25 metadata filtering
    matching_indices = set()
    if metadata_filter:
        for idx, doc in enumerate(index_bundle.chunks):
            match = True
            for key, val in metadata_filter.items():
                meta_val = doc.metadata.get(key)
                if isinstance(val, list):
                    if meta_val not in val:
                        match = False
                        break
                else:
                    if meta_val != val:
                        match = False
                        break
            if match:
                matching_indices.add(idx)

        if not matching_indices:
            logger.info("Metadata filter active but no document chunks match criteria. Returning empty.")
            return []

    # Map metadata filter directly onto Chroma client where query syntax format
    chroma_filter = None
    if metadata_filter:
        filter_parts = []
        for key, val in metadata_filter.items():
            if isinstance(val, list):
                if len(val) == 1:
                    filter_parts.append({key: val[0]})
                elif len(val) > 1:
                    filter_parts.append({key: {"$in": val}})
            else:
                filter_parts.append({key: val})
        if len(filter_parts) == 1:
            chroma_filter = filter_parts[0]
        elif len(filter_parts) > 1:
            chroma_filter = {"$and": filter_parts}

    logger.info(f"Initiating hybrid search for query: '{query}' (k={k}, rrf_k={rrf_k}, filter={metadata_filter})")

    # Build document identity maps to link back to the exact pristine chunk instances
    key_to_doc = {_get_chunk_key(doc): doc for doc in index_bundle.chunks}

    # --- 1. Sparse Ranker (BM25Okapi) ---
    tokenized_query = query.lower().split()
    bm25_scores = index_bundle.bm25.get_scores(tokenized_query)
    
    # Sort all chunk indices by BM25 score in descending order
    ranked_bm25_indices = np.argsort(bm25_scores)[::-1]
    
    bm25_ranks = {}
    rank_count = 1
    for doc_idx in ranked_bm25_indices:
        if metadata_filter and doc_idx not in matching_indices:
            continue
        doc = index_bundle.chunks[doc_idx]
        key = _get_chunk_key(doc)
        # Record 1-based rank
        bm25_ranks[key] = rank_count
        rank_count += 1

    # --- 2. Dense Ranker (Chroma Similarity Search) ---
    # Query Chroma for documents, passing the metadata filter directly
    num_chunks = len(index_bundle.chunks)
    vector_docs = index_bundle.vectorstore.similarity_search(
        query, 
        k=num_chunks,
        filter=chroma_filter
    )
    
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
    _retrieval_cache[cache_key] = fused_results
    return fused_results
