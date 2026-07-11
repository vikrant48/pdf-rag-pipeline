import os
import hashlib
import logging
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

# Load environment variables from .env
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class IndexBundle:
    bm25: BM25Okapi
    vectorstore: Chroma
    chunks: List[Document]

def build_indexes(chunks: List[Document]) -> IndexBundle:
    """
    Builds two indexes from a list of chunks:
    1. Sparse index (BM25Okapi) over lowercased whitespace-tokenized chunk text.
    2. Dense index (Chroma vector store) using HuggingFaceEmbeddings(BAAI/bge-small-en).
    
    The collection name in Chroma is derived from a SHA-256 hash of the sorted list
    of unique doc_ids in the chunk set.
    """
    if not chunks:
        raise ValueError("Cannot build indexes from an empty list of chunks.")

    logger.info(f"Building indexes for {len(chunks)} chunks")

    # 1. Generate Chroma collection name derived from a hash of the doc_ids
    unique_doc_ids = sorted(list({
        str(chunk.metadata.get("doc_id", "unknown")) 
        for chunk in chunks
    }))
    doc_ids_str = ",".join(unique_doc_ids)
    doc_ids_hash = hashlib.sha256(doc_ids_str.encode("utf-8")).hexdigest()
    
    # Restrict collection name to 3-63 chars and follow Chroma naming requirements
    # (starts/ends with alphanumeric, contains only alphanumeric, underscores, or hyphens)
    collection_name = f"col_{doc_ids_hash[:32]}"
    logger.info(f"Generated Chroma collection name: {collection_name} (derived from doc_ids hash)")

    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    # 2. Build Chroma Vector Store
    # Retrieve embedding model and persistence directory from env or defaults
    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en")
    persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    
    logger.info(f"Initializing dense HuggingFaceEmbeddings with model: {embedding_model}")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        # Use CPU by default to keep verification lightweight and generic
        model_kwargs={"device": "cpu"}
    )
    
    # Check if this collection already exists and is populated in Chroma database
    already_indexed = False
    try:
        import chromadb
        client = chromadb.PersistentClient(path=persist_directory)
        existing_collections = [c.name for c in client.list_collections()]
        if collection_name in existing_collections:
            col = client.get_collection(collection_name)
            if col.count() > 0:
                already_indexed = True
                logger.info(f"Chroma collection '{collection_name}' already exists and contains {col.count()} documents. Skipping re-embedding.")
    except Exception as e:
        logger.warning(f"Could not check Chroma database for existing collection '{collection_name}': {e}")

    if already_indexed:
        logger.info(f"Re-initializing Chroma vector store wrapper around existing collection '{collection_name}'...")
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        try:
            col_data = vectorstore._collection.get()
            if col_data and col_data.get("documents"):
                loaded_chunks = []
                metas = col_data.get("metadatas") or [{}] * len(col_data["documents"])
                for doc_text, meta in zip(col_data["documents"], metas):
                    # Ensure metadata is not None
                    loaded_chunks.append(Document(page_content=doc_text, metadata=meta or {}))
                if loaded_chunks:
                    chunks = loaded_chunks
                    logger.info(f"Loaded {len(chunks)} exact document chunks from Chroma to ensure consistency.")
        except Exception as load_err:
            logger.warning(f"Could not load documents from existing collection to ensure consistency: {load_err}")
    else:
        logger.info(f"Creating Chroma vector store at '{persist_directory}'...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
    
    # 3. Build BM25 Okapi Index
    # Read chunk text, lowercase, and tokenize by whitespace
    logger.info("Building BM25 Okapi index...")
    tokenized_corpus = [
        chunk.page_content.lower().split() 
        for chunk in chunks
    ]
    bm25 = BM25Okapi(tokenized_corpus)

    logger.info("Hybrid indexing completed successfully.")
    return IndexBundle(
        bm25=bm25,
        vectorstore=vectorstore,
        chunks=chunks
    )
