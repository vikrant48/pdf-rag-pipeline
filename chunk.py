import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logger = logging.getLogger(__name__)

def chunk_documents(
    documents: List[Document], 
    chunk_size: int = 400, 
    chunk_overlap: int = 100
) -> List[Document]:
    """
    Splits a list of cleaned Document objects into smaller chunks using 
    RecursiveCharacterTextSplitter while preserving source, page, and doc_id metadata.
    """
    logger.info(f"Chunking {len(documents)} documents with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    # Initialize splitter
    # By default, RecursiveCharacterTextSplitter splits by double newlines, single newlines, space, and finally characters
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    # Split the documents
    split_docs = splitter.split_documents(documents)
    
    # Normalize and ensure metadata contains exclusively 'source', 'page', and 'doc_id'
    cleaned_split_docs = []
    for chunk in split_docs:
        meta = chunk.metadata
        # Extract required metadata keys precisely, defaulting if missing
        sanitized_meta = {
            "source": meta.get("source", "unknown"),
            "page": meta.get("page", 0),
            "doc_id": meta.get("doc_id", "unknown")
        }
        chunk.metadata = sanitized_meta
        cleaned_split_docs.append(chunk)
        
    logger.info(f"Created {len(cleaned_split_docs)} chunks from {len(documents)} source pages")
    return cleaned_split_docs
