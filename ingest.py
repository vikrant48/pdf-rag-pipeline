import hashlib
import logging
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# Configure logging
logger = logging.getLogger(__name__)

def calculate_sha256(file_path: str) -> str:
    """Calculate the SHA-256 hash of a file's content to use as a stable doc_id."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read file in chunks to handle large files efficiently
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def ingest_pdfs(file_paths: List[str]) -> List[Document]:
    """
    Accepts a list of arbitrary PDF file paths, loads them with PyPDFLoader,
    and returns a list of LangChain Document objects with generic metadata:
    - source (filename)
    - page (page number)
    - doc_id (stable SHA-256 content hash)
    
    Handles empty/corrupt PDFs gracefully with logging.
    """
    all_documents = []
    
    for path in file_paths:
        if not os.path.exists(path):
            logger.error(f"File does not exist: {path}")
            continue
            
        try:
            logger.info(f"Processing PDF file: {path}")
            
            # Check if file is empty
            if os.path.getsize(path) == 0:
                logger.error(f"PDF file is empty: {path}")
                continue
            
            # Compute stable hash
            doc_id = calculate_sha256(path)
            
            # Load PDF documents
            loader = PyPDFLoader(path)
            loaded_docs = loader.load()
            
            # Ensure custom metadata format
            filename = os.path.basename(path)
            for doc in loaded_docs:
                page = doc.metadata.get("page", 0)
                # Restructure metadata to contain only specified generic keys
                doc.metadata = {
                    "source": filename,
                    "page": page,
                    "doc_id": doc_id
                }
                all_documents.append(doc)
                
        except Exception as e:
            logger.exception(f"Failed to process PDF file {path} due to error: {e}")
            
    return all_documents
