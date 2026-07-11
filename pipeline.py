import os
import logging
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv

from ingest import ingest_pdfs
from clean import clean_documents
from chunk import chunk_documents
from index import build_indexes
from chain import RAGChain

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuration class for the RAGPipeline."""
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "400"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    k: int = int(os.getenv("RETRIEVAL_K", "5"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "5"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en")
    llm_model: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

import json
import time

class RAGPipeline:
    """
    Orchestration class that coordinates loading, cleaning, chunking, 
    indexing, and querying arbitrary PDF files.
    """
    def __init__(self, config: RAGConfig):
        self.config = config
        self.chain = None
        self.registry_path = os.path.join("data", "registry.json")
        self.selected_doc_ids = None
        logger.info(f"Initialized RAGPipeline with config: {config}")

    def list_documents(self):
        """Lists all indexed documents registered in the database, sorted by filename."""
        from db import db_list_docs
        docs = db_list_docs()
        docs.sort(key=lambda x: x["filename"])
        return docs

    def select_documents(self, indices):
        """Sets selected_doc_ids filter based on selection list (e.g. [1,3], integer or 'all')."""
        docs = self.list_documents()
        if not docs:
            self.selected_doc_ids = None
            return
            
        if isinstance(indices, str):
            if indices.strip().lower() == "all":
                self.selected_doc_ids = None
                return
            try:
                parts = [int(p.strip()) for p in indices.split(",") if p.strip()]
                indices = parts
            except ValueError:
                self.selected_doc_ids = None
                return

        if isinstance(indices, int):
            indices = [indices]
            
        selected_ids = []
        for idx in indices:
            if 1 <= idx <= len(docs):
                selected_ids.append(docs[idx - 1]["doc_id"])
        self.selected_doc_ids = selected_ids if selected_ids else None

    def ingest(self, pdf_paths: List[str], s3_metadata: dict = None):
        """
        Coordinates full ingestion phase:
        ingest -> clean -> chunk -> index -> generation chain
        """
        logger.info(f"Starting pipeline ingestion for {len(pdf_paths)} files...")
        # Clear retrieval cache for fresh ingestion queries
        from retrieve import clear_retrieval_cache
        clear_retrieval_cache()

        # 1. Ingest
        raw_docs = ingest_pdfs(pdf_paths)
        if not raw_docs:
            # We still initialize the chain using existing database chunks if registry shows documents
            registry = self.load_registry()
            if registry:
                logger.info("Raw docs empty, but existing registry found. Initializing indexes from DB.")
                # We can construct empty list or mock list to trigger fallback building.
                # Actually, if we have chunks in Chroma, build_indexes(chunks=[]) won't work,
                # but build_indexes accepts chunks. Let's see: we should raise error if absolutely no chunks
                # and no index exists yet.
            raise ValueError("No pages were loaded. Ingestion phase aborted.")

        # 2. Clean
        logger.info("Normalizing text content...")
        cleaned_docs = clean_documents(raw_docs)

        # 3. Chunk
        logger.info(f"Structuring chunks (size={self.config.chunk_size}, overlap={self.config.chunk_overlap})...")
        chunks = chunk_documents(
            cleaned_docs,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

        # 4. Index
        logger.info("Building hybrid indices...")
        index_bundle = build_indexes(chunks)

        # 5. Chain Integration
        logger.info("Wiring generation chain...")
        self.chain = RAGChain(
            index_bundle=index_bundle,
            model_name=self.config.llm_model,
            k=self.config.k,
            rrf_k=self.config.rrf_k,
            rerank_top_n=self.config.rerank_top_n
        )
        
        # 6. Update Database
        from ingest import calculate_sha256
        from db import db_save_doc
        for path in pdf_paths:
            try:
                if os.path.exists(path):
                    doc_id = calculate_sha256(path)
                    filename = os.path.basename(path)
                    pages_count = sum(1 for d in raw_docs if d.metadata.get("doc_id") == doc_id)
                    if pages_count > 0:
                        s3_key = None
                        public_url = None
                        if s3_metadata and path in s3_metadata:
                            s3_key, public_url = s3_metadata[path]
                        db_save_doc(doc_id, filename, s3_key, public_url, pages_count)
                    else:
                        # If file failed validation, remove it from DB if it exists
                        from db import get_connection, is_postgres_conn
                        conn = get_connection()
                        cursor = conn.cursor()
                        is_pg = is_postgres_conn(conn)
                        if is_pg:
                            cursor.execute("DELETE FROM uploaded_pdfs WHERE doc_id = %s;", (doc_id,))
                        else:
                            cursor.execute("DELETE FROM uploaded_pdfs WHERE doc_id = ?;", (doc_id,))
                        conn.commit()
                        cursor.close()
                        conn.close()
            except Exception as e:
                logger.error(f"Failed to update database registry for {path}: {e}")
        
        logger.info("Ingestion successfully completed. System is ready to query.")

    def query(self, question: str, metadata_filter: dict = None, session_id: str = "default") -> str:
        """Queries the underlying generation chain."""
        if self.chain is None:
            raise ValueError("Pipeline has not ingested any documents. Call ingest() first.")
        if self.selected_doc_ids:
            if metadata_filter is None:
                metadata_filter = {}
            metadata_filter["doc_id"] = self.selected_doc_ids
        return self.chain.query(question, metadata_filter=metadata_filter, session_id=session_id)

    def query_stream(self, question: str, metadata_filter: dict = None, session_id: str = "default"):
        """Queries the underlying generation chain yielding chunks in real time."""
        if self.chain is None:
            raise ValueError("Pipeline has not ingested any documents. Call ingest() first.")
        if self.selected_doc_ids:
            if metadata_filter is None:
                metadata_filter = {}
            metadata_filter["doc_id"] = self.selected_doc_ids
        yield from self.chain.query_stream(question, metadata_filter=metadata_filter, session_id=session_id)
