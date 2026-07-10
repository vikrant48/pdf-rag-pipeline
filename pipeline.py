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

class RAGPipeline:
    """
    Orchestration class that coordinates loading, cleaning, chunking, 
    indexing, and querying arbitrary PDF files.
    """
    def __init__(self, config: RAGConfig):
        self.config = config
        self.chain = None
        logger.info(f"Initialized RAGPipeline with config: {config}")

    def ingest(self, pdf_paths: List[str]):
        """
        Coordinates full ingestion phase:
        ingest -> clean -> chunk -> index -> generation chain
        """
        logger.info(f"Starting pipeline ingestion for {len(pdf_paths)} files...")

        # 1. Ingest
        raw_docs = ingest_pdfs(pdf_paths)
        if not raw_docs:
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
        logger.info("Ingestion successfully completed. System is ready to query.")

    def query(self, question: str) -> str:
        """Queries the underlying generation chain."""
        if self.chain is None:
            raise ValueError("Pipeline has not ingested any documents. Call ingest() first.")
        return self.chain.query(question)
