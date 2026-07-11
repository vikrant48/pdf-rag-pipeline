import logging
from sentence_transformers import SentenceTransformer, CrossEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("downloader")

def main():
    logger.info("Downloading embedding model: BAAI/bge-small-en")
    SentenceTransformer('BAAI/bge-small-en')
    
    logger.info("Downloading reranker model: cross-encoder/ms-marco-MiniLM-L-6-v2")
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    logger.info("Model pre-download complete.")

if __name__ == "__main__":
    main()
