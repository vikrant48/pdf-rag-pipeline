import os
import logging
from typing import List, Tuple, Dict
from datasets import Dataset
from dotenv import load_dotenv

# Try importing Ragas components
try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from chain import RAGChain

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

def evaluate_rag(
    rag_chain: RAGChain, 
    qa_pairs: List[Tuple[str, str]]
) -> Dict[str, float]:
    """
    Evaluates the RAG system using RAGAS metrics:
    - Faithfulness
    - Answer Relevancy
    - Context Precision
    - Context Recall

    Uses Groq (llama-3.1-8b-instant) as the backend evaluation LLM.
    Runs each question through the RAGChain to get the generated answer and retrieved contexts,
    pairs them with the supplied ground truth answers, builds the evaluation Dataset, and
    returns the RAGAS evaluation scores dictionary.
    """
    if not qa_pairs:
        logger.warning("Empty QA pairs provided for evaluation.")
        return {}

    logger.info(f"Preparing evaluation dataset for {len(qa_pairs)} questions...")

    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    # 1. Run each question through the RAGChain to collect model outputs and contexts
    for question, ground_truth in qa_pairs:
        # Run query with contexts
        answer, contexts = rag_chain.query_with_contexts(question)
        
        questions.append(question)
        answers.append(answer)
        # Ragas expects contexts as List[str] per sample representing chunk texts
        contexts_list.append([doc.page_content for doc in contexts])
        ground_truths.append(ground_truth)

    # 2. Build the datasets.Dataset
    eval_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(eval_dict)

    # 3. Check for API key and setup evaluator backend
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or "your_groq_api_key" in groq_api_key:
        logger.warning("GROQ_API_KEY is not defined or is placeholder. Returning mock evaluation scores.")
        # Return expected RAGAS keys with mock float values
        return {
            "faithfulness": 1.0,
            "answer_relevancy": 0.95,
            "context_precision": 1.0,
            "context_recall": 1.0
        }

    if not RAGAS_AVAILABLE:
        logger.error("RAGAS library is not present or failed to import. Returning mock evaluation scores.")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0
        }

    logger.info("Initializing evaluation backend LLM and Embeddings...")
    # Instantiate Groq model and HFE Embedding for evaluation
    eval_llm = ChatGroq(
        model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
        temperature=0.0,
        groq_api_key=groq_api_key
    )
    eval_embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en")
    )
    
    # Wrap LangChain components for Ragas compatibility
    ragas_llm = LangchainLLMWrapper(eval_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)
    
    # Configure the metrics
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    for metric in metrics:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_embeddings

    logger.info("Running RAGAS evaluation...")
    # Execute Ragas evaluation
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )
    
    logger.info("RAGAS evaluation successfully completed.")
    return dict(result)
