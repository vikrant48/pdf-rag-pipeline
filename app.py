import os
import logging
from typing import List, Union, Optional
from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from pipeline import RAGPipeline, RAGConfig
from db import initialize_database, db_list_docs, db_get_doc_by_id
from s3 import upload_pdf_to_s3
from ingest import calculate_sha256

# Configure logging at level INFO
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("rag_api")

app = FastAPI(
    title="PDF RAG Service API",
    description="FastAPI service wrapper around the hybrid multi-PDF RAG pipeline",
    version="1.0.0"
)

# Initialize pipeline
config = RAGConfig()
pipeline = RAGPipeline(config)

class QueryPayload(BaseModel):
    question: str
    session_id: str = "default"
    metadata_filter: Optional[dict] = None
    selection: Optional[Union[str, int, List[int]]] = None

class SelectionPayload(BaseModel):
    selection: Union[str, int, List[int]]
    session_id: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def read_dashboard():
    """Serves the single-page wizard dashboard interface."""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        logger.error(f"Failed to read index.html: {e}")
        return HTMLResponse(content="<h1>Dashboard UI file index.html is missing.</h1>", status_code=500)

@app.on_event("startup")
async def startup_event():
    """Initializes schema and runs auto-ingestion for new data folder PDFs."""
    # 1. Initialize schema
    try:
        initialize_database()
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        
    # 2. Scan and ingest new data PDFs
    disable_startup = os.getenv("DISABLE_STARTUP_INGESTION", str(os.getenv("RENDER") == "true")).lower() == "true"
    if disable_startup:
        logger.info("Startup PDF auto-ingestion skipped automatically (Render environment or DISABLE_STARTUP_INGESTION=true active).")
        return

    pdf_dir = "data"
    os.makedirs(pdf_dir, exist_ok=True)
    pdfs = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    if pdfs:
        try:
            # Check already indexed list from DB to prevent re-embedding on startup
            indexed_filenames = {d["filename"] for d in db_list_docs()}
            to_ingest = [p for p in pdfs if os.path.basename(p) not in indexed_filenames]
            if to_ingest:
                logger.info(f"Auto-ingesting new startup files: {to_ingest}")
                pipeline.ingest(to_ingest)
            else:
                logger.info("All startup PDF files are already loaded and indexed.")
        except Exception as e:
            logger.error(f"Failed to auto-ingest PDFs during startup: {e}")
    else:
        logger.info("No default PDFs found in 'data/' to auto-ingest during startup.")

@app.get("/documents")
async def get_documents(session_id: Optional[str] = None):
    """Retrieve the collection list of indexed documents registry."""
    try:
        docs = pipeline.list_documents(session_id)
        return {"documents": docs}
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/select")
async def select_documents(payload: SelectionPayload):
    """Set the document subset scope for subsequent query operations."""
    try:
        pipeline.select_documents(payload.selection, payload.session_id)
        selected_docs = []
        if pipeline.selected_doc_ids:
            all_docs = pipeline.list_documents(payload.session_id)
            selected_docs = [d for d in all_docs if d["doc_id"] in pipeline.selected_doc_ids]
        return {
            "status": "success",
            "selected_doc_ids": pipeline.selected_doc_ids,
            "selected_documents": selected_docs
        }
    except Exception as e:
        logger.error(f"Error setting doc selections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...), session_id: str = Form("system")):
    """
    Uploads user provided PDF to Supabase Storage S3, logs metadata records,
    and runs full text index parser ingestion.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF file uploads are supported.")
        
    local_dir = os.path.join("data", "uploads", session_id)
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, file.filename)
    
    try:
        # Read contents
        contents = await file.read()
        
        # Save locally so pipeline can ingest it
        with open(local_path, "wb") as f:
            f.write(contents)
            
        doc_id = calculate_sha256(local_path)
        
        # Check if already indexed in database
        existing = db_get_doc_by_id(doc_id)
        if existing:
            # We can re-trigger pipeline ingest to rebuild memory client config if process restarted,
            # but embedding cache will handle skipping Chroma embeddings.
            logger.info(f"Document {file.filename} is already indexed in DB context. Processing ingestion skip.")
            
        # Upload key to S3 bucket
        s3_key, public_url = upload_pdf_to_s3(contents, file.filename)
        
        # Run ingestion
        s3_meta = {local_path: (s3_key, public_url)}
        pipeline.ingest([local_path], s3_metadata=s3_meta, session_id=session_id)
        
        # Return registry doc representation
        registered_doc = db_get_doc_by_id(doc_id)
        return {
            "status": "success",
            "message": f"Successfully ingested {file.filename}.",
            "document": registered_doc
        }
    except Exception as e:
        logger.error(f"Error during ingestion route: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_pipeline(payload: QueryPayload):
    """Query the RAG pipeline generating a consolidated answer response."""
    if pipeline.chain is None:
        raise HTTPException(
            status_code=400, 
            detail="Pipeline is not initialized. Please call /ingest first."
        )
    try:
        if payload.selection is not None:
            pipeline.select_documents(payload.selection, payload.session_id)
            
        answer = pipeline.query(
            question=payload.question, 
            metadata_filter=payload.metadata_filter,
            session_id=payload.session_id
        )
        return {
            "question": payload.question,
            "session_id": payload.session_id,
            "answer": answer
        }
    except Exception as e:
        logger.error(f"Error during query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query_stream")
async def query_pipeline_stream(payload: QueryPayload):
    """Query the RAG pipeline yielding response tokens in real time (Server-Sent Events)."""
    if pipeline.chain is None:
        raise HTTPException(
            status_code=400, 
            detail="Pipeline is not initialized. Please call /ingest first."
        )
        
    if payload.selection is not None:
        pipeline.select_documents(payload.selection, payload.session_id)

    async def sse_generator():
        try:
            for chunk in pipeline.query_stream(
                question=payload.question,
                metadata_filter=payload.metadata_filter,
                session_id=payload.session_id
            ):
                yield f"data: {chunk}\n\n"
        except Exception as err:
            logger.error(f"Stream error: {err}")
            yield f"data: [ERROR] {err}\n\n"
            
    return StreamingResponse(sse_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
