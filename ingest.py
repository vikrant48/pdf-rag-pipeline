import hashlib
import logging
import os
import pypdf
import pdfplumber
from typing import List
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

def table_to_markdown(table: List[List[str]]) -> str:
    """Converts a 2D table representing list of rows to a markdown table."""
    if not table or not any(table):
        return ""
    # Clean cell values
    cleaned_table = []
    for row in table:
        if not row:
            continue
        cleaned_table.append([str(cell or "").replace("\n", " ").strip() for cell in row])
    
    if not cleaned_table:
        return ""
        
    header = cleaned_table[0]
    rows = cleaned_table[1:]
    
    md = "\n" + "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join(["---"] * len(header)) + " |\n"
    for row in rows:
        if len(row) < len(header):
            row += [""] * (len(header) - len(row))
        md += "| " + " | ".join(row[:len(header)]) + " |\n"
    return md + "\n"

def extract_page_content_with_tables(page) -> str:
    """
    Extracts text from a pdfplumber page. Filters out char elements falling 
    inside any identified tables to prevent cluttered raw extraction, then 
    appends formatted markdown table blocks.
    """
    tables = page.find_tables()
    if not tables:
        return page.extract_text() or ""
        
    table_bboxes = [t.bbox for t in tables]
    
    # Filter function to exclude characters within the table regions
    def not_in_table(obj):
        if obj.get("object_type") != "char":
            return True
        x0, top, x1, bottom = obj["x0"], obj["top"], obj["x1"], obj["bottom"]
        for tx0, ttop, tx1, tbottom in table_bboxes:
            if (x0 >= tx0 - 1) and (x1 <= tx1 + 1) and (top >= ttop - 1) and (bottom <= tbottom + 1):
                return False
        return True
        
    filtered_page = page.filter(not_in_table)
    text_without_tables = filtered_page.extract_text() or ""
    
    # Convert and group extracted tables to markdown
    md_tables = []
    for t in tables:
        raw_table_data = t.extract()
        md_str = table_to_markdown(raw_table_data)
        if md_str.strip():
            md_tables.append(md_str)
            
    # Combine normal text and table blocks
    return text_without_tables + "\n" + "\n".join(md_tables)

def ingest_pdfs(file_paths: List[str]) -> List[Document]:
    """
    Accepts a list of arbitrary PDF file paths, loads them with pdfplumber,
    and returns a list of LangChain Document objects with generic metadata:
    - source (filename)
    - page (page number)
    - doc_id (stable SHA-256 content hash)
    
    Performs corruption and encryption checks to reject bad files gracefully.
    Extracts tabular content formatting it directly into Markdown table layouts.
    """
    all_documents = []
    
    for path in file_paths:
        if not os.path.exists(path):
            logger.error(f"File does not exist: {path}")
            continue
            
        try:
            logger.info(f"Processing PDF file: {path}")
            
            # Check length/size
            if os.path.getsize(path) == 0:
                logger.error(f"PDF file is empty: {path}")
                continue
            
            # --- 1. Passwords and Corruption Checks ---
            try:
                reader = pypdf.PdfReader(path)
                if reader.is_encrypted:
                    logger.error(f"Validation failed: PDF file is encrypted/password-protected: {path}")
                    continue
                if len(reader.pages) == 0:
                    logger.error(f"Validation failed: PDF contains no pages: {path}")
                    continue
            except Exception as read_err:
                logger.error(f"Validation failed: PDF is corrupted or unreadable: {path}. Error: {read_err}")
                continue

            # Compute stable hash
            doc_id = calculate_sha256(path)
            filename = os.path.basename(path)
            
            # --- 2. pdfplumber Table-Aware Parsing ---
            with pdfplumber.open(path) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    try:
                        page_content = extract_page_content_with_tables(page)
                    except Exception as parse_err:
                        logger.warning(f"Table-aware parsing failed on page {page_idx + 1}, using normal text: {parse_err}")
                        page_content = page.extract_text() or ""
                    
                    if not page_content.strip():
                        page_content = "[Empty Page]"
                    
                    doc = Document(
                        page_content=page_content,
                        metadata={
                            "source": filename,
                            "page": page_idx,
                            "doc_id": doc_id
                        }
                    )
                    all_documents.append(doc)
            
            logger.info(f"Successfully processed PDF file: {path} ({len(pdf.pages)} pages)")
                
        except Exception as e:
            logger.exception(f"Failed to process PDF file {path} due to error: {e}")
            
    return all_documents
