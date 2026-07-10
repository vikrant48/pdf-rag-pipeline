import re
import logging
from collections import defaultdict
from typing import List, Dict, Set
from langchain_core.documents import Document

# Configure logging
logger = logging.getLogger(__name__)

# Regular expressions for standalone page numbers
PAGE_NUMBER_PATTERNS = [
    re.compile(r'^\s*\d+\s*$'),                                      # e.g., "12" or " 12 "
    re.compile(r'^\s*page\s+\d+\s*$', re.IGNORECASE),                # e.g., "Page 12"
    re.compile(r'^\s*-\s*\d+\s*-\s*$'),                             # e.g., "- 12 -"
    re.compile(r'^\s*\[\s*\d+\s*\]\s*$'),                           # e.g., "[12]"
    re.compile(r'^\s*\b(?:page|pg)\b\.?\s*\d+\s*of\s*\d+\s*$', re.IGNORECASE), # e.g., "Page 1 of 10"
]

def is_page_number_line(stripped_line: str) -> bool:
    """Helper to detect if a line is a standalone page number."""
    return any(pattern.match(stripped_line) for pattern in PAGE_NUMBER_PATTERNS)

def detect_headers_footers(docs: List[Document], threshold: float = 0.6) -> Set[str]:
    """
    Identifies lines that appear on more than threshold (60%) of the document pages.
    Only runs if there are 3 or more pages.
    """
    total_pages = len(docs)
    if total_pages < 3:
        return set()

    line_page_counts = defaultdict(int)
    
    for doc in docs:
        # We split, normalize newlines, and extract unique lines on each page
        content = doc.page_content.replace("\r\n", "\n")
        unique_lines_on_page = set()
        
        for line in content.splitlines():
            stripped = line.strip()
            if stripped:
                unique_lines_on_page.add(stripped)
                
        for line in unique_lines_on_page:
            line_page_counts[line] += 1

    recurring_lines = set()
    for line, count in line_page_counts.items():
        if (count / total_pages) > threshold:
            recurring_lines.add(line)
            
    return recurring_lines

def clean_page_content(
    content: str, 
    header_footer_denylist: Set[str]
) -> str:
    """
    Cleans the raw text content of a single page:
    - Normalizes newlines
    - Strips lines in the header_footer_denylist
    - Strips page number lines
    - Rejoins hyphenated line-break words (e.g. multi-\npage -> multipage)
    - Collapses excess vertical and horizontal whitespace
    """
    # 1. Normalize line endings
    content = content.replace("\r\n", "\n")
    
    # 2. Filter out repeating headers, footers, and page numbers
    cleaned_lines = []
    for line in content.splitlines():
        stripped = line.strip()
        
        # Skip empty lines during list filtering (we will collapse newlines later)
        if not stripped:
            cleaned_lines.append("")
            continue
            
        # Check against headers/footers and page number patterns
        if stripped in header_footer_denylist:
            continue
        if is_page_number_line(stripped):
            continue
            
        cleaned_lines.append(line)
        
    text = "\n".join(cleaned_lines)
    
    # 3. Rejoin hyphenated line-break words: (word)-\n(word) -> (word)(word)
    # Handles potential spaces around the newline
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # 4. Collapse excess horizontal spacing (multiple spaces/tabs to a single space)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 5. Collapse excess vertical spacing (3+ newlines to 2 newlines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Final trim of surrounding whitespace
    return text.strip()

def clean_documents(documents: List[Document]) -> List[Document]:
    """
    Takes a list of raw page-level Documents, normalizes the text in place,
    and returns the same list of Document objects with cleaned page_content.
    """
    # Group documents by doc_id (using project source path as fallback)
    grouped_docs = defaultdict(list)
    for doc in documents:
        doc_key = doc.metadata.get("doc_id") or doc.metadata.get("source", "default_doc")
        grouped_docs[doc_key].append(doc)
        
    for doc_id, doc_list in grouped_docs.items():
        logger.info(f"Cleaning document group '{doc_id}' with {len(doc_list)} pages")
        
        # Detect repeated headers/footers for this document group
        header_footer_denylist = detect_headers_footers(doc_list, threshold=0.6)
        if header_footer_denylist:
            logger.info(f"Detected {len(header_footer_denylist)} repeating headers/footers for '{doc_id}'")
            
        # Clean each page's content
        for doc in doc_list:
            doc.page_content = clean_page_content(doc.page_content, header_footer_denylist)
            
    return documents
