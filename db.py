import os
import psycopg2
import sqlite3
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

DB_URL = os.getenv("SUPABASE_DB_URL")

def get_connection():
    """Returns a database connection (psycopg2 for PostgreSQL or sqlite3 for SQLite fallback)."""
    if DB_URL:
        try:
            conn = psycopg2.connect(DB_URL)
            logger.info("Connected to Supabase PostgreSQL database.")
            return conn
        except Exception as e:
            logger.warning(f"Failed to connect to Supabase PostgreSQL: {e}. Falling back to SQLite.")
    
    # Fallback to local SQLite
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/db.sqlite")
    # For SQLite, return sqlite3.Row objects to emulate dict keys
    conn.row_factory = sqlite3.Row
    return conn

def is_postgres_conn(conn) -> bool:
    """Checks if the connection is PostgreSQL."""
    return type(conn).__module__.startswith("psycopg2")

def initialize_database():
    """Initializes the database tables (either Postgres or SQLite)."""
    conn = get_connection()
    cursor = conn.cursor()
    
    is_pg = is_postgres_conn(conn)
    
    if is_pg:
        logger.info("Initializing Supabase PostgreSQL database tables...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS uploaded_pdfs (
                doc_id VARCHAR(64) PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                s3_key VARCHAR(512),
                public_url VARCHAR(1024),
                page_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history_logs (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
    else:
        logger.info("Initializing fallback SQLite database tables...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS uploaded_pdfs (
                doc_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                s3_key TEXT,
                public_url TEXT,
                page_count INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
    conn.commit()
    cursor.close()
    conn.close()
    logger.info("Database initialization completed successfully.")

def db_save_doc(doc_id: str, filename: str, s3_key: str, public_url: str, page_count: int):
    """Saves or updates an uploaded PDF metadata record."""
    conn = get_connection()
    cursor = conn.cursor()
    is_pg = is_postgres_conn(conn)
    
    if is_pg:
        query = """
            INSERT INTO uploaded_pdfs (doc_id, filename, s3_key, public_url, page_count)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (doc_id) DO UPDATE SET
                filename = EXCLUDED.filename,
                s3_key = EXCLUDED.s3_key,
                public_url = EXCLUDED.public_url,
                page_count = EXCLUDED.page_count;
        """
        cursor.execute(query, (doc_id, filename, s3_key, public_url, page_count))
    else:
        query = """
            INSERT OR REPLACE INTO uploaded_pdfs (doc_id, filename, s3_key, public_url, page_count)
            VALUES (?, ?, ?, ?, ?);
        """
        cursor.execute(query, (doc_id, filename, s3_key, public_url, page_count))
        
    conn.commit()
    cursor.close()
    conn.close()
    logger.info(f"Saved document metadata to DB: {filename} ({doc_id})")

def db_list_docs() -> List[dict]:
    """Lists all indexed documents registered in the database."""
    conn = get_connection()
    cursor = conn.cursor()
    is_pg = is_postgres_conn(conn)
    
    query = "SELECT doc_id, filename, s3_key, public_url, page_count, created_at FROM uploaded_pdfs ORDER BY created_at DESC;"
    cursor.execute(query)
    rows = cursor.fetchall()
    
    docs = []
    for r in rows:
        if is_pg:
            docs.append({
                "doc_id": r[0],
                "filename": r[1],
                "s3_key": r[2],
                "public_url": r[3],
                "page_count": r[4],
                "created_at": str(r[5])
            })
        else:
            # sqlite3.Row emulation
            docs.append({
                "doc_id": r["doc_id"],
                "filename": r["filename"],
                "s3_key": r["s3_key"],
                "public_url": r["public_url"],
                "page_count": r["page_count"],
                "created_at": str(r["created_at"])
            })
            
    cursor.close()
    conn.close()
    return docs

def db_get_doc_by_id(doc_id: str) -> Optional[dict]:
    """Gets an uploaded PDF by its doc_id."""
    conn = get_connection()
    cursor = conn.cursor()
    is_pg = is_postgres_conn(conn)
    
    if is_pg:
        query = "SELECT doc_id, filename, s3_key, public_url, page_count FROM uploaded_pdfs WHERE doc_id = %s;"
        cursor.execute(query, (doc_id,))
        r = cursor.fetchone()
        if r:
            cursor.close()
            conn.close()
            return {"doc_id": r[0], "filename": r[1], "s3_key": r[2], "public_url": r[3], "page_count": r[4]}
    else:
        query = "SELECT doc_id, filename, s3_key, public_url, page_count FROM uploaded_pdfs WHERE doc_id = ?;"
        cursor.execute(query, (doc_id,))
        r = cursor.fetchone()
        if r:
            cursor.close()
            conn.close()
            return {"doc_id": r["doc_id"], "filename": r["filename"], "s3_key": r["s3_key"], "public_url": r["public_url"], "page_count": r["page_count"]}
            
    cursor.close()
    conn.close()
    return None

def db_save_chat(session_id: str, question: str, answer: str):
    """Logs a user query and assistant response."""
    conn = get_connection()
    cursor = conn.cursor()
    is_pg = is_postgres_conn(conn)
    
    if is_pg:
        query = "INSERT INTO chat_history_logs (session_id, question, answer) VALUES (%s, %s, %s);"
        cursor.execute(query, (session_id, question, answer))
    else:
        query = "INSERT INTO chat_history_logs (session_id, question, answer) VALUES (?, ?, ?);"
        cursor.execute(query, (session_id, question, answer))
        
    conn.commit()
    cursor.close()
    conn.close()
    logger.info(f"Saved chat history log for session '{session_id}' to DB.")

def db_get_chat_history(session_id: str) -> List[Tuple[str, str]]:
    """Retrieves chat logs for a session_id ordered by entry timestamp."""
    conn = get_connection()
    cursor = conn.cursor()
    is_pg = is_postgres_conn(conn)
    
    if is_pg:
        query = "SELECT question, answer FROM chat_history_logs WHERE session_id = %s ORDER BY id ASC;"
        cursor.execute(query, (session_id,))
        rows = cursor.fetchall()
        history = [(r[0], r[1]) for r in rows]
    else:
        query = "SELECT question, answer FROM chat_history_logs WHERE session_id = ? ORDER BY id ASC;"
        cursor.execute(query, (session_id,))
        rows = cursor.fetchall()
        history = [(r["question"], r["answer"]) for r in rows]
        
    cursor.close()
    conn.close()
    return history
