import os
import sys
from dotenv import load_dotenv
from pipeline import RAGPipeline, RAGConfig

def main():
    # Load environment variables
    load_dotenv()
    
    # 1. Initialize Pipeline Config from Env
    config = RAGConfig()
    pipeline = RAGPipeline(config)
    
    # 2. Check if data folder exists and list PDFs
    pdf_dir = "data"
    os.makedirs(pdf_dir, exist_ok=True)
    
    print("=" * 60)
    print(" PDF RAG Orchestrator CLI ")
    print("=" * 60)
    print(f"Place your PDF files inside the folder: '{pdf_dir}/'")
    
    # List PDFs in folder
    pdfs = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    
    if not pdfs:
        print(f"\nNo PDF files found in '{pdf_dir}/' directory.")
        print("Please add at least one PDF file (e.g. to d:\\RAG\\pdf-rag-service\\data\\) and run this script again.")
        sys.exit(0)
        
    print("\nFound PDF files:")
    for idx, pdf in enumerate(pdfs):
        print(f" [{idx + 1}] {pdf}")
        
    # Ingest the PDFs
    pdf_paths = [os.path.join(pdf_dir, pdf) for pdf in pdfs]
    print(f"\nIngesting {len(pdf_paths)} documents...")
    try:
        pipeline.ingest(pdf_paths)
        print("Documents ingested successfully!")
    except Exception as e:
        print(f"Error during document ingestion: {e}")
        sys.exit(1)
        
    # 3. Interactive CLI Query loop
    print("\nYou can now ask questions about the ingested documents. (Type 'exit' to quit)")
    
    try:
        docs = pipeline.list_documents()
        if docs:
            print("\nPresenting uploaded PDF files:")
            for idx, doc in enumerate(docs):
                print(f"  [{idx + 1}] {doc['filename']} ({doc['page_count']} pages, indexed ✓)")
            initial_sel = input("\nSelect target PDFs to query (e.g. 'all', '1,3', or '2'): ").strip()
            pipeline.select_documents(initial_sel)
            if pipeline.selected_doc_ids:
                selected_names = [d['filename'] for d in docs if d['doc_id'] in pipeline.selected_doc_ids]
                print(f"Scoping query selection to: {', '.join(selected_names)}")
            else:
                print("Scoping query selection to ALL document units.")
    except Exception as pr_err:
        print(f"Failed to load document selection menu: {pr_err}")

    while True:
        try:
            query = input("\nQuery (or type 'select' to change file scope): ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
                
            if query.lower() == "select":
                docs = pipeline.list_documents()
                if docs:
                    print("\nPresenting uploaded PDF files:")
                    for idx, doc in enumerate(docs):
                        print(f"  [{idx + 1}] {doc['filename']} ({doc['page_count']} pages, indexed ✓)")
                    new_sel = input("\nSelect target PDFs to query (e.g. 'all', '1,3', or '2'): ").strip()
                    pipeline.select_documents(new_sel)
                    if pipeline.selected_doc_ids:
                        selected_names = [d['filename'] for d in docs if d['doc_id'] in pipeline.selected_doc_ids]
                        print(f"Scoping query selection key changed to: {', '.join(selected_names)}")
                    else:
                        print("Scoping query selection key reset to ALL document units.")
                else:
                    print("No indexed documents registered.")
                continue

            print("Thinking...")
            print("Answer: ", end="", flush=True)
            for chunk in pipeline.query_stream(query):
                print(chunk, end="", flush=True)
            print()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error querying pipeline: {e}")

if __name__ == "__main__":
    main()
