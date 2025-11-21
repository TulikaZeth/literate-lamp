"""
Enhanced CLI Interface for RAG Bot
Multi-modal document ingestion with OCR and advanced retrieval.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from pdf_processor import DocumentProcessor
from vector_store import VectorStoreManager
from rag_engine import RAGEngine


class RAGBotCLI:
    """Enhanced command-line interface for multi-modal RAG bot."""
    
    def __init__(self):
        """Initialize CLI with enhanced features."""
        load_dotenv()
        
        self.vector_store = VectorStoreManager(
            embedding_model=os.getenv("EMBEDDING_MODEL", "models/sentence-transformers/all-MiniLM-L6-v2")
        )
        
        self.rag_engine = RAGEngine(
            vector_store=self.vector_store,
            model_name=os.getenv("CHAT_MODEL", "gemini-1.5-pro"),
            retrieval_k=int(os.getenv("RETRIEVAL_K", "8")),
            use_reranker=os.getenv("USE_RERANKER", "true").lower() == "true",
            reranker_top_k=int(os.getenv("RERANKER_TOP_K", "4"))
        )
        
        self.doc_processor = DocumentProcessor(
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            use_ocr=os.getenv("USE_OCR", "true").lower() == "true"
        )
    
    def ingest_documents(self, file_paths: list[str]):
        """
        Ingest multi-modal documents into the vector store.
        
        Args:
            file_paths: List of paths to document files (PDF, images, text, DOCX)
        """
        print(f"\n📚 Ingesting {len(file_paths)} file(s)...\n")
        
        chunks = self.doc_processor.process_multiple_documents(file_paths)
        
        if chunks:
            self.vector_store.add_documents(chunks)
            print(f"\n✅ Successfully ingested {len(chunks)} chunks from {len(file_paths)} file(s)")
            print(f"📊 Total documents in knowledge base: {self.vector_store.get_document_count()}\n")
        else:
            print("\n❌ No documents were successfully processed\n")
    
    def query_loop(self):
        """Interactive query loop."""
        doc_count = self.vector_store.get_document_count()
        
        if doc_count == 0:
            print("⚠️  No documents in knowledge base. Please ingest PDFs first.")
            return
        
        print(f"\n💬 RAG Bot Ready! ({doc_count} chunks loaded)")
        print("Ask questions about your documents (type 'quit' to exit)\n")
        print("-" * 60)
        
        while True:
            try:
                question = input("\n🤔 Your Question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                print("\n🤖 Thinking...\n")
                result = self.rag_engine.query(question)
                
                print("💡 Answer:")
                print("=" * 70)
                print(result["answer"])
                print("=" * 70)
                
                if result["source_documents"]:
                    print("\n📚 Sources & Citations:")
                    print("-" * 70)
                    for i, doc in enumerate(result["source_documents"], 1):
                        source = doc.metadata.get('source', 'Unknown')
                        chunk_id = doc.metadata.get('chunk_id', 'N/A')
                        doc_type = doc.metadata.get('doc_type', 'unknown')
                        rerank_score = doc.metadata.get('rerank_score', None)
                        
                        print(f"\n[{i}] {source}")
                        print(f"    Type: {doc_type} | Chunk: {chunk_id}", end="")
                        if rerank_score:
                            print(f" | Relevance: {rerank_score:.3f}")
                        else:
                            print()
                        
                        excerpt = doc.page_content[:200].replace('\n', ' ').strip()
                        print(f"    Excerpt: \"{excerpt}...\"")
                
                # Show metadata
                if result.get("metadata"):
                    print("\n" + "-" * 70)
                    meta = result["metadata"]
                    print(f"ℹ️  {meta['num_sources']} sources", end="")
                    if meta['used_reranker']:
                        print(" (reranked)", end="")
                    print(f" | Model: {meta['model']}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
    
    def clear_database(self):
        """Clear the vector database."""
        confirm = input("⚠️  Are you sure you want to clear the knowledge base? (yes/no): ")
        if confirm.lower() == 'yes':
            self.vector_store.clear_vectorstore()
            print("✅ Knowledge base cleared!")
        else:
            print("❌ Operation cancelled")
    
    def show_stats(self):
        """Show database statistics."""
        doc_count = self.vector_store.get_document_count()
        print(f"\n📊 Knowledge Base Statistics")
        print("-" * 60)
        print(f"Total document chunks: {doc_count}")
        print("-" * 60)


def main():
    """Main CLI entry point."""
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY not found in environment variables")
        print("Please create a .env file with your Google API key")
        print("Get your key from: https://makersuite.google.com/app/apikey")
        sys.exit(1)
    
    cli = RAGBotCLI()
    
    if len(sys.argv) < 2:
        print("\n📚 Enhanced RAG Bot CLI")
        print("=" * 70)
        print("\nUsage:")
        print("  python cli.py ingest <file1> [file2 ...]  # Add documents to knowledge base")
        print("  python cli.py query                        # Start interactive Q&A")
        print("  python cli.py stats                        # Show database stats")
        print("  python cli.py clear                        # Clear knowledge base")
        print("\nSupported Formats:")
        print("  📄 PDF (with OCR for scanned documents)")
        print("  🖼️  Images: PNG, JPG, JPEG (requires OCR)")
        print("  📝 Text: TXT, MD")
        print("  📄 DOCX documents")
        print("\nExamples:")
        print("  python cli.py ingest book.pdf notes.txt diagram.png")
        print("  python cli.py query")
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    if command == "ingest":
        if len(sys.argv) < 3:
            print("❌ Error: Please provide document file paths")
            print("Supported formats: PDF, PNG, JPG, TXT, MD, DOCX")
            sys.exit(1)
        
        file_paths = sys.argv[2:]
        
        # Validate files exist
        for path in file_paths:
            if not Path(path).exists():
                print(f"❌ Error: File not found: {path}")
                sys.exit(1)
        
        cli.ingest_documents(file_paths)
    
    elif command == "query":
        cli.query_loop()
    
    elif command == "stats":
        cli.show_stats()
    
    elif command == "clear":
        cli.clear_database()
    
    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
