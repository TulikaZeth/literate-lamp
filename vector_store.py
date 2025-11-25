"""
Vector Store Manager
Manages ChromaDB vector store for document embeddings and retrieval.
"""

import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


class VectorStoreManager:
    """Manage vector store for document embeddings."""
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "ragbot_docs",
        embedding_model: str = "models/text-embedding-004"
    ):
        """
        Initialize vector store manager.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection
            embedding_model: Google Gemini embedding model to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize embeddings with Google Gemini (no local downloads!)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        # Initialize or load vector store
        self.vectorstore = None
        self._load_or_create_vectorstore()
    
    def _load_or_create_vectorstore(self):
        """Load existing vector store or create a new one."""
        if os.path.exists(self.persist_directory):
            print(f"Loading existing vector store from {self.persist_directory}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
        else:
            print(f"Creating new vector store at {self.persist_directory}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        print(f"Adding {len(documents)} documents to vector store...")
        ids = self.vectorstore.add_documents(documents)
        print(f"Successfully added documents. Total documents in store: {self.get_document_count()}")
        
        return ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of documents to return
            filter: Optional metadata filter
            
        Returns:
            List of similar documents
        """
        return self.vectorstore.similarity_search(
            query,
            k=k,
            filter=filter
        )
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List[tuple]:
        """
        Search for similar documents with relevance scores.
        
        Args:
            query: Query text
            k: Number of documents to return
            filter: Optional metadata filter
            
        Returns:
            List of (document, score) tuples
        """
        return self.vectorstore.similarity_search_with_score(
            query,
            k=k,
            filter=filter
        )
    
    def get_document_count(self) -> int:
        """Get the total number of documents in the vector store."""
        try:
            collection = self.vectorstore._collection
            return collection.count()
        except Exception:
            return 0
    
    def clear_vectorstore(self):
        """Clear all documents from the vector store."""
        print("Clearing vector store...")
        self.vectorstore.delete_collection()
        self._load_or_create_vectorstore()
        print("Vector store cleared.")
    
    def as_retriever(self, search_kwargs: Optional[dict] = None):
        """
        Get a retriever interface for the vector store.
        
        Args:
            search_kwargs: Search parameters (e.g., {'k': 4})
            
        Returns:
            Retriever object
        """
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
