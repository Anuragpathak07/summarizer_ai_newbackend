import os
import logging
import uuid
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from utils.logger_config import setup_logger

logger = setup_logger('rag_service', 'rag.log')

class RAGService:
    def __init__(self):
        # Initialize ChromaDB
        self.persist_directory = os.path.join(os.getcwd(), 'chroma_db')
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)
            
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialization of embedding model is deferred to lazy load
        self.model = None
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="study_materials",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Initialized RAGService and ChromaDB collection")

    def get_model(self):
        """Lazy load the sentence transformer model to speed up server startup."""
        if self.model is None:
            logger.info("Lazy loading embedding model: all-MiniLM-L6-v2 on CPU")
            # Import inside method to completely prevent loading during module import
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        return self.model

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        if not text:
            return chunks
            
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
            
        return chunks

    def index_document(self, text: str, doc_id: str, metadata: Dict[str, Any] = None):
        """Chunk text, generate embeddings, and store in ChromaDB."""
        try:
            logger.info(f"Indexing document {doc_id} (size: {len(text)})")
            
            chunks = self._chunk_text(text)
            if not chunks:
                return

            # Generate embeddings for all chunks Using lazy loaded model
            model = self.get_model()
            embeddings = model.encode(chunks).tolist()
            
            # Prepare IDs and metadatas
            ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
            metadatas = [metadata or {} for _ in range(len(chunks))]
            for m in metadatas:
                m['doc_id'] = doc_id
            
            # Upsert into ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas
            )
            logger.info(f"Successfully indexed {len(chunks)} chunks for document {doc_id}")
            
        except Exception as e:
            logger.error(f"Error indexing document {doc_id}: {str(e)}", exc_info=True)
            raise

    def query(self, query_text: str, n_results: int = 5, filter_dict: Dict[str, Any] = None) -> List[str]:
        """Retrieve relevant text chunks for a query."""
        try:
            logger.info(f"Querying RAG system for: '{query_text[:50]}...'")
            
            model = self.get_model()
            query_embedding = model.encode([query_text]).tolist()
            
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                where=filter_dict
            )
            
            # results['documents'] is a list of lists
            retrieved_chunks = results['documents'][0] if results['documents'] else []
            logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks")
            return retrieved_chunks
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {str(e)}", exc_info=True)
            return []

    def clear_document(self, doc_id: str):
        """Remove chunks belonging to a specific document."""
        try:
            self.collection.delete(where={"doc_id": doc_id})
            logger.info(f"Cleared chunks for document {doc_id}")
        except Exception as e:
            logger.error(f"Error clearing document {doc_id}: {str(e)}")

# Create a singleton instance
rag_service = RAGService()
