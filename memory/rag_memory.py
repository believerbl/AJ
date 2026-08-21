import logging
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict

import config

logger = logging.getLogger(__name__)

class RAGMemory:
    """
    Long-Term Memory using ChromaDB and a local lightweight embedding model.
    """
    def __init__(self):
        try:
            # Initialize ChromaDB client to persist data locally
            self.client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
            
            # Use all-MiniLM-L6-v2 which runs well on CPU
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=config.EMBEDDING_MODEL_NAME
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="jarvis_memory",
                embedding_function=self.embedding_fn
            )
            logger.info(f"RAG Memory initialized at {config.CHROMA_DB_DIR}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.client = None
            self.collection = None

    def store_interaction(self, user_input: str, ai_response: str):
        """
        Stores an interaction in the vector database.
        """
        if not self.collection:
            return

        try:
            # Create a combined document for embedding
            document = f"User: {user_input}\nJarvis: {ai_response}"
            
            # Generate a unique ID (hash of document or simple timestamp)
            import hashlib
            doc_id = hashlib.md5(document.encode('utf-8')).hexdigest()
            
            self.collection.add(
                documents=[document],
                metadatas=[{"type": "conversation", "user_input": user_input}],
                ids=[doc_id]
            )
            logger.debug(f"Stored interaction in RAG memory: {doc_id}")
        except Exception as e:
            logger.error(f"Failed to store interaction: {e}")

    def retrieve_context(self, query: str, n_results: int = 3) -> str:
        """
        Retrieve relevant past interactions to inject into the system prompt.
        """
        if not self.collection:
            return ""

        try:
            # If the collection is empty, it might throw an exception on query
            if self.collection.count() == 0:
                return ""
                
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count())
            )
            
            if not results['documents'] or not results['documents'][0]:
                return ""
                
            # Format retrieved context
            context_str = "Relevant Past Interactions:\n"
            for doc in results['documents'][0]:
                context_str += f"{doc}\n---\n"
                
            return context_str.strip()
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return ""
