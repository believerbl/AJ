import hashlib
import logging
import warnings
from typing import Optional

# Suppress ChromaDB internal offline deserialization warnings
warnings.filterwarnings("ignore", category=UserWarning, module="chromadb")

import chromadb
from chromadb.utils import embedding_functions

import config

logger = logging.getLogger(__name__)


class RAGMemory:
    """
    Persistent long-term memory for Augmented Jackdaw.

    Uses ChromaDB as the vector store and all-MiniLM-L6-v2 (CPU-friendly,
    ~80 MB) as the embedding model. Memories persist across restarts because
    PersistentClient writes to disk at config.CHROMA_DB_DIR.

    Design: "Conscious Memory" — AJ explicitly chooses what to store by
    calling the save_memory tool. Nothing is logged passively, so the
    database stays clean and signal-rich rather than full of noise.
    """

    COLLECTION_NAME = "aj_memory"

    def __init__(self):
        self.collection = None
        self._init_db()

    def _init_db(self):
        try:
            client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)

            # all-MiniLM-L6-v2 runs entirely on CPU, uses ~80 MB RAM,
            # and produces 384-dim embeddings - perfect for a local agent.
            embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=config.EMBEDDING_MODEL_NAME
            )

            self.collection = client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=embedding_fn,
                # cosine distance is better than L2 for semantic similarity
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"RAGMemory ready — {self.collection.count()} memories "
                f"loaded from {config.CHROMA_DB_DIR}"
            )
        except Exception as e:
            logger.error(f"RAGMemory failed to initialise: {e}")
            self.collection = None

    # ------------------------------------------------------------------
    # Public API (called by orchestrator tool wrappers)
    # ------------------------------------------------------------------

    def add_memory(self, text: str) -> str:
        """
        Store a fact or observation that AJ has explicitly decided to remember.

        IDs are content-addressed (MD5 of text) so saving the same fact
        twice is a silent no-op rather than a duplicate entry.

        Returns a short status string that gets injected back into the
        LLM's context as a tool observation.
        """
        if self.collection is None:
            return "Error: memory store is not available."

        if not text.strip():
            return "Error: cannot save an empty memory."

        doc_id = hashlib.md5(text.encode("utf-8")).hexdigest()

        try:
            # upsert so duplicate content never raises an exception
            self.collection.upsert(
                documents=[text],
                metadatas=[{"source": "aj_explicit"}],
                ids=[doc_id],
            )
            logger.info(f"[RAGMemory] saved: {text[:80]}")
            return f"Memory saved: \"{text[:80]}\""
        except Exception as e:
            logger.error(f"[RAGMemory] add_memory failed: {e}")
            return f"Error saving memory: {e}"

    def query_memory(self, query: str, k: int = 3) -> Optional[str]:
        """
        Retrieve the k most semantically relevant memories for a query.

        Returns a formatted string ready to inject into the system prompt,
        or None if the database is empty or unavailable.
        """
        if self.collection is None:
            return None

        count = self.collection.count()
        if count == 0:
            return None

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(k, count),
            )

            docs = results.get("documents", [[]])[0]
            if not docs:
                return None

            lines = ["[AJ's memory - relevant facts from past sessions]"]
            for i, doc in enumerate(docs, 1):
                lines.append(f"{i}. {doc}")

            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[RAGMemory] query_memory failed: {e}")
            return None

    def memory_count(self) -> int:
        """Returns the number of memories currently stored."""
        return self.collection.count() if self.collection else 0
