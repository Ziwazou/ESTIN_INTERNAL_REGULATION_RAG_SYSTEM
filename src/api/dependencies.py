"""
API Dependencies - Dependency injection for FastAPI.

This module manages the lifecycle of expensive resources like:
- Embeddings model
- Vector store connection
- RAG Agent

Using dependency injection ensures these are created once and reused.
"""

from functools import lru_cache

from src.config.settings import get_settings
from src.embeddings import get_embeddings
from src.vectorstore import load_vector_store
from src.rag import create_estin_agent


# Global instances (initialized on first request)
_vector_store = None
_agent = None


@lru_cache()
def get_embeddings_instance():
    """Get or create the embeddings model (singleton)."""
    settings = get_settings()
    return get_embeddings(api_key=settings.hf_api_key)


def get_vector_store_instance():
    """Get or create the vector store connection (singleton)."""
    global _vector_store
    if _vector_store is None:
        settings = get_settings()
        embeddings = get_embeddings_instance()
        _vector_store = load_vector_store(
            embeddings=embeddings,
            pinecone_api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
        )
    return _vector_store


def get_agent_instance():
    """Get or create the RAG agent (singleton)."""
    global _agent
    if _agent is None:
        settings = get_settings()
        vector_store = get_vector_store_instance()
        _agent = create_estin_agent(
            groq_api_key=settings.groq_api_key,
            vector_store=vector_store,
        )
    return _agent

