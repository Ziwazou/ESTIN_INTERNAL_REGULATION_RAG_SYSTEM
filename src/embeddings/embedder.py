"""
Embeddings module using HuggingFace Inference API with intfloat/multilingual-e5-large model.

This model is specifically chosen because:
- Multilingual: Excellent for French documents (ESTIN regulations)
- E5 architecture: State-of-the-art for retrieval tasks
- Large variant: Better accuracy than base/small versions
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings

def get_embeddings(
    api_key: str,
    model_name: str = "intfloat/multilingual-e5-large",
) -> HuggingFaceEndpointEmbeddings:
    """
    Initialize the HuggingFace embeddings model via Inference API.
    
    Args:
        api_key: HuggingFace API token (get from https://huggingface.co/settings/tokens)
        model_name: The embedding model to use (must support feature-extraction task)
        
    Returns:
        HuggingFaceEndpointEmbeddings instance
    """
    embeddings = HuggingFaceEndpointEmbeddings(
        model=model_name,
        task="feature-extraction",
        huggingfacehub_api_token=api_key,
    )
    
    print(f"✅ Initialized embeddings model: {model_name}")
    
    return embeddings


def embed_documents(
    embeddings: HuggingFaceEndpointEmbeddings,
    documents: List[Document],
) -> List[List[float]]:
    """
    Embed a list of documents.
    
    Args:
        embeddings: The embeddings model
        documents: List of Document objects to embed
        
    Returns:
        List of embedding vectors
    """
    texts = [doc.page_content for doc in documents]
    
    print(f"🔄 Embedding {len(texts)} documents...")
    vectors = embeddings.embed_documents(texts)
    print(f"✅ Created {len(vectors)} embeddings (dimension: {len(vectors[0])})")
    
    return vectors


