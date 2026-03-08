
from typing import List, Optional
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings

def get_embeddings(
    api_key: str,
    model_name: str = "intfloat/multilingual-e5-large",
) -> HuggingFaceEndpointEmbeddings:
    
    embeddings = HuggingFaceEndpointEmbeddings(
        model=model_name,
        task="feature-extraction",
        huggingfacehub_api_token=api_key,
    )
    
    print(f"Initialized embeddings model: {model_name}")
    
    return embeddings


def embed_documents(
    embeddings: HuggingFaceEndpointEmbeddings,
    documents: List[Document],
) -> List[List[float]]:
    
    texts = [doc.page_content for doc in documents]
    
    vectors = embeddings.embed_documents(texts)
    
    return vectors


