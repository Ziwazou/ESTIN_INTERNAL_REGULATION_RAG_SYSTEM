
import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


def init_pinecone(api_key: str) -> Pinecone:
    
    pc = Pinecone(api_key=api_key)
    print("Pinecone client initialized")
    return pc


def create_index_if_not_exists(
    pc: Pinecone,
    index_name: str,
    dimension: int = 1024,  # intfloat/multilingual-e5-large dimension
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
) -> None:
   
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"🔨 Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        print(f"✅ Index '{index_name}' created successfully")
    else:
        print(f"📌 Index '{index_name}' already exists")


def create_vector_store(
    documents: List[Document],
    embeddings: Embeddings,
    pinecone_api_key: str,
    index_name: str = "estin-regulations",
) -> PineconeVectorStore:
   
    # Initialize Pinecone
    pc = init_pinecone(pinecone_api_key)
    
    # Create index if needed (1024 is the dimension for multilingual-e5-large)
    create_index_if_not_exists(pc, index_name, dimension=1024)
    
    # Get the index
    index = pc.Index(index_name)
    
    
    original_api_key = os.environ.get("PINECONE_API_KEY")
    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    
    try:
        vector_store = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=index_name,
        )
    except Exception as e:
        print(f"Error creating vector store: {e}")
    
    print(f"Vector store created successfully!")
    
    return vector_store


def load_vector_store(
    embeddings: Embeddings,
    pinecone_api_key: str,
    index_name: str = "estin-regulations",
) -> PineconeVectorStore:

    # Initialize Pinecone
    pc = init_pinecone(pinecone_api_key)
    
    # Get the index
    index = pc.Index(index_name)
    
    print(f"Loading vector store from index: {index_name}")
    
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
    )
    
    # Get index stats
    stats = index.describe_index_stats()
    total_vectors = stats.total_vector_count
    
    print(f"Loaded vector store with {total_vectors} vectors")
    
    return vector_store


def similarity_search(
    vector_store: PineconeVectorStore,
    query: str,
    k: int = 4,
) -> List[Document]:
    
    results = vector_store.similarity_search(query, k=k)
    return results


def delete_index(
    pinecone_api_key: str,
    index_name: str,
) -> None:
    
    pc = init_pinecone(pinecone_api_key)
    pc.delete_index(index_name)
    print(f"Index '{index_name}' deleted")
