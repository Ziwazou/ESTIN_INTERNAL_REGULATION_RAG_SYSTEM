import sys
from pathlib import Path

# Add project root to path before importing src
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_settings
from src.data_processing import load_estin_regulations, chunk_by_articles
from src.embeddings import get_embeddings
from src.vectorstore import create_vector_store, delete_index

def build_index(reset: bool = False):
    """Build the vector store index from PDF documents."""
    settings = get_settings()
    
    pdf_path = project_root / "data" / "documents" / "Reglement-interieur-ESTIN.pdf"
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    
    # Step 1: Load documents
    documents = load_estin_regulations(str(pdf_path))
    print(f"✅ Loaded {len(documents)} pages")
    
    # Step 2: Chunk by articles
    chunks = chunk_by_articles(documents, include_section_context=True)
    print(f"Created {len(chunks)} article chunks")
    
    # Step 3: Initialize embeddings
    embeddings = get_embeddings(api_key=settings.hf_api_key)
    print(f"embeddings initialized")

    # Step 4: Create or reset index
    index_name = settings.pinecone_index_name
    
    if reset:
        print(f"\n Step 4: Resetting index '{index_name}'...")
        delete_index(
            pinecone_api_key=settings.pinecone_api_key,
            index_name=index_name
        )
    
    # Step 5: Create vector store
    print(f"\n🗄️  Step 5: Creating vector store...")
    vector_store = create_vector_store(
        documents=chunks,
        embeddings=embeddings,
        pinecone_api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
    )
    
    print("✅ Index build completed successfully!")
    
    return vector_store


if __name__ == "__main__":
    
 
    try:
        build_index(reset=True)
    except Exception as e:
        print(f"\n Error: {e}", file=sys.stderr)
        sys.exit(1)

