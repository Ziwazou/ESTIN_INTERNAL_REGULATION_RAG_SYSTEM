
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
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
    
    print("=" * 60)
    print("🔨 Building ESTIN RAG Vector Store Index")
    print("=" * 60)
    
    # Step 1: Load documents
    print("\n📄 Step 1: Loading documents...")
    documents = load_estin_regulations(str(pdf_path))
    print(f"✅ Loaded {len(documents)} pages")
    
    # Step 2: Chunk by articles
    print("\n✂️  Step 2: Chunking documents by articles...")
    chunks = chunk_by_articles(documents, include_section_context=True)
    print(f"✅ Created {len(chunks)} article chunks")
    
    # Step 3: Initialize embeddings
    print("\n🧠 Step 3: Initializing embeddings model...")
    embeddings = get_embeddings(api_key=settings.hf_api_key)
    
    # Step 4: Create or reset index
    index_name = settings.pinecone_index_name
    
    if reset:
        print(f"\n🗑️  Step 4: Resetting index '{index_name}'...")
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
    
    print("\n" + "=" * 60)
    print("✅ Index build completed successfully!")
    print("=" * 60)
    
    return vector_store


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build vector store index")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing index before creating new one"
    )
    
    args = parser.parse_args()
    
    try:
        build_index(reset=args.reset)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

