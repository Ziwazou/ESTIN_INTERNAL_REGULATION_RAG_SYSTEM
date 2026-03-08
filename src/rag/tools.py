from typing import List, Tuple
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore


def create_retrieval_tool(
    vector_store: PineconeVectorStore,
    k: int = 4,
):
    
    
    @tool(response_format="content_and_artifact")
    def retrieve_estin_regulations(query: str) -> Tuple[str, List[Document]]:
        """Recherche dans le règlement intérieur ESTIN les articles pertinents pour une question donnée (en français)."""
        # Search for relevant documents
        retrieved_docs = vector_store.similarity_search(query, k=k)
        
        # Format the results with source information
        serialized = "\n\n---\n\n".join(
            _format_document(doc, i) for i, doc in enumerate(retrieved_docs, 1)
        )
        
        # Add header
        result = f"{len(retrieved_docs)} articles trouvés dans le règlement intérieur:\n\n{serialized}"
        
        return result, retrieved_docs
    
    return retrieve_estin_regulations


def _format_document(doc: Document, index: int) -> str:
    metadata = doc.metadata
    
    # Extract relevant metadata
    article_num = metadata.get("article_number", "N/A")
    section_num = metadata.get("section_number", "")
    section_title = metadata.get("section_title", "")
    subsection_num = metadata.get("subsection_number", "")
    subsection_title = metadata.get("subsection_title", "")
    
    # Build source string
    source_parts = []
    if section_num and section_title:
        source_parts.append(f"Section {section_num}: {section_title}")
    if subsection_num and subsection_title:
        source_parts.append(f"Sous-section {subsection_num}: {subsection_title}")
    
    source_str = " > ".join(source_parts) if source_parts else "Règlement Intérieur ESTIN"
    
    # Format output
    formatted = f"**[Résultat {index}]**\n"
    formatted += f"📍 Source: {source_str}\n"
    if article_num != "N/A":
        formatted += f"📄 Article: {article_num}\n"
    formatted += f"\n{doc.page_content}"
    
    return formatted


