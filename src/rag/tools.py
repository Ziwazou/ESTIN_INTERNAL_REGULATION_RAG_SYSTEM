"""
RAG Tools module - Retrieval tools for the ESTIN regulations agent.

In LangChain v1.0+, RAG is implemented as an agent with tools.
The retrieval tool wraps the vector store search functionality,
allowing the agent to decide when and how to retrieve information.
"""

from typing import List, Tuple
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore


def create_retrieval_tool(
    vector_store: PineconeVectorStore,
    k: int = 4,
):
    """
    Create a retrieval tool that searches the ESTIN regulations.
    
    Args:
        vector_store: The Pinecone vector store with indexed documents
        k: Number of documents to retrieve per query
        
    Returns:
        A tool function that can be used by the agent
    """
    
    @tool(response_format="content_and_artifact")
    def retrieve_estin_regulations(query: str) -> Tuple[str, List[Document]]:
        """
        Rechercher des informations dans le règlement intérieur de l'ESTIN.
        
        Utilisez cet outil pour trouver des articles et dispositions spécifiques
        concernant les règles de l'école, les obligations des enseignants,
        le régime disciplinaire, l'hygiène et sécurité, etc.
        
        Args:
            query: La question ou le sujet à rechercher (en français)
            
        Returns:
            Les articles pertinents du règlement intérieur avec leurs sources
        """
        # Search for relevant documents
        retrieved_docs = vector_store.similarity_search(query, k=k)
        
        # Format the results with source information
        serialized = "\n\n---\n\n".join(
            _format_document(doc, i) for i, doc in enumerate(retrieved_docs, 1)
        )
        
        # Add header
        result = f"📚 {len(retrieved_docs)} articles trouvés dans le règlement intérieur:\n\n{serialized}"
        
        return result, retrieved_docs
    
    return retrieve_estin_regulations


def _format_document(doc: Document, index: int) -> str:
    """
    Format a document for display with metadata.
    
    Args:
        doc: The document to format
        index: The result number
        
    Returns:
        Formatted string with source and content
    """
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


