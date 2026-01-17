
import re
from typing import List
from langchain_core.documents import Document
from .loaders import get_full_text


def chunk_by_articles(
    documents: List[Document],
    include_section_context: bool = True,
) -> List[Document]:
    # Combine all pages using the loader utility function
    full_text = get_full_text(documents)
    
    # Get source metadata from first document
    source_metadata = documents[0].metadata if documents else {}
    
    # Parse the document structure
    chunks = _parse_document_structure(full_text, source_metadata, include_section_context)
    
    print(f"📑 Created {len(chunks)} article-based chunks")
    
    return chunks


def _parse_document_structure(
    text: str,
    base_metadata: dict,
    include_section_context: bool,
) -> List[Document]:
    
    chunks = []
    
    # Current context (tracks where we are in the document)
    current_section = ""
    current_section_num = ""
    current_subsection = ""
    current_subsection_num = ""
    
    # Patterns for parsing
    # 💡 re.MULTILINE makes ^ and $ match line boundaries
    # 💡 Case-sensitive: only matches "Article" (capital A), not "article"
    section_pattern = re.compile(r'^(\d+)\s+([A-ZÉÈÀ][A-ZÉÈÀ\s]+)$', re.MULTILINE)
    subsection_pattern = re.compile(r'^(\d+\.\d+)\s+(.+)$', re.MULTILINE)
    article_pattern = re.compile(r'^Article\s+(\d+)\s*:?\s*(.*)', re.MULTILINE)
    
    # Split text by articles
    # 💡 Case-sensitive: only matches "Article" (capital A), not "article" in text
    # 💡 ^[ ]* allows optional leading spaces (some PDFs have spaces before "Article")
    # 💡 ^ ensures "Article" is at the start of a line (with optional spaces), preventing false matches
    parts = re.split(r'(^[ ]*Article\s+\d+\s*:?)', text, flags=re.MULTILINE)
    
    # Process the text before the first article (section headers, intro)
    if parts:
        intro_text = parts[0]
        
        # Find all section/subsection updates in intro
        for match in section_pattern.finditer(intro_text):
            current_section_num = match.group(1)
            current_section = match.group(2).strip()
            
        for match in subsection_pattern.finditer(intro_text):
            current_subsection_num = match.group(1)
            current_subsection = match.group(2).strip()
    
    # Process articles
    i = 1
    while i < len(parts):
        if i + 1 < len(parts):
            article_header = parts[i]  # "Article X :" or possibly "article X"
            article_body = parts[i + 1]  # The article content
            
            # CRITICAL: Only process headers that start with "Article" (A majuscule)
            # Ignore "article" (a minuscule) which might appear in content
            if not article_header.strip().startswith('Article'):
                # Skip this split - it's not a real article header, just content
                i += 2
                continue
            
            # Extract article number
            # 💡 Case-sensitive: only matches "Article" (capital A)
            article_match = re.match(r'Article\s+(\d+)', article_header)
            article_num = article_match.group(1) if article_match else "?"
            
            # Update context from this article's text
            for match in section_pattern.finditer(article_body):
                current_section_num = match.group(1)
                current_section = match.group(2).strip()
                
            for match in subsection_pattern.finditer(article_body):
                current_subsection_num = match.group(1)
                current_subsection = match.group(2).strip()
            
            # Build the chunk content
            if include_section_context:
                context_header = _build_context_header(
                    current_section_num, current_section,
                    current_subsection_num, current_subsection
                )
                chunk_content = f"{context_header}\n\n{article_header.strip()}{article_body.strip()}"
            else:
                chunk_content = f"{article_header.strip()}{article_body.strip()}"
            
            # Clean up the content
            chunk_content = _clean_text(chunk_content)
            
            # Only create chunk if there's meaningful content
            if len(chunk_content.strip()) > 50:
                # Create metadata for this chunk
                # 💡 Metadata enables filtering: "find articles in Section 5"
                chunk_metadata = {
                    **base_metadata,
                    "article_number": int(article_num) if article_num.isdigit() else article_num,
                    "section_number": current_section_num,
                    "section_title": current_section,
                    "subsection_number": current_subsection_num,
                    "subsection_title": current_subsection,
                }
                
                chunks.append(Document(
                    page_content=chunk_content,
                    metadata=chunk_metadata
                ))
        
        i += 2
    
    return chunks


def _build_context_header(
    section_num: str,
    section_title: str,
    subsection_num: str,
    subsection_title: str,
) -> str:
    
    parts = []
    
    if section_num and section_title:
        parts.append(f"[Section {section_num}: {section_title}]")
    
    if subsection_num and subsection_title:
        parts.append(f"[Sous-section {subsection_num}: {subsection_title}]")
    
    return "\n".join(parts)


def _clean_text(text: str) -> str:
    # Remove page markers if any
    text = re.sub(r'\[PAGE \d+\]', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
    text = re.sub(r' {2,}', ' ', text)  # Max 1 space
    
    # Remove page numbers that appear alone
    text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
    
    return text.strip()


