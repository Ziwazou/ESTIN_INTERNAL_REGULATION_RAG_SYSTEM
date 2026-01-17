
from pathlib import Path
from typing import List, Optional
from pypdf import PdfReader
from langchain_core.documents import Document


def load_estin_regulations( file_path: str ) -> List[Document]:
    
    pdf_path = Path(file_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}\n")
    
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    
        
    documents = []
    
    for page_num in range(1, total_pages-1):

        page = reader.pages[page_num]
        text = page.extract_text()
        doc = Document(
            page_content=text,
            metadata={
                "source": str(pdf_path),
                "file_name": pdf_path.name,
                "page": page_num + 1 ,  
            }
        )
        documents.append(doc)
        
    return documents


def get_full_text(documents: List[Document]) -> str:
    full_text = ""
    
    for doc in documents:
        page_num = doc.metadata.get("page")
        full_text += f"\n\n[PAGE {page_num}]\n\n"
        full_text += doc.page_content
    
    return full_text.strip()

