"""
ESTIN RAG API - FastAPI Application

REST API for querying the ESTIN internal regulations using RAG.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from uuid import uuid4
from contextlib import asynccontextmanager

# Add project root to path (for direct execution)
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config.settings import Settings, get_settings
from src.rag import invoke_agent, get_last_message
from src.api.dependencies import get_agent_instance


# =============================================================================
# Request/Response Models
# =============================================================================

class QuestionRequest(BaseModel):
    """Request model for asking a question."""
    question: str = Field(..., min_length=3, max_length=1000)
    thread_id: Optional[str] = None


class SourceDocument(BaseModel):
    """A source document from the retrieval."""
    content: str
    article_number: Optional[Any] = None
    section_number: Optional[str] = None
    section_title: Optional[str] = None


class AnswerResponse(BaseModel):
    """Response model for question answering."""
    answer: str
    thread_id: str
    sources: List[SourceDocument] = []


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    components: Dict[str, str]


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str
    detail: Optional[str] = None


# =============================================================================
# Application Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    This runs on startup and shutdown:
    - Startup: Initialize connections (lazy loading on first request)
    - Shutdown: Clean up resources
    """
    # Startup
    print("🚀 Starting ESTIN RAG API...")
    yield
    # Shutdown
    print("👋 Shutting down ESTIN RAG API...")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="ESTIN RAG API",
    description="""
    API pour interroger le règlement intérieur de l'ESTIN 
    (École Supérieure en Sciences et Technologies de l'Informatique et du Numérique).
    
    ## Fonctionnalités
    
    * **Question-Réponse**: Posez des questions sur le règlement et obtenez des réponses précises
    
    ## Exemple d'utilisation
    
    ```python
    import requests
    
    response = requests.post(
        "http://localhost:8000/api/v1/ask",
        json={"question": "Quelles sont les sanctions du 3ème degré?"}
    )
    print(response.json()["answer"])
    ```
    """,
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# CORS Middleware
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health Check Endpoints
# =============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Vérifier l'état du service",
)
async def health_check():
    """
    Vérifier l'état de santé de l'API et de ses composants.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        components={
            "api": "ok",
            "database": "ok",
        }
    )


@app.get(
    "/",
    tags=["Health"],
    summary="Racine de l'API",
)
async def root():
    """Point d'entrée de l'API."""
    return {
        "message": "Bienvenue sur l'API ESTIN RAG",
        "docs": "/docs",
        "health": "/health",
    }


# =============================================================================
# Question-Answering Endpoints
# =============================================================================

@app.post(
    "/api/v1/ask",
    response_model=AnswerResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Erreur serveur"}
    },
    tags=["RAG"],
    summary="Poser une question sur le règlement",
)
async def ask_question(
    request: QuestionRequest,
    settings: Settings = Depends(get_settings),
):
    """
    Poser une question sur le règlement intérieur de l'ESTIN.
    
    L'agent RAG va:
    1. Rechercher les articles pertinents
    2. Générer une réponse basée sur le contexte
    3. Citer les sources utilisées
    """
    try:
        # Get or create thread_id
        thread_id = request.thread_id or str(uuid4())
        
        # Get the agent
        agent = get_agent_instance()
        
        # Invoke the agent
        result = invoke_agent(agent, request.question, thread_id=thread_id)
        
        # Extract the answer
        answer = get_last_message(result)
        
        # Extract sources from tool calls if available
        sources = _extract_sources(result)
        
        return AnswerResponse(
            answer=answer,
            thread_id=thread_id,
            sources=sources,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement de la question: {str(e)}"
        )


# =============================================================================
# Helper Functions
# =============================================================================

def _extract_sources(result: dict) -> list:
    """
    Extract source documents from agent result.
    
    Looks for tool call artifacts that contain retrieved documents.
    """
    sources = []
    messages = result.get("messages", [])
    
    for msg in messages:
        # Check for tool messages with artifacts
        if hasattr(msg, 'artifact') and msg.artifact:
            for doc in msg.artifact:
                if hasattr(doc, 'page_content'):
                    sources.append(SourceDocument(
                        content=doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                        article_number=doc.metadata.get("article_number"),
                        section_number=doc.metadata.get("section_number"),
                        section_title=doc.metadata.get("section_title"),
                    ))
    
    return sources


# =============================================================================
# Run with Uvicorn (for development)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )

