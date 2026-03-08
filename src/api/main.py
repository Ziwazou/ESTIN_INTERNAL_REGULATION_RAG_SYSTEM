import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from uuid import uuid4
from contextlib import asynccontextmanager

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.config.settings import Settings, get_settings

# Frontend directory path
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"
from src.rag import invoke_agent, get_last_message
from src.api.dependencies import get_agent_instance



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
    
    # Startup
    print("Starting ESTIN RAG API...")
    yield
    # Shutdown
    print("Shutting down ESTIN RAG API...")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="ESTIN RAG API",
    description="""
    API pour interroger le règlement intérieur de l'ESTIN 
    (École Supérieure en Sciences et Technologies de l'Informatique et du Numérique).
    
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
# Static Files (Frontend)
# =============================================================================

# Mount static files for frontend assets (CSS, JS)
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Vérifier l'état du service",
)
async def health_check():
    
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
    tags=["Frontend"],
    summary="Interface utilisateur",
    include_in_schema=False,
)
async def root():
    """Serve the frontend interface."""
    frontend_index = FRONTEND_DIR / "index.html"
    if frontend_index.exists():
        return FileResponse(str(frontend_index))
    return {
        "message": "Bienvenue sur l'API ESTIN RAG",
        "docs": "/docs",
        "health": "/health",
    }


@app.get(
    "/api",
    tags=["Health"],
    summary="Racine de l'API",
)
async def api_root():
    """Point d'entrée de l'API."""
    return {
        "message": "Bienvenue sur l'API ESTIN RAG",
        "docs": "/docs",
        "health": "/health",
    }


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


# 

def _extract_sources(result: dict) -> list:
    
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




if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )

