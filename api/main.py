"""
Vercel serverless entrypoint.
Exposes the FastAPI app so Vercel can find it at api/main.py.
"""
import sys
from pathlib import Path

# Project root (parent of api/)
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.api.main import app
