# =============================================================================
# ESTIN RAG System - Main Source Package
# =============================================================================
# 
# 💡 WHY __init__.py?
# This file marks a directory as a Python "package", allowing you to:
# 1. Import modules using dot notation: from src.rag import chain
# 2. Control what gets exported when someone imports the package
# 3. Run initialization code when the package is imported
#
# In Python 3.3+, __init__.py is technically optional (implicit namespace 
# packages), but it's still best practice to include it for clarity.
# =============================================================================

__version__ = "0.1.0"

#make src as package       
from . import api
from . import config
from . import embeddings
from . import rag
from . import vectorstore
from . import data_processing

__all__ = [
    "api",
    "config",
    "embeddings",
    "rag",
    "vectorstore",
    "data_processing",
]

