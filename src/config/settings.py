"""
=============================================================================
ESTIN RAG System - Configuration Settings
=============================================================================

💡 WHAT IS THIS FILE?
This file centralizes all configuration for the application using Pydantic 
Settings. It reads from environment variables and provides type-safe access 
to configuration values throughout the codebase.

💡 WHY PYDANTIC SETTINGS?
1. Type Safety: All settings have defined types (str, int, etc.)
2. Validation: Invalid values are caught at startup, not runtime
3. Documentation: Each setting has a description
4. Environment Variables: Automatically reads from .env files
5. Default Values: Sensible defaults when env vars are missing

💡 HOW IT WORKS:
When you create a Settings() instance:
1. Pydantic looks for a .env file
2. Reads environment variables
3. Validates and converts types
4. Raises clear errors if something is wrong

Example:
    from src.config.settings import get_settings
    settings = get_settings()
    print(settings.openai_api_key)  # Your API key
=============================================================================
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    💡 NAMING CONVENTION:
    The field name in Python (e.g., openai_api_key) maps to the environment
    variable name in UPPERCASE (OPENAI_API_KEY). Pydantic handles this 
    automatically.
    """
    
    # -------------------------------------------------------------------------
    # OpenAI Configuration
    # -------------------------------------------------------------------------
    openai_api_key: str = "your_openai_api_key_here"
    openai_model: str = "gpt-4-turbo"
    openai_embedding_model: str = "text-embedding-3-small"
    
    # -------------------------------------------------------------------------
    # Vector Store Configuration
    # -------------------------------------------------------------------------
    chroma_persist_directory: str = "./data/chroma_db"
    chroma_collection_name: str = "estin_regulations"
    
    # -------------------------------------------------------------------------
    # Application Settings
    # -------------------------------------------------------------------------
    environment: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    
    # -------------------------------------------------------------------------
    # Document Processing Settings
    # -------------------------------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # -------------------------------------------------------------------------
    # Pydantic Settings Configuration
    # -------------------------------------------------------------------------
    model_config = SettingsConfigDict(
        # 💡 This tells Pydantic where to find the .env file
        env_file=".env",
        # 💡 .env values override environment variables
        env_file_encoding="utf-8",
        # 💡 Makes field names case-insensitive for env vars
        case_sensitive=False,
        # 💡 Allows extra fields (won't raise error for unknown env vars)
        extra="ignore"
    )


# =============================================================================
# Settings Singleton Pattern
# =============================================================================
# 
# 💡 WHAT IS @lru_cache?
# lru_cache is a decorator that caches function results. The first time 
# get_settings() is called, it creates a Settings instance. All subsequent 
# calls return the same cached instance.
#
# 💡 WHY USE THIS PATTERN?
# 1. Performance: Settings are only loaded once
# 2. Consistency: All parts of the app use the same settings instance
# 3. Memory: No duplicate Settings objects
#
# This is called the "Singleton Pattern" - ensuring only one instance exists.
# =============================================================================

@lru_cache()
def get_settings() -> Settings:
    """
    Get the application settings (cached singleton).
    
    Returns:
        Settings: The application configuration.
        
    Example:
        settings = get_settings()
        print(f"Running in {settings.environment} mode")
    """
    return Settings()

