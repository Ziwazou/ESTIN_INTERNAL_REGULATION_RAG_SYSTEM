from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
  
    groq_api_key: str 
    
    hf_api_key: str

    pinecone_api_key: str
    
    # Vector Store Configuration (Pinecone)
    pinecone_index_name: str = "estin-regulations"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    
    # Application Settings
    environment: str = "development"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    log_level: str = "INFO"
    
    
    # Pydantic Settings Configuration
    model_config = SettingsConfigDict(
        # 💡 This tells Pydantic where to find the .env file
        env_file=".env",
        # 💡 .env values override environment variables
        env_file_encoding="utf-8",
        # 💡 Makes field names case-insensitive for env vars
        case_sensitive=False,
    )



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

