"""Configuration management for Inner Monologue agent system."""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to use pydantic-settings, fallback to simple class
try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback if pydantic-settings not available
    from pydantic import BaseModel as BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Gemini API
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_model: str = "gemini-2.5-flash"  # Use basic gemini-pro for free tier
    gemini_temperature: float = 0.7
    gemini_max_tokens: int = 2048
    
    # PyBullet Environment
    use_gui: bool = os.getenv("USE_GUI", "true").lower() == "true"
    pybullet_assets_path: Optional[Path] = None
    
    # Agent Settings
    max_iterations: int = 3  # Reduce to limit API calls
    enable_reflection: bool = False  # Disable reflection to reduce calls
    tool_timeout: float = 30.0
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: Optional[Path] = None
    
    # Paths
    project_root: Path = Path(__file__).parent
    llm_tamp_path: Path = project_root.parent / "LLM-TAMP"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

