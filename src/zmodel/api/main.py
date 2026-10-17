"""Main application entry point for running Z-ModelPilot REST API with Uvicorn."""

import uvicorn
from zmodel.api.app import create_app
from zmodel.api.config import get_api_settings

app = create_app()

if __name__ == "__main__":
    settings = get_api_settings()
    uvicorn.run(
        "zmodel.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
