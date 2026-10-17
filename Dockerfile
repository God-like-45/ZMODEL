# ==============================================================================
# Z-ModelPilot — Production-Hardened Multi-Service Container
# ==============================================================================
FROM python:3.11-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root user and group
RUN groupadd -g 1000 zmodel && \
    useradd -u 1000 -g zmodel -s /bin/bash -m zmodel

WORKDIR /app

# Upgrade pip and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and configuration
COPY src/ ./src/
COPY profiles_data/ ./profiles_data/
COPY sample_models/ ./sample_models/
COPY alembic/ ./alembic/
COPY alembic.ini .
COPY pyproject.toml .
COPY app.py .
COPY .streamlit/ ./.streamlit/

# Set Python path and ownership
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
RUN chown -R zmodel:zmodel /app

USER zmodel

# Expose API (8000) and Streamlit (8501) ports
EXPOSE 8000 8501

# Default command starts the FastAPI REST API
CMD ["uvicorn", "zmodel.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
