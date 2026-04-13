FROM python:3.11-slim

WORKDIR /app

# Install build dependencies for native extensions (chromadb, torch).
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformer embedding model at build time
# so the container starts instantly without a download on first run.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Remove build dependencies to reduce image size.
RUN apt-get purge -y --auto-remove build-essential

# Copy application code.
COPY core/ core/
COPY ui/ ui/
COPY tests/ tests/

# Create the persistent store directory.
RUN mkdir -p /app/store

ENV STORE_PATH=/app/store
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8501

# Streamlit health check.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "ui/app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
