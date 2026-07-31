# Dockerfile for ODW.ai Vault — sovereign, self-hosted RAG platform.
#
# Runtime system dependency installed here:
#   - ffmpeg: media duration detection (listed in the README as required)
#
# External services NOT bundled — run them separately and point Vault at them:
#   - Ollama (local LLM server):  https://ollama.com
#   - unar   (archive extraction): apt install unar
#   - sf     (Siegfried format ID): https://github.com/richardlehane/siegfried/releases
#
# Build:  docker build -t odw-vault .
# Run:    docker run -p 8765:8765 -v "$PWD/config.toml:/app/config.toml" odw-vault

FROM python:3.11-slim

WORKDIR /app

# System dependencies:
#   - build-essential / python3-dev: compile native Python deps (fasttext, chromadb)
#   - ffmpeg: media duration detection (README runtime dep)
#   - curl: container healthcheck against /health
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Copy the project metadata and the package directories imported by
# `vault serve` (-> uvicorn api.main:app). cli.py is the `vault` entry point.
COPY pyproject.toml README.md ./
COPY cli.py .
COPY api/ ./api/
COPY pipeline/ ./pipeline/
COPY rag/ ./rag/
COPY ui/ ./ui/
COPY eval/ ./eval/
COPY prompts/ ./prompts/
COPY seeds/ ./seeds/

# Install the project. Heavy ML dependencies are pulled from pyproject.
RUN pip install --no-cache-dir .

# Provide a default config so the container can boot. Override at runtime with
# a volume mount: -v "$PWD/config.toml:/app/config.toml"
COPY config.example.toml ./config.toml

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8765/health || exit 1

CMD ["vault", "serve", "--host", "0.0.0.0", "--port", "8765"]
