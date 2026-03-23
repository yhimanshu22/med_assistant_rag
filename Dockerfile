# Use a Python image with uv pre-installed for efficiency
FROM python:3.10-slim AS builder

# Install uv using the official binary
COPY --from=ghcr.io/astral-sh/uv:0.5.21 /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the filtered requirements file
COPY requirements_docker.txt ./

# Create virtual environment and install dependencies using uv pip
# This avoids the strict platform checks that uv sync performs on uv.lock
RUN uv venv /app/.venv
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
RUN uv pip install --no-cache -r requirements_docker.txt

# Copy project configuration and source
COPY pyproject.toml ./
COPY src ./src

# Install the project itself without re-installing dependencies
RUN uv pip install --no-deps .

# Final runtime stage
FROM python:3.10-slim

WORKDIR /app

# Install libgomp1 which is often required by ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy source code and necessary data folders
COPY src ./src
RUN mkdir -p data chroma_db models_cache

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose the API port
EXPOSE 8000

# Command to run the FastAPI application
CMD ["python", "-m", "med_assistant.api.main"]
