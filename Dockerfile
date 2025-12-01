# ForexSwing AI 2025 - Production Docker Image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    libgomp1 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements_app.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements_app.txt

# Install additional production dependencies
RUN pip install --no-cache-dir \
    gunicorn \
    schedule \
    python-dotenv \
    google-generativeai \
    yfinance \
    alpha_vantage

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p \
    data/models \
    data/MarketData \
    data/logs \
    user_data/strategies \
    user_data/notebooks

# Set permissions
RUN chmod +x companion_api_service.py

# Expose ports
EXPOSE 8080 8082

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8082/api/status || exit 1

# Default command (can be overridden)
CMD ["python", "companion_api_service.py", "8082"]
