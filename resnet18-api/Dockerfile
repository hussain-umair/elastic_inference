FROM python:3.11-slim

WORKDIR /app

# Install only runtime essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt

# Copy application
COPY model_server.py .

EXPOSE 8001

# Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
#     CMD python -c "import requests; requests.get('http://localhost:8001/health')" || exit 1

# Run app with limited threads for CPU
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
CMD ["python", "model_server.py"]
