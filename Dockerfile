# Multi-stage: backend + frontend
FROM python:3.11-slim

WORKDIR /app

# Copy backend + frontend
COPY backend/ backend/
COPY frontend/ frontend/

# Install backend deps
RUN pip install --no-cache-dir -r backend/requirements.txt

# Install frontend deps
RUN pip install --no-cache-dir -r frontend/requirements.txt

# Expose backend + frontend ports
EXPOSE 8001 8501

CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port 8001"]
