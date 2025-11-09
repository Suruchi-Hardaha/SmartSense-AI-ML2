
FROM python:3.11-slim

WORKDIR /app


COPY backend/ backend/
COPY frontend/ frontend/


RUN pip install --no-cache-dir -r backend/requirements.txt


RUN pip install --no-cache-dir -r frontend/requirements.txt

# Expose backend + frontend ports
EXPOSE 8001 8501

CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port 8001"]
