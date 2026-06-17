# Single image used by both the API and the dashboard services (see compose).
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so the layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the trained model artifact.
COPY src/ src/
COPY app/ app/
COPY data/ data/
COPY models/ models/

# API on 8000, Streamlit dashboard on 8501 (compose maps both).
EXPOSE 8000 8501

# Default command serves the API; the dashboard service overrides it in compose.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
