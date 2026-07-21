FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MATTRESS_INTEL_DATA_DIR=/app/data \
    MATTRESS_INTEL_OUTPUT_DIR=/app/outputs \
    MATTRESS_INTEL_ARTIFACT_DIR=/app/artifacts \
    MATTRESS_INTEL_DATABASE_PATH=/app/data/mattress_intelligence.sqlite3

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY examples ./examples
COPY pages ./pages
COPY app.py ./app.py
COPY VERSION ./
RUN pip install --no-cache-dir ".[ui,api]"

EXPOSE 8501 8000
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
