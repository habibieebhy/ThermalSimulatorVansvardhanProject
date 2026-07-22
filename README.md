# BRIXTA Mattress Intelligence v1.3.0

Distributed evidence acquisition for mattress catalogues and product pages, followed by deterministic product resolution, knowledge graphs, similarity, constraint solving, and Bayesian candidate ranking.

## Architecture

```text
Jina Search ─────────────┐
Firecrawl Search fallback├─> candidate URLs
Official sitemap ────────┘
             ↓
Firecrawl rendered capture → Jina Reader → local crawler/Playwright fallback
             ↓
HTML + embedded JSON + browser network JSON + PDFs + images
             ↓
MinIO object storage (local artifact fallback)
             ↓
Local Tesseract OCR → GPT-5 nano image/document transcription fallback
             ↓
Atomic observations and exact product candidates
             ↓
Neon/PostgreSQL (SQLite fallback)
             ↓
Deterministic entity resolution, graph, TF-IDF similarity, CP-SAT constraints,
Bayesian candidate ranking, confidence, and Excel export
```

GPT is an **evidence transcription worker only**. It may classify a page, identify an exact model, and transcribe labels visible in an image. It does not generate candidate constructions, posterior probabilities, graph conclusions, or confidence scores.

## Services

| Capability | Primary | Fallback |
|---|---|---|
| Discovery | Jina Search | Firecrawl Search, sitemap, seed URLs |
| Page capture | Firecrawl | Jina Reader, local HTTP/Playwright |
| Image discovery | Firecrawl image manifest | HTML, markdown, Playwright network responses |
| OCR | Local Tesseract | GPT-5 nano vision transcription |
| Structured database | Neon/PostgreSQL | SQLite |
| Raw evidence | MinIO | Local `artifacts/` |
| Job execution | Celery + Redis | Synchronous CLI/UI/API |
| Analysis | Deterministic code | No LLM fallback |

Free tiers are useful for prototyping but are provider-controlled quotas, not a permanent guarantee. OpenAI API usage is billed according to the account and model pricing.

## Local installation

Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m playwright install chromium
```

Install Tesseract locally:

```bash
brew install tesseract       # macOS
# or: sudo apt-get install tesseract-ocr
```

## Configure

```bash
cp .env.example .env
chmod 600 .env
```

Minimum service configuration:

```dotenv
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5-nano
JINA_API_KEY=...
FIRECRAWL_API_KEY=...
MATTRESS_INTEL_SEARCH_PROVIDER=services
MATTRESS_INTEL_CAPTURE_STRATEGY=services_first
```

Neon:

```dotenv
DATABASE_URL=postgresql://...-pooler.../neondb?sslmode=require
DATABASE_DIRECT_URL=postgresql://.../neondb?sslmode=require
```

MinIO and Celery when running locally through Docker Compose:

```dotenv
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=mattress-intelligence
CELERY_ENABLED=true
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1
```

## Validate connections

```bash
mattress-lab openai-check
mattress-lab jina-check
mattress-lab firecrawl-check
mattress-lab database-check
mattress-lab storage-check
mattress-lab doctor
```

## Run synchronously

```bash
mattress-lab collect \
  --company "The Sleep Company" \
  --domain "https://thesleepcompany.in" \
  --market "India" \
  --max-pages 100 \
  --max-external-pages 25 \
  --max-depth 4 \
  --max-assets-per-document 30 \
  --max-vision-assets 80 \
  --llm openai \
  --search-provider services \
  --capture-strategy services_first \
  --search --external \
  --output outputs/first_collection.xlsx
```

## Run through Celery

Start infrastructure and applications:

```bash
docker compose up --build
```

Or run Redis/MinIO independently and start a worker:

```bash
celery -A mattress_intelligence.celery_app:celery_app worker --loglevel=INFO --concurrency=2
```

Submit a job:

```bash
mattress-lab collect \
  --company "The Sleep Company" \
  --domain "https://thesleepcompany.in" \
  --llm openai --search-provider services \
  --search --external --enqueue
```

Check it:

```bash
mattress-lab job-status --task-id TASK_ID
```

## Catalogue support

For PDFs, the system:

1. preserves the original PDF in MinIO/local artifacts;
2. extracts text with `pypdf`;
3. renders bounded pages with PyMuPDF;
4. runs local OCR;
5. sends only selected high-value pages to GPT vision;
6. stores visible product names, table values, and top-to-bottom layer labels as observed evidence;
7. merges repeated products across pages and other sources.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The workbook includes Products, Variants, Layers, Assets, Evidence Observations, Discovery Log, Crawl Log, Acquisition Log, Recognition Log, Sources, Similar Products, Configurations, Graph Edges, Review Queue, and Run Metadata.
