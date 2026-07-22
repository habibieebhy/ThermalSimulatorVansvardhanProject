# Architecture — BRIXTA Mattress Intelligence v1.3.0

## Trust boundary

The acquisition layer collects and transcribes public evidence. The deterministic layer performs all reasoning.

### Allowed model operations

- rank search results for evidence value;
- classify a document as exact product, catalogue, retailer, patent, guide, or irrelevant;
- transcribe explicit text and labels from HTML, PDF pages, tables, and images;
- return `null` when a value is not visible.

### Forbidden model operations

- inventing density, thickness, chemistry, layer order, or product identity;
- building candidate configurations;
- assigning Bayesian posterior or confidence;
- creating graph conclusions.

## Acquisition flow

1. Query planner generates complementary searches.
2. Jina Search runs first; Firecrawl Search augments/fills gaps.
3. Official robots/sitemaps seed a priority queue.
4. Service-first capture attempts Firecrawl, then Jina Reader, then HTTP/Playwright.
5. PDF/XML fidelity remains local HTTP.
6. Playwright records image endpoints and bounded JSON XHR/fetch payloads.
7. Documents, network JSON, images, and rendered PDF pages are content-addressed in MinIO and cached locally.
8. Local OCR runs before GPT vision.
9. Explicit observations retain source, asset, locator, excerpt, method, and confidence.

## Persistence

Neon/PostgreSQL is selected when `DATABASE_URL` exists. `DATABASE_DIRECT_URL` may be used for schema creation while the pooled URL serves runtime queries. SQLite is a complete local fallback.

MinIO is selected only when endpoint and credentials exist. Local content-addressed artifacts remain available as cache/fallback.

## Distributed jobs

Redis carries compact Celery messages and task state. Full run results are written to PostgreSQL/SQLite; large bytes are written to MinIO/local artifacts. Celery workers execute complete collection or research jobs with late acknowledgement and single-job prefetch.

## Deterministic intelligence

- Product entity resolution merges repeated exact models across sources.
- Persistent TF-IDF/cosine similarity uses current and historical products.
- CP-SAT or bounded enumeration preserves thickness and optional mass constraints.
- Bayesian-style ranking combines material priors, weight likelihood, firmness likelihood, and comparable-product evidence.
- Knowledge graph edges preserve source/product/asset/observation/layer/material/configuration relationships.
