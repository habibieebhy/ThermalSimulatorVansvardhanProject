# Upgrade order from v1.2.x

1. Back up `.env`, `data/`, `outputs/`, and `artifacts/`.
2. Extract the v1.3 patch into the project root.
3. Run `bash APPLY_PATCH.sh`.
4. Install Tesseract and Playwright Chromium.
5. Add Jina, Firecrawl, OpenAI, Neon, MinIO, Redis/Celery settings to `.env`.
6. Run connection checks.
7. Run a small collection with 10–20 pages and 5–10 vision assets.
8. Review Discovery, Crawl, Acquisition, Assets, Recognition, and Observations sheets.
9. Increase page/asset budgets only after the small run is clean.
10. Run deterministic research after the evidence workbook is reviewed.

Old SQLite runs remain readable. New asset/OCR fields default to empty during payload migration.
