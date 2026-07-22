# Validation

## Static and unit validation

```bash
python -m compileall -q src app.py pages tests
PYTHONPATH=src python -m unittest discover -s tests -v
```

Expected: 25 tests pass.

## Local fallback smoke test

```bash
mattress-lab demo
mattress-lab doctor
```

## Service checks

```bash
mattress-lab openai-check
mattress-lab jina-check
mattress-lab firecrawl-check
mattress-lab database-check
mattress-lab storage-check
mattress-lab worker-check
```

## Acceptance criteria for a real catalogue

- original PDF and rendered pages stored with hashes;
- OCR text retained on assets;
- GPT output contains only visible labels;
- exact products separated across pages/sections;
- repeated product appearances merge without losing source provenance;
- inferred configurations remain absent in collection-only mode;
- Neon contains the run and MinIO contains large evidence objects;
- Redis task result contains only a compact summary.
