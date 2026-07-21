# Upgrade order from the uploaded thermal build

1. `.env.example` and `settings.py` — OpenAI key/model and recognition threshold.
2. `llm.py` — Responses API web search, result triage, document recognition, strict schemas.
3. `search.py` — provider routing for OpenAI/Tavily/Gemini/none.
4. `crawler.py` — product-first URL priority and rejection of location/store/category SEO pages.
5. `extraction.py` — exact-product admission, bounded thickness parsing, recognition log.
6. `models.py` — recognition-log result schema and deterministic observation model.
7. `pipeline.py` — strict LLM handoff followed by deterministic algorithms.
8. `similarity.py`, `configurations.py`, `inference.py`, `graph.py` — algorithmic analysis only.
9. `storage.py` and `exporter.py` — Recognition Log, persistent corpus, Excel/SQLite output.
10. `cli.py`, `api.py`, and Streamlit pages — controls and visibility.
11. `pyproject.toml`, `VERSION`, tests, and docs — package/version/validation.
12. `APPLY_PATCH.sh` — removes the old thermal system, reinstalls, and runs all tests.
