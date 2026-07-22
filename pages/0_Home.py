"""Workflow landing page."""
from __future__ import annotations
import streamlit as st

st.title("BRIXTA Mattress Intelligence Lab v1.3")
st.caption("Distributed acquisition • explicit transcription • deterministic inference")
st.info(
    "Jina Search/Reader and Firecrawl discover and capture public evidence. Local OCR runs first; "
    "GPT-5 nano may transcribe visible product/layer information. It never performs downstream analysis."
)

st.code('''mattress-lab collect \\
  --company "The Sleep Company" \\
  --domain "https://thesleepcompany.in" \\
  --market "India" \\
  --max-pages 100 --max-external-pages 25 --max-depth 4 \\
  --llm openai --search-provider services \\
  --search --external --enqueue \\
  --output outputs/first_collection.xlsx''', language="bash")

st.subheader("Pipeline")
st.write(
    "Discovery → service/local capture → raw documents and images in MinIO → OCR/GPT transcription → "
    "atomic observations → canonical products → Neon/Postgres → deterministic graph, similarity, constraints, and Bayesian ranking."
)
