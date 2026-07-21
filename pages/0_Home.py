"""Workflow landing page."""

from __future__ import annotations

import streamlit as st


st.title("BRIXTA Mattress Intelligence Lab")
st.caption("Search and recognition first • Deterministic inference last")

st.info(
    "OpenAI may discover URLs, reject generic pages, recognize exact products, and extract "
    "published facts. It never generates the knowledge graph, configurations, Bayesian "
    "posterior, or confidence score."
)

left, middle, right = st.columns(3)
with left:
    st.subheader("1 · Search and crawl")
    st.write(
        "OpenAI, Tavily, or sitemap-only discovery supplies candidate URLs. robots.txt, sitemap "
        "parsing, URL priority, depth limits, and separate external budgets control fetching."
    )
with middle:
    st.subheader("2 · Recognize evidence")
    st.write(
        "Strict URL, name, JSON-LD, and optional model gates separate exact products from "
        "collections, location pages, stores, and guides. Rejected pages can still contribute "
        "atomic observations."
    )
with right:
    st.subheader("3 · Infer deterministically")
    st.write(
        "Persistent similarity supplies comparables. Constraint solving removes impossible "
        "stacks. Bayesian-style code ranks survivors, and the graph records provenance."
    )

st.subheader("Recommended first run")
st.code(
    '''mattress-lab collect \\
  --company "The Sleep Company" \\
  --domain "https://thesleepcompany.in" \\
  --market "India" \\
  --max-pages 100 \\
  --max-external-pages 25 \\
  --max-depth 4 \\
  --llm openai \\
  --search-provider openai \\
  --search --external \\
  --output outputs/first_collection.xlsx''',
    language="bash",
)

st.subheader("Trust boundary")
st.write(
    "Discovery and Recognition logs show model decisions. Evidence Observations are document-level "
    "matches. Observed Claims belong to admitted products. Candidate configurations remain "
    "algorithmic hypotheses, never manufacturer disclosures."
)
