"""Streamlit entry point for the mattress evidence and inference platform."""

from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="BRIXTA Mattress Intelligence Lab",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

home = st.Page("pages/0_Home.py", title="Research Workflow", icon="🧭", default=True)
collection = st.Page(
    "pages/1_Evidence_Collection_Lab.py",
    title="Evidence & Algorithms Lab",
    icon="🔎",
)

navigation = st.navigation(
    {
        "Start here": [home],
        "Research tools": [collection],
    }
)
navigation.run()
