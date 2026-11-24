from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GUIDE_PATH = PROJECT_ROOT / "docs" / "model_guide.md"

st.set_page_config(
    page_title="How the Model Works",
    page_icon="📘",
    layout="wide",
)

st.markdown(
    '<div style="color:#d97706;font-size:0.78rem;font-weight:700;'
    'letter-spacing:0.12em;text-transform:uppercase;">MODEL GUIDE</div>',
    unsafe_allow_html=True,
)

st.markdown(GUIDE_PATH.read_text(encoding="utf-8"))
