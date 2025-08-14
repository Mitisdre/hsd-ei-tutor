# HSD-EI Tutor (DE) — Streamlit + RAG

An AI tutor for HSD EI (Info 1–4). Retrieval-Augmented Generation with citations.

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your OPENAI_API_KEY
streamlit run app/main.py
