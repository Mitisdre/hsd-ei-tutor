# HSD-EI Tutor (DE) — RAG MVP

[![CI](https://github.com/Mtisdre/hsd-ei-tutor/actions/workflows/ci.yml/badge.svg)](https://github.com/Mtisdre/hsd-ei-tutor/actions)
![License](https://img.shields.io/badge/license-MIT-informational)

A German-language tutor for the **HSD EI** courses (Info 1–4), built with **Streamlit + RAG**.  
Features: **Q&A with citations**, **Quiz generator**, **Hybrid ranking** (Embeddings + RapidFuzz), **OCR fallback** for scanned PDFs, and **CI** (pytest + black).

## 🚀 Quickstart

```bash
# 1) Clone & set up
git clone https://github.com/Mtisdre/hsd-ei-tutor.git
cd hsd-ei-tutor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Configure env (copy & edit with your key)
cp .env.example .env
# set: OPENAI_API_KEY=..., USE_FAKE_EMBEDDINGS=0

# 3) Ingest your PDFs
python -m ingest.pdf_ingest docs/

# 4) Run the app
streamlit run app/main.py
