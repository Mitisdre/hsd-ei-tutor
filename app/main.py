# app/main.py
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import streamlit as st
from app.qa import answer

st.set_page_config(page_title="HSD-EI Tutor (DE)", page_icon="🎓", layout="wide")
st.title("🎓 HSD-EI Tutor (DE)")
st.caption("RAG-basiertes System mit Quellenangaben (Info 1–4) — MVP-Skelett")

with st.sidebar:
    st.header("Einstellungen")
    course = st.selectbox("Kurs", ["Alle", "Info1", "Info2", "Info3", "Info4"])
    topk = st.slider("Top-k Kontexte", 2, 10, 6)

q = st.text_input("Deine Frage (Deutsch):", placeholder="Was ist das Ohmsche Gesetz?")
if st.button("Antwort holen", type="primary") and q:
    sel_course = None if course == "Alle" else course
    text, hits = answer(q, course=sel_course, k=topk)
    st.markdown("### Antwort")
    st.write(text)
