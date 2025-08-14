# app/main.py
# --- path bootstrap (so "from app.*" works when running app/main.py) ---
import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ----------------------------------------------------------------------

import streamlit as st
from app.qa import answer, generate_quiz

st.set_page_config(page_title="HSD-EI Tutor (DE)", page_icon="🎓", layout="wide")

st.title("🎓 HSD-EI Tutor (DE)")
st.caption("RAG-basiertes System mit Quellenangaben (Info 1–4) — Q&A + Quiz")

with st.sidebar:
    st.header("Einstellungen")
    course = st.selectbox("Kurs", ["Alle", "Info1", "Info2", "Info3", "Info4"])
    topk = st.slider("Top-k Kontexte", 2, 10, 6)
    st.divider()
    st.markdown(
        "Hybrid-Ranking aktiv: Embeddings + Keyword (RapidFuzz). "
        "Für Produktionsbetrieb bitte `.env` → `HYBRID_ALPHA` anpassen."
    )

tab1, tab2 = st.tabs(["❓ Q&A", "📝 Quiz"])

with tab1:
    q = st.text_input(
        "Deine Frage (Deutsch):", placeholder="Was ist das Ohmsche Gesetz?"
    )
    if st.button("Antwort holen", type="primary") and q:
        sel_course = None if course == "Alle" else course
        with st.spinner("Denke nach..."):
            text, hits = answer(q, course=sel_course, k=topk)
        st.markdown("### Antwort")
        st.write(text)
        if hits:
            with st.expander("📎 Verwendete Kontexte / Quellen"):
                for i, h in enumerate(hits, 1):
                    meta = h["meta"]
                    src = meta.get("source", "Unknown")
                    p_from, p_to = meta.get("page_start"), meta.get("page_end")
                    pages = f"S. {p_from}" if p_from == p_to else f"S. {p_from}-{p_to}"
                    st.markdown(f"**[{i}] {src}, {pages}**")
                    st.caption(
                        h["document"][:500]
                        + ("..." if len(h["document"]) > 500 else "")
                    )

with tab2:
    col_a, col_b = st.columns([2, 1])
    with col_a:
        topic = st.text_input(
            "Thema (Deutsch):",
            placeholder="z.B. Bode-Diagramm, Ohmsches Gesetz, Rekursion ...",
        )
    with col_b:
        n_q = st.number_input(
            "Anzahl der Fragen", min_value=3, max_value=15, value=5, step=1
        )

    if st.button("Quiz erzeugen", type="secondary") and topic:
        sel_course = None if course == "Alle" else course
        with st.spinner("Generiere Quiz..."):
            quiz_md, hits = generate_quiz(
                topic, course=sel_course, n=int(n_q), k=max(topk * 2, 10)
            )
        st.markdown("### Quiz")
        st.markdown(quiz_md)
        if hits:
            with st.expander("📎 Verwendete Kontexte / Quellen"):
                for i, h in enumerate(hits, 1):
                    meta = h["meta"]
                    src = meta.get("source", "Unknown")
                    p_from, p_to = meta.get("page_start"), meta.get("page_end")
                    pages = f"S. {p_from}" if p_from == p_to else f"S. {p_from}-{p_to}"
                    st.markdown(f"**[{i}] {src}, {pages}**")
                    st.caption(
                        h["document"][:500]
                        + ("..." if len(h["document"]) > 500 else "")
                    )
