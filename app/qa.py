# app/qa.py
import os
import hashlib
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from chromadb import PersistentClient
from chromadb.config import Settings

load_dotenv()

# --- Environment & defaults ---
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "hsd_ei")

# CI'da otomatik fake embeddings kullan
CI = os.getenv("GITHUB_ACTIONS", "false") == "true"
USE_FAKE = os.getenv("USE_FAKE_EMBEDDINGS", "1" if CI else "0") == "1"

# Hybrid rerank parametresi (0..1): 1=embedding ağırlıklı, 0=keyword ağırlıklı
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.6"))


# ---------- Embeddings ----------
def _fake_embed(texts: List[str]) -> List[List[float]]:
    """Deterministik, 64-dim sahte embedding (CI/test için)"""
    vecs = []
    for t in texts:
        h = hashlib.sha256(t.encode("utf-8")).digest()
        v = [((h[i % len(h)] / 255.0) - 0.5) for i in range(64)]
        vecs.append(v)
    return vecs


def _openai_embed(texts: List[str]) -> List[List[float]]:
    """Gerçek OpenAI embeddings"""
    from openai import OpenAI

    client = OpenAI()
    model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def embed(texts: List[str]) -> List[List[float]]:
    return _fake_embed(texts) if USE_FAKE else _openai_embed(texts)


# ---------- Chroma ----------
def get_collection():
    """Chroma koleksiyonunu döndür (telemetri kapalı)"""
    disable = os.getenv("CHROMA_DISABLE_TELEMETRY", "1") == "1"
    settings = Settings(anonymized_telemetry=not disable)
    client = PersistentClient(path=CHROMA_PATH, settings=settings)
    return client.get_or_create_collection(COLLECTION_NAME)


def add_documents(docs: List[str], metas: List[Dict[str, Any]], ids: List[str]):
    col = get_collection()
    embs = embed(docs)
    col.add(documents=docs, metadatas=metas, ids=ids, embeddings=embs)


# ---------- Hybrid query ----------
def _keyword_score(q: str, doc: str) -> float:
    """0..1 arası hızlı keyword benzerliği (RapidFuzz)"""
    try:
        from rapidfuzz.fuzz import token_set_ratio

        return token_set_ratio(q, doc) / 100.0
    except Exception:
        return 0.0


def _hybrid_rerank(
    query_text: str, hits: List[Dict[str, Any]], alpha: float
) -> List[Dict[str, Any]]:
    """Embedding benzerliği (~1-distance) + keyword skoru karışımıyla sıralama"""
    ranked = []
    for h in hits:
        dist = h.get("distance") or 0.0
        emb_sim = max(0.0, min(1.0, 1.0 - dist))  # 0..1
        kw = _keyword_score(query_text, h.get("document", ""))
        score = alpha * emb_sim + (1 - alpha) * kw
        h2 = dict(h)
        h2["score"] = score
        ranked.append(h2)
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def query(query_text: str, k: int = 6, course: str | None = None):
    """Önce embedding tabanlı geniş getir (k*2), sonra hybrid rerank ile ilk k sonucu döndür."""
    col = get_collection()
    qemb = embed([query_text])[0]
    where = {"course": course} if course else None

    res = col.query(
        query_embeddings=[qemb],
        n_results=max(k * 2, 10),  # geniş getir
        where=where,
        include=["documents", "metadatas", "distances"],  # 'ids' include EDİLMEZ
    )

    hits: List[Dict[str, Any]] = []
    if res and res.get("ids"):
        for i in range(len(res["ids"][0])):
            hits.append(
                {
                    "id": res["ids"][0][i],
                    "document": res["documents"][0][i] if res.get("documents") else "",
                    "meta": res["metadatas"][0][i] if res.get("metadatas") else {},
                    "distance": (
                        res["distances"][0][i] if res.get("distances") else None
                    ),
                }
            )

    if not hits:
        return []

    # Hybrid rerank (embedding + keyword)
    hits = _hybrid_rerank(query_text, hits, HYBRID_ALPHA)
    return hits[:k]


# ---------- Prompt & Answer ----------
def build_prompt_de(query_text: str, contexts: List[Dict[str, Any]]) -> str:
    lines = []
    for i, c in enumerate(contexts, 1):
        src = c["meta"].get("source", "Unknown")
        p_from = c["meta"].get("page_start")
        p_to = c["meta"].get("page_end")
        pages = f"S. {p_from}" if p_from == p_to else f"S. {p_from}-{p_to}"
        lines.append(f"[{i}] Quelle: {src}, {pages}\n{c['document']}\n")
    context_block = "\n---\n".join(lines) if lines else "(kein Kontext gefunden)"

    return (
        "Du bist ein Tutor für HSD EI. Antworte **ausschließlich auf Deutsch**.\n"
        "Benutze NUR den gegebenen Kontext; erfinde keine Informationen.\n"
        "Gib am Ende Quellen in der Form [Quelle: Datei.pdf, S. x–y] an. "
        "Wenn unsicher oder Kontext fehlt, erkläre, was fehlt.\n\n"
        f"FRAGE:\n{query_text}\n\n"
        f"KONTEXT:\n{context_block}\n\n"
        "ANTWORT:"
    )


def answer(
    query_text: str, course: str | None = None, k: int = 6
) -> Tuple[str, List[Dict[str, Any]]]:
    hits = query(query_text, k=k, course=course)
    prompt = build_prompt_de(query_text, hits)

    if USE_FAKE:
        # Offline/CI: Demo cevap + (varsa) 2 alıntı
        citations = ""
        if hits:
            cit = []
            for h in hits[:2]:
                src = h["meta"].get("source", "Unknown")
                p1 = h["meta"].get("page_start")
                p2 = h["meta"].get("page_end")
                pages = f"{p1}-{p2}" if (p1 and p2 and p1 != p2) else f"{p1}"
                cit.append(f"[Quelle: {src}, S. {pages}]")
            citations = "\n\n" + " ".join(cit)
        return f"(Demo-Antwort) {query_text}{citations}", hits

    from openai import OpenAI

    client = OpenAI()
    model = os.getenv("CHAT_MODEL", "gpt-5-mini")

    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
    )
    text = resp.output_text
    return text, hits


# ---------- Quiz ----------
def build_quiz_prompt_de(topic: str, contexts: List[Dict[str, Any]], n: int) -> str:
    lines = []
    for i, c in enumerate(contexts, 1):
        src = c["meta"].get("source", "Unknown")
        p_from = c["meta"].get("page_start")
        p_to = c["meta"].get("page_end")
        pages = f"S. {p_from}" if p_from == p_to else f"S. {p_from}-{p_to}"
        lines.append(f"[{i}] Quelle: {src}, {pages}\n{c['document']}\n")
    context_block = "\n---\n".join(lines) if lines else "(kein Kontext gefunden)"

    return (
        "Du bist ein Tutor für HSD EI. Erstelle prüfungsnahe Fragen **auf Deutsch**.\n"
        f"Thema: {topic}\n"
        f"Erzeuge genau {n} Fragen. Für jede Frage liefere **kurze Lösung**.\n"
        "Nutze NUR den Kontext. Am Ende liste die Quellen im Format [Quelle: Datei.pdf, S. x–y].\n"
        "Format:\n"
        "1) Frage...\n   Lösung: ...\n"
        "2) Frage...\n   Lösung: ...\n"
        "...\n"
        "\nKONTEXT:\n"
        f"{context_block}\n\nANTWORT:"
    )


def generate_quiz(
    topic: str, course: str | None = None, n: int = 5, k: int = 20
) -> Tuple[str, List[Dict[str, Any]]]:
    # Topic için geniş getir, hybrid rerank, ilk k bağlam
    hits = query(topic, k=min(k, 20), course=course)
    if USE_FAKE:
        # Basit sahte quiz (CI/offline)
        body = "\n".join(
            [f"{i+1}) Frage zu {topic}\n   Lösung: (Demo)" for i in range(n)]
        )
        cits = ""
        if hits:
            cit = []
            for h in hits[:2]:
                src = h["meta"].get("source", "Unknown")
                p1 = h["meta"].get("page_start")
                p2 = h["meta"].get("page_end")
                pages = f"{p1}-{p2}" if (p1 and p2 and p1 != p2) else f"{p1}"
                cit.append(f"[Quelle: {src}, S. {pages}]")
            cits = "\n\n" + " ".join(cit)
        return body + cits, hits

    prompt = build_quiz_prompt_de(topic, hits, n)
    from openai import OpenAI

    client = OpenAI()
    model = os.getenv("CHAT_MODEL", "gpt-5-mini")
    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
    )
    return resp.output_text, hits
