# app/qa.py
import os
import hashlib
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from chromadb import PersistentClient
from chromadb.config import Settings

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "hsd_ei")

# CI'da otomatik fake embedding
CI = os.getenv("GITHUB_ACTIONS", "false") == "true"
USE_FAKE = os.getenv("USE_FAKE_EMBEDDINGS", "1" if CI else "0") == "1"

# ---------- Embeddings ----------
def _fake_embed(texts: List[str]) -> List[List[float]]:
    vecs = []
    for t in texts:
        h = hashlib.sha256(t.encode("utf-8")).digest()
        vecs.append([((h[i % len(h)] / 255.0) - 0.5) for i in range(64)])
    return vecs

def _openai_embed(texts: List[str]) -> List[List[float]]:
    from openai import OpenAI
    client = OpenAI()
    model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]

def embed(texts: List[str]) -> List[List[float]]:
    return _fake_embed(texts) if USE_FAKE else _openai_embed(texts)

# ---------- Chroma ----------
def get_collection():
    disable = os.getenv("CHROMA_DISABLE_TELEMETRY", "1") == "1"
    settings = Settings(anonymized_telemetry=not disable)
    client = PersistentClient(path=CHROMA_PATH, settings=settings)
    return client.get_or_create_collection(COLLECTION_NAME)

def add_documents(docs: List[str], metas: List[Dict[str, Any]], ids: List[str]):
    col = get_collection()
    embs = embed(docs)
    col.add(documents=docs, metadatas=metas, ids=ids, embeddings=embs)

def query(query_text: str, k: int = 6, course: str | None = None):
    col = get_collection()
    qemb = embed([query_text])[0]
    where = {"course": course} if course else None

    # 'ids' include EDİLMEZ (Chroma 0.5 kabul etmiyor); zaten res["ids"] içinde döner.
    res = col.query(
        query_embeddings=[qemb],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits: List[Dict[str, Any]] = []
    if res and res.get("ids"):
        for i in range(len(res["ids"][0])):
            hits.append({
                "id": res["ids"][0][i],
                "document": res["documents"][0][i] if res.get("documents") else "",
                "meta": res["metadatas"][0][i] if res.get("metadatas") else {},
                "distance": res["distances"][0][i] if res.get("distances") else None,
            })
    return hits

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

def answer(query_text: str, course: str | None = None, k: int = 6) -> Tuple[str, List[Dict[str, Any]]]:
    hits = query(query_text, k=k, course=course)
    prompt = build_prompt_de(query_text, hits)

    if USE_FAKE:
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
    resp = client.responses.create(model=model, input=[{"role": "user", "content": prompt}])
    return resp.output_text, hits
