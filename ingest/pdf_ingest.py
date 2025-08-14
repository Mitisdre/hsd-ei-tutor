# ingest/pdf_ingest.py
# --- path bootstrap: allow "from app.*" when running as a script ---
import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# -------------------------------------------------------------------

import pathlib, re
from typing import List, Dict, Tuple

from pypdf import PdfReader
import tiktoken

from app.qa import add_documents

# OCR & render
OCR_ENABLE = os.getenv("OCR_ENABLE", "1") == "1"
OCR_LANGS = os.getenv("OCR_LANGS", "deu+eng")
OCR_MIN_CHARS = int(os.getenv("OCR_MIN_CHARS", "30"))
OCR_RENDER_SCALE = float(os.getenv("OCR_RENDER_SCALE", "2.0"))


def is_probably_pdf(path: pathlib.Path) -> bool:
    try:
        with open(path, "rb") as f:
            sig = f.read(4)
        return sig == b"%PDF"
    except Exception:
        return False


def clean_text(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s).strip()
    return s


def chunk_by_tokens(text: str, target: int = 1000, overlap: int = 150) -> List[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    toks = enc.encode(text)
    chunks, i = [], 0
    while i < len(toks):
        j = min(i + target, len(toks))
        chunks.append(enc.decode(toks[i:j]))
        if j == len(toks):
            break
        i = max(0, j - overlap)
    return chunks


def ocr_page(path: pathlib.Path, page_index: int) -> str:
    """Tek sayfayı img'e render edip Tesseract ile OCR yap."""
    if not OCR_ENABLE:
        return ""
    try:
        import pypdfium2 as pdfium
        import pytesseract

        # Sayfayı render et
        pdf = pdfium.PdfDocument(str(path))
        if page_index < 0 or page_index >= len(pdf):
            return ""
        page = pdf[page_index]
        pil = page.render(scale=OCR_RENDER_SCALE).to_pil()
        txt = pytesseract.image_to_string(pil, lang=OCR_LANGS)
        return clean_text(txt or "")
    except Exception as e:
        print(f"OCR skipped ({path.name} p{page_index+1}): {type(e).__name__}: {e}")
        return ""


def extract_pdf_chunks(pdf_path: pathlib.Path) -> List[Tuple[str, Dict]]:
    reader = PdfReader(str(pdf_path))
    pages = []
    # pypdf ile ilk deneme
    for pi, p in enumerate(reader.pages, 1):
        txt = clean_text(p.extract_text() or "")
        # Gerekirse OCR fallback
        if len(txt) < OCR_MIN_CHARS:
            ocr_txt = ocr_page(pdf_path, pi - 1)
            if len(ocr_txt) > len(txt):
                txt = ocr_txt
        pages.append((pi, txt))

    # sayfa-birleştirmeli chunking
    docs = []
    buf = ""
    start_page = 1
    for pi, txt in pages:
        if not txt.strip():
            continue
        if not buf:
            start_page = pi
        buf += "\n" + txt
        chs = chunk_by_tokens(buf, target=1000, overlap=150)
        for c in chs[:-1]:
            docs.append((c, {"page_start": start_page, "page_end": pi}))
        buf = chs[-1]
    if buf.strip():
        end_page = pages[-1][0] if pages else 1
        docs.append((buf, {"page_start": start_page, "page_end": end_page}))
    return docs


if __name__ == "__main__":
    base = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "docs")
    items = list(base.rglob("*.pdf"))
    if not items:
        print("No PDFs found under ./docs")
        sys.exit(0)

    docs, metas, ids = [], [], []
    skipped = []

    for pdf in items:
        if not is_probably_pdf(pdf):
            skipped.append((str(pdf), "invalid header"))
            continue
        try:
            course = pdf.parent.name
            chunks = extract_pdf_chunks(pdf)
        except Exception as e:
            skipped.append((str(pdf), f"{type(e).__name__}: {e}"))
            continue

        for idx, (txt, m) in enumerate(chunks):
            docs.append(txt)
            metas.append(
                {
                    "source": pdf.name,
                    "course": (
                        course if course.lower().startswith("info") else "Unknown"
                    ),
                    **m,
                }
            )
            ids.append(f"{pdf.stem}-{idx:05d}")
        print(f"Ingest: {pdf.name} -> {len(chunks)} chunks")

    if docs:
        add_documents(docs, metas, ids)
        print(f"Done. Added {len(ids)} chunks.")
    if skipped:
        print("\nSkipped files:")
        for name, reason in skipped:
            print(f" - {name}  [{reason}]")
