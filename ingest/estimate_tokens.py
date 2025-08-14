# ingest/estimate_tokens.py
import sys, pathlib
from pypdf import PdfReader
import tiktoken


def is_probably_pdf(path: pathlib.Path) -> bool:
    try:
        with open(path, "rb") as f:
            sig = f.read(4)
        # PDF imzası %PDF -> 0x25 0x50 0x44 0x46
        return sig == b"%PDF"
    except Exception:
        return False


def pdf_tokens(path: str) -> int:
    text = ""
    for p in PdfReader(path).pages:
        t = p.extract_text() or ""
        text += t + "\n"
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


if __name__ == "__main__":
    base = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "docs")
    total = 0
    skipped = []
    for pdf in base.rglob("*.pdf"):
        if not is_probably_pdf(pdf):
            skipped.append((str(pdf), "invalid header"))
            continue
        try:
            n = pdf_tokens(str(pdf))
            print(f"{pdf}: ~{n} tokens")
            total += n
        except Exception as e:
            skipped.append((str(pdf), f"{type(e).__name__}: {e}"))
            continue

    print(f"TOTAL ~{total} tokens")
    print(
        f"EST_EMBED_COST (text-embedding-3-small @ $0.02/M) ≈ ${total/1_000_000*0.02:.2f}"
    )
    if skipped:
        print("\nSkipped files:")
        for name, reason in skipped:
            print(f" - {name}  [{reason}]")
