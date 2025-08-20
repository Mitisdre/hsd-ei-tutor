import os
os.environ["USE_FAKE_EMBEDDINGS"] = "1"
from app import qa


def fake_query(*args, **kwargs):
    return [{"document": "Doc", "meta": {"source": "foo.pdf"}}]


def test_answer_citation_without_pages(monkeypatch):
    monkeypatch.setattr(qa, "query", fake_query)
    text, _ = qa.answer("frage?")
    assert "[Quelle: foo.pdf]" in text
    assert "S." not in text


def test_generate_quiz_citation_without_pages(monkeypatch):
    monkeypatch.setattr(qa, "query", fake_query)
    text, _ = qa.generate_quiz("topic", n=1)
    assert "[Quelle: foo.pdf]" in text
    assert "S." not in text
