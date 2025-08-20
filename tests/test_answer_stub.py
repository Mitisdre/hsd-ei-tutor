import importlib


def test_answer_returns_text_and_hits_list(tmp_path, monkeypatch):
    monkeypatch.setenv("USE_FAKE_EMBEDDINGS", "1")
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path))

    qa = importlib.import_module("app.qa")

    qa.add_documents(
        ["Testdokument"],
        [{"source": "dummy.pdf", "page_start": 1, "page_end": 1}],
        ["1"],
    )

    text, hits = qa.answer("Testfrage?")

    assert isinstance(text, str)
    assert isinstance(hits, list)
    assert len(hits) >= 1
    for hit in hits:
        assert "document" in hit
        assert "meta" in hit
        assert "score" in hit
        assert 0.0 <= hit["score"] <= 1.0
