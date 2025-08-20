import os

os.environ["USE_FAKE_EMBEDDINGS"] = "1"
from app.qa import answer


def test_answer_returns_text_and_hits_list():
    text, hits = answer("Testfrage?")
    assert isinstance(text, str)
    assert isinstance(hits, list)
