def test_repo_structure_minimal():
    import os

    assert os.path.isdir("app")
    assert os.path.isdir("ingest")
    assert os.path.isdir("docs")
