import pytest
from pathlib import Path
from training.common import sha256_path

def test_sha256_path_directory(tmp_path: Path):
    d = tmp_path / "dataset"
    d.mkdir()
    (d / "a.txt").write_text("hello")
    (d / "b.txt").write_text("world")
    
    hash1 = sha256_path(d)
    
    d2 = tmp_path / "dataset2"
    d2.mkdir()
    (d2 / "a.txt").write_text("hello")
    (d2 / "b.txt").write_text("world")
    
    hash2 = sha256_path(d2)
    
    assert hash1 == hash2

    # Ensure changing file content changes hash
    (d2 / "b.txt").write_text("world2")
    assert hash1 != sha256_path(d2)
