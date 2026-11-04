import zipfile

import pytest

from fabfederate.data.acquisition import extract_archive


def test_archive_traversal_cannot_write_outside_destination(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../outside.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_archive(archive, tmp_path / "data")
    assert not (tmp_path / "outside.txt").exists()


def test_archive_budget_is_checked_before_extracting(tmp_path):
    archive = tmp_path / "large.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("a.txt", "123456")
    with pytest.raises(ValueError, match="budget"):
        extract_archive(archive, tmp_path / "data", max_bytes=2)


def test_normal_archive_is_extracted_and_hashed(tmp_path):
    archive = tmp_path / "normal.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("nested/source.txt", "recording")
    files = extract_archive(archive, tmp_path / "data")
    assert len(files) == 1
    assert files[0].read_text() == "recording"
