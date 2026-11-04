"""Acquire the official NASA-linked archive with integrity and bounded extraction."""

from datetime import datetime, timezone
from pathlib import Path
import stat
import zipfile

import requests

from fabfederate.utils import file_hash, write_json

NASA_CATALOGUE = (
    "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/"
    "pcoe/pcoe-data-set-repository/"
)
NASA_ARCHIVE = "https://phm-datasets.s3.amazonaws.com/NASA/3.+Milling.zip"


def extract_archive(archive: Path, destination: Path, max_bytes: int = 128_000_000) -> list[Path]:
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        if sum(item.file_size for item in members) > max_bytes:
            raise ValueError("Archive exceeds extraction budget")
        for item in members:
            target = (destination / item.filename).resolve()
            mode = item.external_attr >> 16
            if not target.is_relative_to(destination) or stat.S_ISLNK(mode):
                raise ValueError(f"Unsafe archive entry: {item.filename}")
        source.extractall(destination)
        return [destination / item.filename for item in members if not item.is_dir()]


def acquire_nasa(root: str | Path = "data/raw", force: bool = False) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "nasa_milling.zip"
    if force or not archive.exists():
        temporary = archive.with_suffix(".download")
        try:
            with requests.get(NASA_ARCHIVE, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                total = 0
                with temporary.open("wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        total += len(block)
                        if total > 128_000_000:
                            raise ValueError("Download exceeds 128 MB source budget")
                        stream.write(block)
            if not zipfile.is_zipfile(temporary):
                raise ValueError("Source response is not a ZIP archive")
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    files = extract_archive(archive, root / "nasa")
    for nested in list(files):
        if nested.suffix.lower() == ".zip":
            files.extend(extract_archive(nested, nested.parent))
    matrices = sorted(p for p in files if p.name == "mill.mat")
    if len(matrices) != 1:
        raise ValueError("Expected one audited mill.mat in NASA archive")
    manifest = {
        "dataset_id": "nasa_milling",
        "catalogue_url": NASA_CATALOGUE,
        "archive_url": NASA_ARCHIVE,
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": file_hash(archive),
        "archive_bytes": archive.stat().st_size,
        "mat_path": str(matrices[0]),
        "files": [{"path": str(p), "sha256": file_hash(p)} for p in files],
        "citation": "A. Agogino and K. Goebel (2007), BEST Lab UC Berkeley, Milling Data Set, NASA PCoE",
        "terms": "Public archive linked by NASA; no separate license file found. Preserve attribution; confirm terms before redistribution.",
    }
    write_json(root / "nasa_source_manifest.json", manifest)
    return manifest
