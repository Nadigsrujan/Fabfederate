"""Portable, finite JSON provenance and hashes."""

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, allow_nan=False, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def source_snapshot(root: str | Path) -> dict:
    """Identify executed project code independently of an uncommitted Git revision."""
    root = Path(root)
    paths = [p for folder in ("src",) for p in (root / folder).rglob("*.py")]
    paths += [root / name for name in ("pyproject.toml", "requirements.txt") if (root / name).is_file()]
    files = {str(path.relative_to(root)): file_hash(path) for path in sorted(paths)}
    return {"sha256": stable_hash(files), "files": files}
