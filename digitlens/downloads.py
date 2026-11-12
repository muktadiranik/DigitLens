"""Fetch public model and training files with checksum verification."""

import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


def fetch(url: str, destination: Path, checksum: str) -> Path:
    """Get a data file, verify SHA-256 and cache the validated bytes."""
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == checksum:
        return destination

    print(f"Downloading {destination.name}", flush=True)
    with urlopen(url, timeout=45) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != checksum:
        raise ValueError(f"SHA-256 verification failed for {url}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        path = Path(temporary.name)
    try:
        path.write_bytes(content)
        path.replace(destination)
    finally:
        path.unlink(missing_ok=True)
    return destination
