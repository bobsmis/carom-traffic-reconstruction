from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "ua_detrac_clean_intersection.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_sha256: str | None = None) -> None:
    if destination.exists() and (expected_sha256 is None or sha256(destination) == expected_sha256):
        print(f"verified: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)
    if expected_sha256 is not None and sha256(temporary) != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"checksum mismatch for {url}")
    temporary.replace(destination)
    print(f"downloaded: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    download(
        config["source_url"],
        ROOT / config["video_path"],
        config["source_sha256"],
    )
    download(
        config["detector"]["weights_url"],
        ROOT / config["detector"]["weights"],
    )


if __name__ == "__main__":
    main()
