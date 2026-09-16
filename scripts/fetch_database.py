#!/usr/bin/env python3
"""Fetch the pinned public retail DuckDB asset without requiring Git LFS.

An existing database is never replaced unless it is the exact expected LFS
pointer. Downloads are verified in a same-directory temporary file before the
atomic replacement, and the destination is checked again immediately before it.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import urllib.request


REVISION = "a3278ad102829a6084dde086244a0ef665a8011c"
EXPECTED_SIZE = 488_910_848
EXPECTED_SHA256 = "bd2bb1b3a7cf62aa94c191527aa30db5bf415aea97db718a1e0c11c59fc2ec2d"
URL = (
    "https://media.githubusercontent.com/media/Snowflake-Labs/data-eng-bench/"
    f"{REVISION}/base-image/database/retail.duckdb"
)
DEFAULT_DESTINATION = (
    Path(__file__).resolve().parents[1]
    / "vendor/data-eng-bench/base-image/database/retail.duckdb"
)
POINTER_LINES = [
    "version https://git-lfs.github.com/spec/v1",
    f"oid sha256:{EXPECTED_SHA256}",
    f"size {EXPECTED_SIZE}",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def destination_status(path: Path) -> tuple[str, bytes | None]:
    if path.is_symlink():
        raise RuntimeError(f"Refusing a symlink destination: {path}")
    if not path.exists():
        return "missing", None
    if not path.is_file():
        raise RuntimeError(f"Destination is not a regular file: {path}")
    size = path.stat().st_size
    if size == EXPECTED_SIZE and sha256_file(path) == EXPECTED_SHA256:
        return "verified", None
    if size <= 1024:
        data = path.read_bytes()
        try:
            if data.decode("ascii").strip().splitlines() == POINTER_LINES:
                return "pointer", data
        except UnicodeDecodeError:
            pass
    raise RuntimeError(
        f"Refusing to overwrite existing file with unrecognized contents: {path}"
    )


def fetch(destination: Path) -> None:
    destination = destination.absolute()
    initial_status = destination_status(destination)
    if initial_status[0] == "verified":
        print(f"Already verified: {destination}")
        print(f"size={EXPECTED_SIZE} sha256={EXPECTED_SHA256}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        request = urllib.request.Request(
            URL, headers={"User-Agent": "shared-world-gepa-dataset-fetch/1.0"}
        )
        digest = hashlib.sha256()
        received = 0
        last_progress = 0
        with urllib.request.urlopen(request, timeout=60) as response:
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) != EXPECTED_SIZE:
                raise RuntimeError(f"Unexpected Content-Length: {content_length}")
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=".retail.duckdb.", suffix=".part",
                dir=destination.parent, delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                while chunk := response.read(4 * 1024 * 1024):
                    received += len(chunk)
                    if received > EXPECTED_SIZE:
                        raise RuntimeError("Response exceeds the expected database size")
                    digest.update(chunk)
                    handle.write(chunk)
                    if received - last_progress >= 64 * 1024 * 1024:
                        print(
                            f"Downloaded {received:,}/{EXPECTED_SIZE:,} bytes",
                            file=sys.stderr, flush=True,
                        )
                        last_progress = received
                handle.flush()
                os.fsync(handle.fileno())

        if received != EXPECTED_SIZE or digest.hexdigest() != EXPECTED_SHA256:
            raise RuntimeError(
                f"Database verification failed: size={received} "
                f"sha256={digest.hexdigest()}"
            )
        final_status = destination_status(destination)
        if final_status[0] == "verified":
            print(f"Another process already installed the verified file: {destination}")
            return
        if final_status != initial_status:
            raise RuntimeError("Destination changed during download; refusing replacement")
        os.replace(temporary_path, destination)
        temporary_path = None
        print(f"Verified and installed: {destination}")
        print(f"size={received} sha256={digest.hexdigest()}")
        print(f"source_revision={REVISION}")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    try:
        fetch(args.destination)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Database fetch failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
