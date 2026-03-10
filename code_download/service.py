"""Code Download Service — FastAPI application.

Serves versioned pipeline code tarballs from object storage (OBS/S3).

Endpoints:
  GET /code/download?pipeline=<name>&version=<ver>  — download code.tar.gz
  GET /code/check?pipeline=<name>&version=<ver>      — check local cache status
  GET /v1/health                                     — health check
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="Pipeline Code Download Service", version="1.0.0")

# ---------------------------------------------------------------------------
# Configuration (via environment variables)
# ---------------------------------------------------------------------------
OBS_BASE_URL = os.environ.get(
    "OBS_BASE_URL",
    "https://pipeline-release-bucket.obs.example.com",
)
CACHE_DIR = Path(os.environ.get("CODE_CACHE_DIR", "/tmp/pipeline-cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class CacheStatus(BaseModel):
    pipeline: str
    version: str
    cached: bool
    cache_path: str | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/code/check", response_model=CacheStatus)
def check_cache(
    pipeline: str = Query(..., description="Pipeline name"),
    version: str = Query(..., description="Pipeline version"),
) -> CacheStatus:
    cache_path = _cache_path(pipeline, version)
    exists = cache_path.exists()
    return CacheStatus(
        pipeline=pipeline,
        version=version,
        cached=exists,
        cache_path=str(cache_path) if exists else None,
    )


@app.get("/code/download")
def download_code(
    pipeline: str = Query(..., description="Pipeline name"),
    version: str = Query(..., description="Pipeline version"),
) -> FileResponse:
    """Download (and cache) the code.tar.gz for the given pipeline version."""
    cache_path = _cache_path(pipeline, version)

    if not cache_path.exists():
        logger.info("Cache miss: fetching %s %s from OBS", pipeline, version)
        _fetch_from_obs(pipeline, version, cache_path)
    else:
        logger.info("Cache hit: serving %s %s from %s", pipeline, version, cache_path)

    return FileResponse(
        path=str(cache_path),
        media_type="application/gzip",
        filename=f"{pipeline}-{version}.tar.gz",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cache_path(pipeline: str, version: str) -> Path:
    return CACHE_DIR / pipeline / version / "code.tar.gz"


def _fetch_from_obs(pipeline: str, version: str, dest: Path) -> None:
    """Download code.tar.gz + manifest.json from OBS and verify checksum."""
    manifest_url = f"{OBS_BASE_URL}/{pipeline}/{version}/manifest.json"
    archive_url = f"{OBS_BASE_URL}/{pipeline}/{version}/code.tar.gz"

    dest.parent.mkdir(parents=True, exist_ok=True)

    # Fetch manifest first to get expected checksum
    try:
        manifest_resp = httpx.get(manifest_url, timeout=30)
        manifest_resp.raise_for_status()
        manifest = manifest_resp.json()
        expected_sha256: str | None = manifest.get("sha256")
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch manifest for {pipeline}@{version}: {exc}",
        ) from exc

    # Download archive with streaming + temp file (atomic write)
    tmp = Path(tempfile.mktemp(dir=dest.parent, suffix=".tmp"))
    sha256 = hashlib.sha256()
    try:
        with httpx.stream("GET", archive_url, timeout=300) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    f.write(chunk)
                    sha256.update(chunk)
    except httpx.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise HTTPException(
            status_code=502,
            detail=f"Failed to download archive for {pipeline}@{version}: {exc}",
        ) from exc

    # Verify checksum
    actual_sha256 = sha256.hexdigest()
    if expected_sha256 and actual_sha256 != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise HTTPException(
            status_code=502,
            detail=(
                f"Checksum mismatch for {pipeline}@{version}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            ),
        )

    # Atomic rename
    shutil.move(str(tmp), str(dest))
    logger.info(
        "Cached %s@%s at %s (sha256=%s)", pipeline, version, dest, actual_sha256
    )
