"""代码下载服务 — FastAPI 应用。

从对象存储（OBS/S3）提供版本化的流水线代码压缩包。

接口：
  GET /code/download?pipeline=<name>&version=<ver>  — 下载 code.tar.gz
  GET /code/check?pipeline=<name>&version=<ver>      — 查询本地缓存状态
  GET /v1/health                                     — 健康检查
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
# 配置（通过环境变量注入）
# ---------------------------------------------------------------------------
OBS_BASE_URL = os.environ.get(
    "OBS_BASE_URL",
    "https://pipeline-release-bucket.obs.example.com",
)
CACHE_DIR = Path(os.environ.get("CODE_CACHE_DIR", "/tmp/pipeline-cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


class CacheStatus(BaseModel):
    pipeline: str
    version: str
    cached: bool
    cache_path: str | None = None


# ---------------------------------------------------------------------------
# 路由
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
    pipeline: str = Query(..., description="流水线名称"),
    version: str = Query(..., description="流水线版本号"),
) -> FileResponse:
    """下载指定版本的 code.tar.gz（命中缓存则直接返回本地文件）。"""
    cache_path = _cache_path(pipeline, version)

    if not cache_path.exists():
        logger.info("缓存未命中：从 OBS 拉取 %s %s", pipeline, version)
        _fetch_from_obs(pipeline, version, cache_path)
    else:
        logger.info("缓存命中：从 %s 直接返回 %s %s", cache_path, pipeline, version)

    return FileResponse(
        path=str(cache_path),
        media_type="application/gzip",
        filename=f"{pipeline}-{version}.tar.gz",
    )


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------


def _cache_path(pipeline: str, version: str) -> Path:
    return CACHE_DIR / pipeline / version / "code.tar.gz"


def _fetch_from_obs(pipeline: str, version: str, dest: Path) -> None:
    """从 OBS 下载 code.tar.gz，并通过 manifest.json 校验 SHA256。"""
    manifest_url = f"{OBS_BASE_URL}/{pipeline}/{version}/manifest.json"
    archive_url = f"{OBS_BASE_URL}/{pipeline}/{version}/code.tar.gz"

    dest.parent.mkdir(parents=True, exist_ok=True)

    # 先拉取 manifest 获取期望的校验值
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

    # 流式下载压缩包至临时文件（原子写，防止读到不完整文件）
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

    # 校验 SHA256
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

    # 原子重命名，写入缓存
    shutil.move(str(tmp), str(dest))
    logger.info(
        "已缓存 %s@%s 至 %s（sha256=%s）", pipeline, version, dest, actual_sha256
    )
