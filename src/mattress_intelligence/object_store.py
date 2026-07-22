"""Content-addressed local and S3-compatible artifact storage."""

from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from .settings import Settings


@dataclass(frozen=True, slots=True)
class StoredObject:
    sha256: str
    local_path: str
    object_uri: str | None
    content_type: str
    size_bytes: int


class ObjectStore(Protocol):
    def put_bytes(
        self,
        body: bytes,
        *,
        content_type: str,
        source_url: str,
        namespace: str,
    ) -> StoredObject: ...

    def get_bytes(self, *, local_path: str | None, object_uri: str | None) -> bytes: ...


def _extension(content_type: str, source_url: str) -> str:
    media_type = content_type.split(";", 1)[0].strip().casefold()
    extension = mimetypes.guess_extension(media_type) or ""
    if not extension:
        suffix = Path(urlsplit(source_url).path).suffix
        extension = suffix if 0 < len(suffix) <= 10 else ""
    if extension == ".jpe":
        extension = ".jpg"
    return extension


class LocalObjectStore:
    """Always-on content-addressed local cache."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(
        self,
        body: bytes,
        *,
        content_type: str,
        source_url: str,
        namespace: str,
    ) -> StoredObject:
        digest = hashlib.sha256(body).hexdigest()
        extension = _extension(content_type, source_url)
        target = self.root / namespace / digest[:2] / f"{digest}{extension}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(body)
        return StoredObject(
            sha256=digest,
            local_path=str(target),
            object_uri=None,
            content_type=content_type,
            size_bytes=len(body),
        )

    def get_bytes(self, *, local_path: str | None, object_uri: str | None) -> bytes:
        if not local_path:
            raise FileNotFoundError("No local artifact path is available.")
        return Path(local_path).read_bytes()


class MinioObjectStore:
    """Mirror content-addressed artifacts to MinIO/S3 while retaining a local cache."""

    def __init__(self, settings: Settings) -> None:
        try:
            from minio import Minio
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "MinIO storage is configured but the minio package is not installed."
            ) from exc

        if not settings.object_storage_enabled:
            raise ValueError("MinIO endpoint and credentials are required.")
        self.local = LocalObjectStore(settings.artifact_dir)
        self.bucket = settings.minio_bucket
        self.client = Minio(
            settings.minio_endpoint or "",
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
            region=settings.minio_region,
        )
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket, location=settings.minio_region)

    def put_bytes(
        self,
        body: bytes,
        *,
        content_type: str,
        source_url: str,
        namespace: str,
    ) -> StoredObject:
        local = self.local.put_bytes(
            body,
            content_type=content_type,
            source_url=source_url,
            namespace=namespace,
        )
        extension = Path(local.local_path).suffix
        object_name = f"{namespace}/{local.sha256[:2]}/{local.sha256}{extension}"
        from io import BytesIO

        self.client.put_object(
            self.bucket,
            object_name,
            BytesIO(body),
            length=len(body),
            content_type=content_type.split(";", 1)[0],
            metadata={
                "source-url-sha256": hashlib.sha256(source_url.encode("utf-8")).hexdigest(),
                "content-sha256": local.sha256,
            },
        )
        return StoredObject(
            sha256=local.sha256,
            local_path=local.local_path,
            object_uri=f"s3://{self.bucket}/{object_name}",
            content_type=content_type,
            size_bytes=len(body),
        )

    def get_bytes(self, *, local_path: str | None, object_uri: str | None) -> bytes:
        if local_path and Path(local_path).exists():
            return Path(local_path).read_bytes()
        if not object_uri or not object_uri.startswith("s3://"):
            raise FileNotFoundError("No readable local or MinIO artifact location is available.")
        remainder = object_uri.removeprefix("s3://")
        bucket, _, object_name = remainder.partition("/")
        response = self.client.get_object(bucket, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()


def build_object_store(settings: Settings) -> ObjectStore:
    if settings.object_storage_enabled:
        return MinioObjectStore(settings)
    return LocalObjectStore(settings.artifact_dir)
