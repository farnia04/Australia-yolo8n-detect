"""MinIO/S3 upload of run outputs (template pattern).

RUN OUTPUTS ONLY: the Kaggle dataset and the imagery tiles are downloaded
inside the stages themselves (network), never read back from object storage.
A storage outage must never fail the pipeline - the local outputs stay
authoritative and the upload stage reports the error instead of raising.
"""
from __future__ import annotations

import os

from src.config import *


class MinIOClient:
    """Thin boto3 wrapper with the endpoint/bucket settings from src.config."""

    def __init__(self, endpoint_url=None, access_key=None, secret_key=None,
                 bucket=None, secure=None):
        self.endpoint_url = endpoint_url or MLOPS_MINIO_ENDPOINT
        self.access_key = access_key or MLOPS_MINIO_ACCESS_KEY
        self.secret_key = secret_key or MLOPS_MINIO_SECRET_KEY
        self.bucket = bucket or MLOPS_MINIO_BUCKET
        self.secure = MLOPS_MINIO_SECURE if secure is None else bool(secure)
        self._s3 = None

    @property
    def s3(self):
        if self._s3 is None:
            import boto3
            from botocore.config import Config as BotoConfig
            self._s3 = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name="us-east-1",
                config=BotoConfig(signature_version="s3v4"),
            )
        return self._s3

    def ensure_bucket(self):
        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except Exception:
            self.s3.create_bucket(Bucket=self.bucket)
            log(f"[MINIO] created bucket {self.bucket}")
        return self.bucket

    def upload_file(self, local_path, key):
        self.s3.upload_file(str(local_path), self.bucket, key)
        return key

    def upload_dir(self, local_dir, prefix=""):
        """Upload every file below local_dir; returns the uploaded keys."""
        keys = []
        for root, _dirs, files in os.walk(local_dir):
            for name in sorted(files):
                full = os.path.join(root, name)
                rel = os.path.relpath(full, local_dir).replace(os.sep, "/")
                key = "/".join(part for part in (str(prefix).strip("/"), rel) if part)
                keys.append(self.upload_file(full, key))
        return keys


def upload_run_outputs(run):
    """Stage 12 - upload outputs/ to s3://<bucket>/runs/<run_id>/..."""
    if not MLOPS_MINIO_ENABLED:
        log("[MINIO] MLOPS_MINIO_ENABLED=0 -> upload skipped")
        return dict(skipped=True, reason="MLOPS_MINIO_ENABLED=0")
    if not os.path.isdir(run.outputs_dir):
        log(f"[MINIO] nothing to upload: {run.outputs_dir} missing")
        return dict(skipped=True, reason="outputs_dir missing")

    try:
        client = MinIOClient()
        bucket = client.ensure_bucket()
        prefix = f"runs/{run.run_id}"
        keys = client.upload_dir(run.outputs_dir, prefix)
    except Exception as e:
        log(f"[MINIO] upload failed ({e}) -> local outputs kept as source of truth")
        return dict(ok=False, error=str(e), bucket=MLOPS_MINIO_BUCKET,
                    endpoint=MLOPS_MINIO_ENDPOINT)

    log(f"[MINIO] uploaded {len(keys)} file(s) -> "
        f"{MLOPS_MINIO_ENDPOINT}/{bucket}/{prefix}/")
    return dict(ok=True, bucket=bucket, endpoint=MLOPS_MINIO_ENDPOINT,
                prefix=prefix, uploaded=len(keys), keys=keys[:50])
