#!/usr/bin/env python3
"""Seed the MinIO bucket with the pipeline configuration (template pattern).

  python seed_storage.py            # create the bucket + upload the config files
  python seed_storage.py --check    # only verify the bucket is reachable

The pipeline itself never READS from storage (inputs are downloaded inside the
stages); this script exists so a fresh environment has the bucket and a copy of
the run configuration next to the run outputs.
"""
from __future__ import annotations

import argparse
import os
import sys

from src.config import *
from src.storage import MinIOClient

SEEDED_FILES = ("configs/pipeline.env", "requirements.txt", "configs/pipeline.env.example")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="seed_storage.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="only create/verify the bucket, upload nothing")
    args = parser.parse_args(argv)

    client = MinIOClient()
    bucket = client.ensure_bucket()
    print(f"bucket: {bucket} @ {client.endpoint_url}")
    if args.check:
        return 0

    root = os.path.dirname(os.path.abspath(__file__))
    uploaded = []
    for rel in SEEDED_FILES:
        path = os.path.join(root, *rel.split("/"))
        if not os.path.exists(path):
            continue
        key = "configs/" + os.path.basename(path)
        uploaded.append(client.upload_file(path, key))
        print(f"uploaded {rel} -> {key}")
    if not uploaded:
        print("nothing to upload (no config files found)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:                       # noqa: BLE001 - CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
