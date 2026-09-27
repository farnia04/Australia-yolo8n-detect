"""Unit tests for the MLOps validation glue (no network, no training).

  python tests/test_validation.py

Covers: city-index metadata (reuse / stale -> archive), the splits-config
guard (missing / matching / mismatched), and checkpoint compatibility
(none / matching / stale / unknown provenance -> archive, never delete).
"""
import json
import os
import sys
import tempfile

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = tempfile.mkdtemp(prefix="mlops_valid_")
os.environ["AUS_WORK_DIR"] = WORK
os.environ["AUS_MOUNT_DRIVE"] = "0"
sys.path.insert(0, PROJECT)

import src.config as cfg       # noqa: E402
import src.data as data        # noqa: E402
import src.train as train      # noqa: E402


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ------------------------------------------------------- index metadata
assert data.invalidate_stale_index() is False          # nothing planted
open(cfg.FGB_PATH, "wb").write(b"fake-fgb")
open(cfg.HIST_PATH, "wb").write(b"fake-hist")
data._write_index_meta()
assert data.invalidate_stale_index() is False          # healthy index reused
assert os.path.exists(cfg.FGB_PATH)

meta = data._read_index_meta()
assert meta["scope"] == "sydney_dense", meta
assert meta["city_bbox"] == [151.07, -34.0, 151.3, -33.76], meta
meta["city_bbox"] = [0.0, 0.0, 1.0, 1.0]               # simulate a bbox change
write_json(data._index_meta_path(), meta)
assert data.invalidate_stale_index() is True           # stale -> archived
assert not os.path.exists(cfg.FGB_PATH)
assert not os.path.exists(cfg.HIST_PATH)
stale = [d for d in os.listdir(cfg.DATA_DIR) if d.startswith("stale_index_")]
assert len(stale) == 1, stale
archived = os.path.join(cfg.DATA_DIR, stale[0])
assert os.path.exists(os.path.join(archived, os.path.basename(cfg.FGB_PATH)))
assert os.path.exists(os.path.join(archived, os.path.basename(cfg.HIST_PATH)))
print("index metadata validation OK")

# ---------------------------------------------------------- splits guard
try:
    train._assert_splits_match_config()
except SystemExit as exc:
    print("splits guard (missing) OK:", exc)
else:
    raise AssertionError("missing splits must fail fast")

disk = dict(train._dataset_fingerprint())              # matching splits
disk.update(block_patches_eff=1, split_chunk_blocks=1, buffer_blocks=1,
            region_patches=8)
write_json(cfg.SPLIT_META, disk)
train._assert_splits_match_config()                    # passes
disk["seed"] = 999                                     # simulate config change
write_json(cfg.SPLIT_META, disk)
try:
    train._assert_splits_match_config()
except SystemExit as exc:
    print("splits guard (mismatch) OK:", exc)
else:
    raise AssertionError("stale splits must fail fast")
print("splits guard OK")

# ------------------------------------------------- checkpoint validation
# (LAST_PT/BEST_PT live in runs/<scope>/aus_buildings_seg/weights/)
os.makedirs(os.path.dirname(cfg.LAST_PT), exist_ok=True)
assert train.validate_checkpoints() is True            # nothing to check

open(cfg.LAST_PT, "wb").write(b"fake-ckpt")            # matching provenance
train._write_checkpoint_meta()
assert train.validate_checkpoints() is True
assert os.path.exists(cfg.LAST_PT)
print("checkpoint validation OK (compatible resume allowed)")

meta = train._read_checkpoint_meta()                   # stale provenance
meta["model"] = "yolov8n-seg.pt"
write_json(train.CHECKPOINT_META, meta)
assert train.validate_checkpoints() is False
assert not os.path.exists(cfg.LAST_PT)
stale = [d for d in os.listdir(cfg.CHECKPOINT_DIR) if d.startswith("stale_")]
assert len(stale) == 1, stale
archived = os.path.join(cfg.CHECKPOINT_DIR, stale[0])
assert os.path.exists(os.path.join(archived, "last.pt"))
assert os.path.exists(os.path.join(archived, "checkpoint_meta.json"))
print("checkpoint validation OK (incompatible archived, never deleted)")

open(cfg.LAST_PT, "wb").write(b"fake-ckpt-2")          # unknown provenance
assert not os.path.exists(train.CHECKPOINT_META)
assert train.validate_checkpoints() is False
assert not os.path.exists(cfg.LAST_PT)
print("checkpoint validation OK (legacy/foreign weights archived)")

print("\nALL VALIDATION TESTS PASSED")
