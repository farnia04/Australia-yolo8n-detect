# ========================================================================

# src/config.py - configuration + logging + pipeline state + RunPaths

# ------------------------------------------------------------------------

# Generated from the Colab cells: the env loader, CELL 2 and CELL 2b are

# included verbatim (only their order is preserved), then the MLOps run

# management block that the notebook does not need.

# ========================================================================


# --- MLOps: optional configs/pipeline.env loader -------------------------------
# KEY=VALUE lines, "#" comments. Loaded BEFORE CELL 2 applies its AUS_*
# environment overrides, so a fresh checkout runs with sane local defaults
# while the Colab behaviour (no file -> shell environment only) is preserved.
# Set PIPELINE_ENV_FILE to point at another file.
import os as _os


def _load_pipeline_env():
    path = _os.environ.get("PIPELINE_ENV_FILE")
    if not path:
        path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                             "configs", "pipeline.env")
    if not _os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _key, _val = _line.split("=", 1)
                _key, _val = _key.strip(), _val.strip().strip('"').strip("'")
                if _key and _val:
                    _os.environ.setdefault(_key, _val)
    except Exception:
        pass


_load_pipeline_env()
del _load_pipeline_env

# ============================================================================
# CELL 2 - configuration
# ============================================================================
import os
# MEMORY FIX: reduce CUDA allocator fragmentation. Must be set before the CUDA
# caching allocator is created (i.e. before the first GPU allocation), so it
# goes here, before torch is used. On dense batches the failure mode is usually
# "cannot find a big enough free block" rather than "truly out of capacity".
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

CONFIG = dict(
    # --- data source / persistence ---
    dataset_handle="joebeachcapital/australia-building-footprints",
    work_dir="/content/drive/MyDrive/aus_seg",  # persistent Google Drive
    mount_drive=True,          # mount Drive at /content/drive (Colab only)
    resume=True,               # skip any stage whose outputs already exist/valid
    keep_tile_cache=False,     # delete per-patch tiles after stitching (Drive space)
    manifest_flush_every=20,   # write download_manifest every N processed patches
    patch_batch=64,            # patches per batch (parallel tile fetch, then label)
    build_limit=None,          # cap patches to BUILD (None = all). Keeps the full
                               # split, processes train/val/test interleaved, so you
                               # can stop anytime with a usable, balanced dataset.

    # --- city scope: work on ONE Australian city, not the whole country ------
    restrict_to_city=True,
    city_name="sydney",
    sub_area="dense",   # isolate this run: scope becomes "sydney_dense"
    # (xmin, ymin, xmax, ymax) in lon/lat -> dense inner Sydney (busy, building-packed)
    city_bbox=(151.07, -34.00, 151.30, -33.76),
    keep_national_index=True,  # keep the national FGB (~2 GB) to reuse for other cities

    # --- tiling ---
    imagery_provider="esri", # "esri" (best) or "eox_s2cloudless" (free fallback)
    zoom=16,                 # ESRI World Imagery zoom (~2.0 m/px at -33 lat)
    patch_tiles=3,           # 3x3 tiles -> 768x768 px patch (~1.5 km at z16)
    yolo_imgsz=640,          # YOLO trains on 640 (patch is resized)

    # --- sampling (city scope: None = keep every valid patch) ---
    max_patches=350,         # cap total patches (target ~200-500, dense area only)
    min_buildings=5,         # keep only DENSE patches (busy, building-packed areas)
    region_patches=128,      # stratification cell size in patches (~50 km)
    seed=42,

    # --- leakage-safe split ---
    block_patches=8,         # spatial block size in patches (auto-reduced for
                             # small/coarse scopes so val/test always exist)
    split_chunk_blocks=40,   # contiguous Morton-ordered blocks per chunk
    buffer_blocks=1,         # drop patches within N blocks of another split
    train_frac=0.80,
    val_frac=0.10,
    test_frac=0.10,

    # --- imagery / io ---
    download_workers=16,
    jpeg_quality=92,
    blank_std_threshold=8.0, # skip near-uniform tiles (no coverage / clouds)
    min_poly_area_px=25.0,   # drop tiny polygons after rasterization

    # --- training ---
    model="yolov8s-seg.pt",
    epochs=100,
    batch=4,                # MEMORY FIX: peak VRAM scales with objects x batch.
                            # Dense patches (hundreds/thousands of buildings)
                            # overflowed a 15 GB T4 at batch=16. Ultralytics
                            # gradient accumulation (nbs) keeps the EFFECTIVE
                            # batch constant, so metrics stay the same. CELL 11
                            # auto-backs off further (4 -> 2 -> 1) on CUDA OOM.
    patience=20,
    device=0,

    # --- ONNX export / INT8 quantization / CPU deployment ---
    quantization_enabled=True,   # master switch for the ONNX+INT8 stage
    quant_bits=8,                # 8-bit integer quantization
    quant_calib_samples=24,      # number of TRAIN images used for calibration
    cpu_inference_enabled=True,  # run the INT8 model with onnxruntime CPU
    onnx_opset=13,               # opset >= 13: QDQ per-channel bias needs the
                                 # DequantizeLinear 'axis' attribute (added in 13)
    quant_per_channel=True,      # per-channel weight scales (better accuracy)
    quant_activation_type="uint8",  # unsigned 8-bit activations (QUInt8)
    quant_weight_type="int8",       # signed 8-bit weights (QInt8)
    onnx_imgsz=None,             # None -> reuse yolo_imgsz
    onnx_compare_samples=16,     # images used for FP32-vs-INT8 comparison
    onnx_bench_runs=10,          # timed runs per model for the CPU benchmark
    onnx_bench_warmup=2,         # warmup runs (not timed)
    # YOLOv8 concatens box coords (up to ~640) with class scores (0..1) into the
    # same output tensor. Quantizing that final Concat with one shared scale
    # crushes the class scores to 0. Excluding the output-producing nodes keeps
    # the head outputs in FP32 while the backbone/neck stay fully INT8.
    quant_exclude_output_nodes=True,
)

# Run profile: "full" = ALL valid patches + long training (the real run).
# "quick" is only for a fast smoke test. Set AUS_PROFILE=quick to try it out.
PROFILE = os.environ.get("AUS_PROFILE", "custom")
_PRESETS = {
    "quick": dict(max_patches=200, epochs=10, model="yolov8s-seg.pt", batch=16),
    "full":  dict(max_patches=None, epochs=500, model="yolov8s-seg.pt", batch=16),
}
CONFIG.update(_PRESETS.get(PROFILE, {}))

# environment overrides (handy for quick local/CPU smoke tests), e.g. AUS_MAX_PATCHES=50
for _k in list(CONFIG):
    _env = os.environ.get("AUS_" + _k.upper())
    if _env is not None:
        _cur = CONFIG[_k]
        if _k in ("max_patches", "build_limit"):
            CONFIG[_k] = None if _env.strip().lower() in ("", "none", "all", "-1") else int(_env)
        elif _k == "device":
            CONFIG[_k] = int(_env) if _env.isdigit() else _env
        elif isinstance(_cur, bool):
            CONFIG[_k] = _env.lower() in ("1", "true", "yes", "on")
        elif isinstance(_cur, int):
            CONFIG[_k] = int(_env)
        elif isinstance(_cur, float):
            CONFIG[_k] = float(_env)
        else:
            CONFIG[_k] = _env

for k, v in CONFIG.items():
    globals()[k] = v

if not torch.cuda.is_available():
    device = "cpu"

assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9, "split fractions must sum to 1"

DATA_DIR = os.path.join(work_dir, "data")
CACHE_DIR = os.path.join(work_dir, "tile_cache")
ONNX_DIR = os.path.join(work_dir, "onnx")          # FP32 / INT8 exports + metrics
LOGS_DIR = os.path.join(work_dir, "logs")
PRED_DIR = os.path.join(work_dir, "predictions")
CHECKPOINT_DIR = os.path.join(work_dir, "checkpoints")
# per-city scope so different cities never mix images/splits/runs
SCOPE = (city_name + ("_" + sub_area if sub_area else "")) if restrict_to_city else "australia"
YOLO_DIR = os.path.join(work_dir, "yolo", SCOPE)
RUNS_DIR = os.path.join(work_dir, "runs", SCOPE)
SPLITS_DIR = os.path.join(work_dir, "splits", SCOPE)
for d in (work_dir, DATA_DIR, YOLO_DIR, CACHE_DIR, RUNS_DIR, ONNX_DIR,
          SPLITS_DIR, LOGS_DIR, PRED_DIR, CHECKPOINT_DIR):
    os.makedirs(d, exist_ok=True)

# national index (all of Australia) is kept so we can extract any city from it
NATIONAL_FGB = os.path.join(DATA_DIR, "australia_buildings.fgb")
NATIONAL_HIST = os.path.join(DATA_DIR, "patch_histogram.npz")
# the index actually used for this run: city-scoped when restrict_to_city=True.
# The histogram depends on zoom + patch_tiles, so its name includes them (the
# FlatGeobuf itself is zoom-independent).
HIST_TAG = f"z{zoom}_p{patch_tiles}"
if restrict_to_city:
    FGB_PATH = os.path.join(DATA_DIR, f"{SCOPE}_buildings.fgb")
    HIST_PATH = os.path.join(DATA_DIR, f"{SCOPE}_hist_{HIST_TAG}.npz")
else:
    FGB_PATH = NATIONAL_FGB
    HIST_PATH = os.path.join(DATA_DIR, f"australia_hist_{HIST_TAG}.npz")
YAML_PATH = os.path.join(YOLO_DIR, "dataset.yaml")
WEIGHTS_DIR = os.path.join(RUNS_DIR, "aus_buildings_seg", "weights")
LAST_PT = os.path.join(WEIGHTS_DIR, "last.pt")
BEST_PT = os.path.join(WEIGHTS_DIR, "best.pt")
ONNX_FP32 = os.path.join(ONNX_DIR, "best_fp32.onnx")
ONNX_INT8 = os.path.join(ONNX_DIR, "best_int8.onnx")
QUANT_METRICS_PATH = os.path.join(ONNX_DIR, "quantization_metrics.json")
STATE_PATH = os.path.join(work_dir, f"pipeline_state_{SCOPE}.json")
MANIFEST_PATH = os.path.join(work_dir, f"download_manifest_{SCOPE}.jsonl")  # append-only
CALIB_JSON = os.path.join(ONNX_DIR, "calibration_samples.json")
QUANT_META = os.path.join(ONNX_DIR, "quant_meta.json")
SPLIT_META = os.path.join(SPLITS_DIR, "split_meta.json")
SPLIT_FILES = {n: os.path.join(SPLITS_DIR, f"{n}.txt") for n in ("train", "val", "test")}

PATCH_PX = patch_tiles * 256
print("work_dir :", work_dir)
print("profile  :", PROFILE, "| max_patches:", max_patches, "| epochs:", epochs, "| model:", model)
print("patch    :", patch_tiles, "tiles =", PATCH_PX, "px")
print("city     :", (f"{city_name} {city_bbox}" if restrict_to_city else "ALL Australia"))


# ============================================================================
# CELL 2b - Google Drive + logging + pipeline state + download manifest
# ----------------------------------------------------------------------------
# Everything that matters lives on Drive so a Colab disconnect never loses
# progress. This cell is safe to run repeatedly (idempotent).
# ============================================================================
import csv
import json
import shutil
import time as _time

# --- mount Google Drive (Colab only; ignored when running locally) ----------
if mount_drive:
    try:
        from google.colab import drive
        if not os.path.ismount("/content/drive"):
            drive.mount("/content/drive")
        print("[DRIVE] mounted /content/drive")
    except Exception as e:
        print(f"[DRIVE] not mounted ({e}); continuing with local work_dir")

# --- logging ----------------------------------------------------------------
LOG_PATH = os.path.join(LOGS_DIR, "pipeline.log")


def log(msg):
    """Print with a timestamp and append to logs/pipeline.log (survives restarts)."""
    line = f"{_time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(msg)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# --- pipeline state (a hint, never the source of truth) ---------------------
STAGES = ["dataset_download", "index_build", "split_generation",
          "patch_generation", "label_generation", "training",
          "onnx_export", "quantization", "cpu_inference"]


def load_state():
    if os.path.exists(STATE_PATH):
        try:
            return json.load(open(STATE_PATH))
        except Exception:
            pass
    return {s: "pending" for s in STAGES}


def save_state(state):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_PATH)          # atomic: never a half-written state file


STATE = load_state()


def set_stage(name, status, **extra):
    STATE[name] = status
    STATE.setdefault("meta", {}).update(extra)
    STATE["updated"] = _time.strftime("%Y-%m-%d %H:%M:%S")
    save_state(STATE)
    log(f"[STATE] {name} = {status}")


def record_batch_state(failed=None, good=None):
    """MEMORY FIX: persist which micro-batch OOM'd / which one worked. Without
    this, a re-run after a hard crash would start the ladder at the configured
    batch again and walk into the SAME OOM. Only metadata; nothing else in the
    pipeline reads or writes this key."""
    mem = STATE.setdefault("mem", {})
    changed = False
    if failed is not None and mem.get("failed_batch") != int(failed):
        mem["failed_batch"] = int(failed)
        changed = True
    if good is not None and mem.get("last_good_batch") != int(good):
        mem["last_good_batch"] = int(good)
        changed = True
    if changed:
        mem["updated"] = _time.strftime("%Y-%m-%d %H:%M:%S")
        save_state(STATE)
        log(f"[MEM] recorded batch history: {json.dumps(mem)}")


# --- download manifest (append-only JSONL: cheap at 100k+ patches) ----------
MANIFEST_FIELDS = ["patch_id", "split", "source", "status", "image_path",
                   "label_path", "file_size", "n_polygons", "validated", "error"]


def load_manifest():
    """Replay the append-only manifest; the last record for a patch wins."""
    rows = {}
    if not os.path.exists(MANIFEST_PATH):
        return rows
    try:
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("patch_id"):
                    rows[r["patch_id"]] = r
    except Exception as e:
        log(f"[WARN] manifest unreadable ({e}); starting fresh")
    return rows


def append_manifest(records):
    """Append new/updated rows incrementally (called every batch)."""
    if not records:
        return
    with open(MANIFEST_PATH, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({k: r.get(k, "") for k in MANIFEST_FIELDS}) + "\n")


log(f"[CHECK] work_dir = {work_dir}")
log(f"[CHECK] state = {json.dumps({k: STATE.get(k) for k in STAGES})}")


# ============================================================================
# MLOps run management (added for the aus_yolo_mlops conversion)
# ----------------------------------------------------------------------------
# CELL 2 stays RUN-AGNOSTIC: work_dir and every AUS_* key keep their original
# meaning, so the Colab notebook and this project share the same inputs
# (data/, yolo/, runs/, onnx/, splits/, checkpoints/). Only the PER-RUN
# outputs live under MLOPS_RUNS_DIR/<run_id>/, so two runs of the same scope
# never overwrite each other.
# ============================================================================
import dataclasses


def _env_flag(name, default=True):
    _v = os.environ.get(name)
    return default if _v is None else _v.strip().lower() in ("1", "true", "yes", "on")


MLOPS_RUNS_DIR = os.environ.get("MLOPS_RUNS_DIR") or os.path.join(work_dir, "mlops_runs")
MLOPS_DEFAULT_BASE_DIR = os.environ.get("MLOPS_BASE_DIR") or None

# MinIO / S3 - RUN OUTPUTS ONLY. Inputs (Kaggle dataset, ESRI tiles) are
# downloaded inside the stages themselves, never read back from storage.
MLOPS_MINIO_ENABLED = _env_flag("MLOPS_MINIO_ENABLED", True)
MLOPS_MINIO_ENDPOINT = os.environ.get("MLOPS_MINIO_ENDPOINT", "http://127.0.0.1:9000")
MLOPS_MINIO_ACCESS_KEY = os.environ.get("MLOPS_MINIO_ACCESS_KEY", "minioadmin")
MLOPS_MINIO_SECRET_KEY = os.environ.get("MLOPS_MINIO_SECRET_KEY", "minioadmin")
MLOPS_MINIO_BUCKET = os.environ.get("MLOPS_MINIO_BUCKET", "aus-yolo-mlops")
MLOPS_MINIO_SECURE = _env_flag("MLOPS_MINIO_SECURE", False)

# MLflow tracking
MLOPS_MLFLOW_ENABLED = _env_flag("MLOPS_MLFLOW_ENABLED", True)
MLOPS_MLFLOW_TRACKING_URI = os.environ.get("MLOPS_MLFLOW_TRACKING_URI",
                                           "http://127.0.0.1:5000")
MLOPS_MLFLOW_EXPERIMENT = os.environ.get("MLOPS_MLFLOW_EXPERIMENT",
                                         "australia_building_footprints")

# optional lon/lat bbox for the predict_area stage: xmin,ymin,xmax,ymax
MLOPS_PREDICT_BBOX = os.environ.get("MLOPS_PREDICT_BBOX", "")


@dataclasses.dataclass(frozen=True)
class RunPaths:
    """Per-run directory layout.

    inputs/  - what a stage resolves/downloads for THIS run
    outputs/ - everything the run produces: tracked by MLflow, uploaded to MinIO
    """

    run_id: str
    base_dir: str
    run_dir: str
    inputs_dataset: str
    inputs_splits: str
    inputs_imagery: str
    outputs_dir: str
    out_metrics: str
    out_predictions: str
    out_visualizations: str
    out_groundtruth: str
    out_models: str
    out_summary: str
    out_mlruns: str
    out_logs: str

    @classmethod
    def build(cls, run_id, base_dir=None):
        run_id = str(run_id).strip()
        if not run_id or any(c in run_id for c in ("/", "\\", "..")):
            raise ValueError("invalid run_id: %r" % (run_id,))
        base = os.path.abspath(base_dir or MLOPS_DEFAULT_BASE_DIR or MLOPS_RUNS_DIR)
        run_dir = os.path.join(base, run_id)
        ins = os.path.join(run_dir, "inputs")
        outs = os.path.join(run_dir, "outputs")
        paths = cls(
            run_id=run_id, base_dir=base, run_dir=run_dir,
            inputs_dataset=os.path.join(ins, "dataset"),
            inputs_splits=os.path.join(ins, "splits"),
            inputs_imagery=os.path.join(ins, "imagery"),
            outputs_dir=outs,
            out_metrics=os.path.join(outs, "metrics"),
            out_predictions=os.path.join(outs, "predictions"),
            out_visualizations=os.path.join(outs, "visualizations"),
            out_groundtruth=os.path.join(outs, "groundtruth"),
            out_models=os.path.join(outs, "models"),
            out_summary=os.path.join(outs, "summary"),
            out_mlruns=os.path.join(outs, "mlruns"),
            out_logs=os.path.join(outs, "logs"),
        )
        paths.ensure()
        return paths

    def ensure(self):
        for d in (self.run_dir, self.inputs_dataset, self.inputs_splits,
                  self.inputs_imagery, self.outputs_dir, self.out_metrics,
                  self.out_predictions, self.out_visualizations,
                  self.out_groundtruth, self.out_models, self.out_summary,
                  self.out_mlruns, self.out_logs):
            os.makedirs(d, exist_ok=True)
        return self

    def to_dict(self):
        return dataclasses.asdict(self)


def run_paths(run_id, base_dir=None):
    """Convenience wrapper: run_paths("run_001")."""
    return RunPaths.build(run_id, base_dir)


# --- per-run log ------------------------------------------------------------
# CELL 2b's log() writes to the shared logs/pipeline.log. Every MLOps
# invocation additionally mirrors log() lines into the active run's
# outputs/logs/pipeline.log, so each run keeps its own log history
# (upload_outputs ships it to MinIO). pipeline.run_stage() sets the active
# run log; without it, behaviour is exactly the notebook's.
_ACTIVE_RUN_LOG = None


def set_active_run_log(path):
    """Mirror subsequent log() lines into this run's outputs/logs/pipeline.log
    (pass None to stop mirroring)."""
    global _ACTIVE_RUN_LOG
    _ACTIVE_RUN_LOG = path


_base_log = log  # CELL 2b's original (kept verbatim, only routing is added)


def log(msg):
    _base_log(msg)
    if _ACTIVE_RUN_LOG is not None:
        try:
            with open(_ACTIVE_RUN_LOG, "a", encoding="utf-8") as _f:
                _f.write(f"{_time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\\n")
        except Exception:
            pass
