# aus_yolo_mlops

MLOps restructure of the Australia building-footprints **YOLOv8-seg** Colab
pipeline (`australia_yolo_seg_colab.py` / `.ipynb`, 25 cells) into a staged,
CLI-driven, containerised project in the style of the skyscrapper MLOps
template (config / stage modules / tracker / object storage / Airflow DAG).

**What is preserved:** the dataset, splits, leakage guard, labels, model
(`yolov8s-seg.pt`), `imgsz=640`, training/eval/quantisation logic and every
hyper-parameter are byte-identical to the notebook - the stage bodies are
*generated* from exact line slices of the cell files, not retyped. The Colab
files themselves are untouched; only the *structure* around them changed.

**What is new:** run-scoped artifact folders, MinIO upload of run outputs,
MLflow tracking (with local fallback), an Airflow DAG, Docker Compose
infrastructure, and a CLI dispatcher.

---

## 1. Stages

13 stages; the first 11 run in order for `run_cli.py all`, the last two are
appended by `all` (skippable with `--no-mlflow` / `--no-upload`).

| # | Stage | Source cells | What it does | Key outputs (per run) |
|---|-------|--------------|--------------|------------------------|
| 1 | `cleanup_previous_run` | narrowed 2c | Legacy `sydney`-scope outputs, crash `*.tmp` leftovers | `--` |
| 2 | `fetch_dataset` | 4, 5 | Environment check; Kaggle download (skipped when a city/national index exists) | `inputs/dataset/raw_dataset.json` |
| 3 | `fetch_region` | 6 | Build the FlatGeobuf index + per-patch histogram (fast extract from the national index) | `data/sydney_dense_*.{fgb,npz}` |
| 4 | `prepare_data` | 7, 8 | Patch sampling + spatially buffered, leakage-safe split; verifies zero block overlap | `splits/sydney_dense/*.txt`, `split_meta.json` |
| 5 | `build_dataset` | 4b, 9, 10 | Imagery preflight (geo-blocking), tile download, YOLO labels, `dataset.yaml` | `yolo/sydney_dense/`, `download_manifest_*.jsonl` |
| 6 | `train_segmentation` | 11 | YOLOv8-seg training, resumable from `last.pt` | `outputs/models/{best,last}.pt`, `training_metrics.json` |
| 7 | `export_onnx` | 14 | FP32 ONNX export | `outputs/models/best_fp32.onnx` |
| 8 | `quantize_model` | 15-18 | INT8 calibration + quantisation, CPU benchmark, FP32-vs-INT8 metrics | `outputs/models/best_int8.onnx`, `quantization_metrics.json` |
| 9 | `finalize_results` | 12, 13, 19 | val/test evaluation, test figure, run summary | `outputs/{metrics,visualizations,summary}/…` |
| 10 | `compare_groundtruth` | 21 | Green GT polygons vs red predictions, per-image IoU + area (centre crop 384) | `outputs/groundtruth/*` |
| 11 | `predict_area` | 20 | Inference on an arbitrary lon/lat bbox (`MLOPS_PREDICT_BBOX`) | `outputs/predictions/…` |
| 12 | `log_mlflow` | new | Params + metrics + `.json`/`.csv` artifacts to MLflow (falls back to a local summary if the server is down) | `outputs/mlruns/run_metrics.json` |
| 13 | `upload_outputs` | new | Upload `outputs/` to MinIO at `runs/<run_id>/outputs/…` | remote |

> Cell 2c runs **before** cell 5 on purpose: INDEX_READY must be recomputed
> after any cleanup - exactly the notebook's `2c -> 4 -> 5 -> 6` order.
>
> Deliberate narrowings of CELL 2c (the notebook wiped more):
> `checkpoints/`, `onnx/`, `predictions/`, `tile_cache/` are **kept** so a
> later run can resume from `last.pt` (every run's weights are additionally
> preserved in `mlops_runs/<run_id>/outputs/` + MinIO); the city index is
> **not** blindly wiped - staleness is handled by index metadata (below);
> `logs/pipeline.log` is **kept** (each run mirrors its own lines into
> `mlops_runs/<run_id>/outputs/logs/pipeline.log`).

## 2. Layout

```
aus_yolo_mlops/
├── run_cli.py                  # stage dispatcher (list / <stage> / all)
├── seed_storage.py             # create the MinIO bucket + seed config files
├── requirements.txt            # CELL 1 deps + mlflow/boto3
├── configs/
│   ├── pipeline.env            # local defaults (loaded by src/config.py)
│   └── pipeline.env.example
├── src/
│   ├── config.py               # ENV loader -> CELL 2 -> CELL 2b -> MLOPS/RunPaths
│   ├── helpers.py              # CELL 3 (imports, tile math, helpers)
│   ├── data.py                 # stages 1-4  (cells 2c,4,5,6,7,8)
│   ├── dataset.py              # stage 5     (cells 4b,9,10)
│   ├── train.py                # stage 6     (cell 11)
│   ├── export.py               # stage 7     (cell 14)
│   ├── quantize.py             # stage 8     (cells 15-18)
│   ├── evaluate.py             # stages 9-10 (cells 12,13,19,21)
│   ├── inference.py            # stage 11    (cell 20)
│   ├── track.py                # stage 12    (MLflow, local fallback)
│   ├── storage.py              # stage 13    (MinIO, outputs only)
│   └── pipeline.py             # stage registry + run_pipeline()
├── airflow/aus_yolo_mlops_pipeline.py
├── tests/test_smoke.py         # imports, registry, CLI, fallbacks
├── tests/test_stages.py        # stages 1-4 against a local fixture (offline)
├── Dockerfile.airflow          # Airflow 2.10.3 + isolated /opt/pipeline-venv
├── Dockerfile.mlflow           # mlflow server (python:3.11-slim)
└── docker-compose.yml          # postgres, minio, mlflow, airflow
```

## 3. Quickstart (local)

```bash
pip install -r requirements.txt          # Python 3.11
python run_cli.py list                   # show the 13 stages
python run_cli.py all run_001            # full pipeline + MLflow + MinIO
python run_cli.py train_segmentation run_001   # one stage
python run_cli.py all run_002 --no-upload --no-mlflow
```

- `run_id` becomes the folder `work/mlops_runs/<run_id>/`.
- The first run downloads the Kaggle dataset (~6.4 GB) unless a national index
  (`australia_buildings.fgb`, ~2 GB) already sits in `work/data/`.
- Training needs a CUDA GPU; everything before/after runs on CPU
  (`AUS_PROFILE=quick` shrinks the run for a smoke test).

## 4. Docker / Airflow

```bash
docker compose up -d minio mlflow        # storage + tracking only
docker compose up -d                     # + Airflow UI on http://localhost:8080
docker compose run --rm airflow-scheduler airflow dags list
```

| Service | URL | Credentials |
|---------|-----|-------------|
| Airflow | http://localhost:8080 | `airflow` / `airflow` |
| MLflow  | http://localhost:5000 | - |
| MinIO console | http://localhost:9001 | `minioadmin` / `minioadmin` |

`minio-init` creates the buckets (`aus-yolo-mlops`, `mlruns`). Trigger the DAG
`aus_yolo_mlops_pipeline` with param `{"run_id": "run_001"}`: every task is the
same `run_cli.py <stage> "$RUN_ID"` command you would run by hand (the DAG runs
inside the container, `/project`, with the isolated `/opt/pipeline-venv`).

Storage/tracking only, pipeline on the host:

```bash
export MLOPS_MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLOPS_MINIO_ENDPOINT=http://127.0.0.1:9000
python seed_storage.py                   # create bucket + upload config
python run_cli.py all run_001
```

## 5. Configuration

`configs/pipeline.env` is loaded before CELL 2 (shell/docker env wins; override
the file location with `PIPELINE_ENV_FILE`). Any CELL 2 key `K` can be
overridden with `AUS_K`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `AUS_WORK_DIR` | `work` | Shared notebook inputs/outputs (`data/`, `yolo/`, `runs/`, `onnx/`, `splits/`, `checkpoints/`, `logs/`) - same layout as Colab |
| `AUS_MOUNT_DRIVE` | `0` | Keep `0` outside Colab (no Google Drive) |
| `AUS_PROFILE` | `custom` | `custom` (as-is), `quick` (smoke), `full` |
| `AUS_DATASET_PATH` | - | Use a local copy instead of the Kaggle download |
| `AUS_MAX_PATCHES`, `AUS_EPOCHS`, … | CELL 2 values | Generic `AUS_<KEY>` overrides |
| `MLOPS_RUNS_DIR` | `<work>/mlops_runs` | Per-run artifact root |
| `MLOPS_MLFLOW_ENABLED` / `_TRACKING_URI` / `_EXPERIMENT` | `1` / `http://127.0.0.1:5000` / `australia_building_footprints` | Tracking (failures never fail the pipeline) |
| `MLOPS_MINIO_ENABLED` / `_ENDPOINT` / `_BUCKET` | `1` / `http://127.0.0.1:9000` / `aus-yolo-mlops` | Output upload (`0` = skip) |
| `MLOPS_PREDICT_BBOX` | - | `xmin,ymin,xmax,ymax` for `predict_area` |

## 6. Artifacts

```
<work_dir>/                                  # shared, run-agnostic (as in Colab)
├── data/sydney_dense_{buildings.fgb,hist_z16_p3.npz}, australia_buildings.fgb
├── yolo/sydney_dense/{images,labels,dataset.yaml}
├── splits/sydney_dense/{train,val,test}.txt + split_meta.json
├── runs/sydney_dense/{aus_buildings_seg,eval_val,eval_test}/results.csv
├── checkpoints/{best,last}.pt   onnx/*.onnx   logs/pipeline.log
└── mlops_runs/<run_id>/                    # per-run artifact root
    ├── inputs/{dataset,splits,imagery}/    # copies/pointers of stage inputs
    └── outputs/{models,metrics,predictions,visualizations,groundtruth,summary,mlruns,logs}/
        └── summary/pipeline_result.json    # written even when a stage fails
```

Remote: MinIO `runs/<run_id>/outputs/…` (`.json`/`.csv`/figures/weights of the
run), MLflow experiment `australia_building_footprints` (params from CONFIG +
`run_id`, metrics from `training/eval/quantization/compare` JSON, artifacts
`.json`/`.csv` only).

## 7. Tests

```bash
python tests/test_smoke.py       # imports, RunPaths, 13-stage registry, CLI, MLflow fallback, storage gate
python tests/test_stages.py      # offline end-to-end: stages 1-4 against a national-index fixture
python tests/test_validation.py  # index metadata, splits guard, checkpoint compatibility (archive, never delete)
```

Both are network-free and GPU-free; `test_stages.py` asserts the leakage guard
(zero block overlap) and the fast extract path.

## 8. Design notes

- **Verbatim generation.** `src/*` stage bodies are emitted by a generator from
  exact line ranges of the split cell files (every emitted file is
  `compile()`-checked); only `# MLOps:` glue (stage prologues, JSON bridges,
  persistence) is hand-written. Module-level `from src.config import *` +
  `from src.helpers import *` recreate CELL 3's globals inside the stages.
- **Run-agnostic CELL 2.** `work_dir` and every `AUS_*` key keep their
  original Colab meaning; per-run outputs live under `mlops_runs/<run_id>`
  (`RunPaths`, validated `run_id`), so concurrent runs never collide.
- **Storage is outputs-only.** Inputs (Kaggle/ESRI tiles) are downloaded inside
  the stages; MinIO receives only `outputs/`. `upload_outputs` degrades to
  `{"skipped": true}` when disabled and never fails the pipeline.
- **MLflow with a local fallback.** If the tracking server is unreachable the
  run metrics are still written to `outputs/mlruns/run_metrics.json`.
- **Staleness validation, not blind wipes.** CELL 5/6/11 only test file
  existence, so the MLOps layer validates provenance instead of deleting:
  - city index: `data/<scope>_index_meta.json` records scope/bbox/zoom/
    patch_tiles; a mismatch (or a missing file) archives the index to
    `data/stale_index_<timestamp>/` and rebuilds it. A healthy index is
    reused across runs.
  - splits: `train_segmentation` fails fast when `split_meta.json` on disk
    was built for different config knobs (re-run `prepare_data`).
  - checkpoints: a bare `last.pt` is never enough to resume. Every train
    stage writes `checkpoints/checkpoint_meta.json` (model, imgsz, dataset
    fingerprint); on mismatch - or on legacy/foreign weights with unknown
    provenance - the files are archived to `checkpoints/stale_<timestamp>/`
    (never deleted) and training starts from scratch. Optimizer knobs
    (batch, patience, lr) may change on resume; data/architecture may not.
- **Per-run logs.** Every `log()` line is mirrored into
  `mlops_runs/<run_id>/outputs/logs/pipeline.log` (in addition to the shared
  `logs/pipeline.log`), so each run's history survives and is uploaded.
- **Fail-visible CLI/DAG.** `run_pipeline` writes
  `outputs/summary/pipeline_result.json` even on failure and re-raises, so both
  the CLI and the Airflow task exit non-zero with a usable artefact.
