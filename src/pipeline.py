"""Stage registry + run_pipeline() for the aus_yolo_mlops pipeline.

Stage modules are imported LAZILY (by dotted path), so `run_cli.py --help`,
`run_cli.py list` and the Airflow DAG never pay for the torch/ultralytics
imports until a stage actually runs.

Stage order (mirrors the notebook's cell order):

  1  cleanup_previous_run  narrowed CELL 2c   legacy scope + crash leftovers
  2  fetch_dataset         CELL 4 + CELL 5    Kaggle dataset
  3  fetch_region          CELL 6             FlatGeobuf index + histogram
  4  prepare_data          CELL 7 + CELL 8    sampling + leakage-safe split
  5  build_dataset         CELL 4b + 9 + 10   imagery + labels + dataset.yaml
  6  train_segmentation    CELL 11            YOLOv8-seg training
  7  export_onnx           CELL 14            FP32 ONNX export
  8  quantize_model        CELL 15..18        INT8 calibration/quant/metrics
  9  finalize_results      CELL 12 + 13 + 19  eval + figures + run summary
  10 compare_groundtruth   CELL 21            GT vs prediction, mean IoU
  11 predict_area          CELL 20            optional arbitrary-bbox inference
  12 log_mlflow            new                 MLflow tracking (local fallback)
  13 upload_outputs        new                 MinIO upload of outputs/
"""
from __future__ import annotations

import importlib
import json
import os
import time

from src.config import *

STAGES = [
    ("cleanup_previous_run", "src.data", "stage_cleanup_previous_run"),
    ("fetch_dataset", "src.data", "stage_fetch_dataset"),
    ("fetch_region", "src.data", "stage_fetch_region"),
    ("prepare_data", "src.data", "stage_prepare_data"),
    ("build_dataset", "src.dataset", "stage_build_dataset"),
    ("train_segmentation", "src.train", "stage_train_segmentation"),
    ("export_onnx", "src.export", "stage_export_onnx"),
    ("quantize_model", "src.quantize", "stage_quantize_model"),
    ("finalize_results", "src.evaluate", "stage_finalize_results"),
    ("compare_groundtruth", "src.evaluate", "stage_compare_groundtruth"),
    ("predict_area", "src.inference", "stage_predict_area"),
    ("log_mlflow", "src.track", "log_run"),
    ("upload_outputs", "src.storage", "upload_run_outputs"),
]

STAGE_NAMES = [name for name, _module, _fn in STAGES]
PIPELINE_STAGES = [name for name in STAGE_NAMES
                   if name not in ("log_mlflow", "upload_outputs")]


def resolve_stage(name):
    """Import the stage function on demand (cheap for `list` / `--help`)."""
    for stage_name, module_name, fn_name in STAGES:
        if stage_name == name:
            module = importlib.import_module(module_name)
            return getattr(module, fn_name)
    raise KeyError(f"unknown stage {name!r}; valid stages: {', '.join(STAGE_NAMES)}")


def _jsonable(obj):
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return json.loads(json.dumps(obj, default=str))


def run_stage(name, run):
    """Run one stage and return a uniform, JSON-serialisable record."""
    fn = resolve_stage(name)
    # MLOps: mirror this stage's log() lines into the run's own log file
    # (mlops_runs/<run_id>/outputs/logs/pipeline.log).
    set_active_run_log(os.path.join(run.out_logs, "pipeline.log"))
    log(f"[STAGE] >>> {name} (run {run.run_id})")
    started = time.time()
    result = fn(run)
    if result is None:
        result = {}
    elif not isinstance(result, dict):
        result = {"value": result}
    elapsed = round(time.time() - started, 2)
    log(f"[STAGE] <<< {name} ok in {elapsed}s")
    return dict(stage=name, ok=True, seconds=elapsed, result=_jsonable(result))


def _write_result(run, payload):
    path = os.path.join(run.out_summary, "pipeline_result.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)
    return path


def run_pipeline(run_id, base_dir=None, *, stages=None,
                 log_mlflow=True, upload_outputs=True):
    """Run the pipeline for one run_id; returns the pipeline_result payload.

    On failure the partial result is still written to
    outputs/summary/pipeline_result.json and the exception is re-raised, so
    both the CLI and the Airflow task fail visibly with a usable artefact.
    """
    run = RunPaths.build(run_id, base_dir)
    set_active_run_log(os.path.join(run.out_logs, "pipeline.log"))
    log(f"[PIPELINE] run_id={run.run_id} run_dir={run.run_dir}")

    if stages is None:
        names = list(PIPELINE_STAGES)
    else:
        names = list(stages)
        unknown = [n for n in names if n not in STAGE_NAMES]
        if unknown:
            raise KeyError(f"unknown stage(s) {unknown}; "
                           f"valid stages: {', '.join(STAGE_NAMES)}")

    results = []
    started = time.time()
    try:
        for name in names:
            results.append(run_stage(name, run))
        if log_mlflow:
            results.append(run_stage("log_mlflow", run))
        if upload_outputs:
            results.append(run_stage("upload_outputs", run))
    except Exception as exc:
        payload = dict(run_id=run.run_id, run=run.to_dict(), stages=results,
                       ok=False, error=str(exc),
                       seconds=round(time.time() - started, 2))
        path = _write_result(run, payload)
        log(f"[PIPELINE] FAILED during {results[-1]['stage'] if results else 'startup'}: "
            f"{exc} (partial result in {path})")
        raise

    payload = dict(run_id=run.run_id, run=run.to_dict(), stages=results,
                   ok=True, seconds=round(time.time() - started, 2))
    path = _write_result(run, payload)
    log(f"[PIPELINE] done in {payload['seconds']}s -> {path}")
    return payload
