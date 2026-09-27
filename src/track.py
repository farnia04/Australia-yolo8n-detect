"""MLflow tracking for the aus_yolo_mlops pipeline.

Template pattern: params come from the CONFIG dict, metrics are collected from
the run's outputs/metrics + outputs/groundtruth files, and everything falls
back to a local JSON summary when MLflow is disabled or unreachable - a run
must never fail because the tracking server is down.
"""
from __future__ import annotations

import json
import math
import os
import re

from src.config import *


def _metric_key(name):
    """Make a column name MLflow-safe: 'metrics/mAP50-95(B)' -> 'metrics_mAP50_95_B'."""
    key = re.sub(r"[^0-9A-Za-z_]+", "_", str(name))
    key = re.sub(r"_+", "_", key).strip("_")
    return key or "metric"


def _to_number(value):
    """Return value as a finite float, or None when it is not a usable metric."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except Exception:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _flatten_numbers(data, prefix, out):
    for key, value in (data or {}).items():
        if isinstance(value, dict):
            _flatten_numbers(value, f"{prefix}{key}_", out)
            continue
        number = _to_number(value)
        if number is not None:
            out[_metric_key(f"{prefix}{key}")] = number


def collect_params(run):
    """CONFIG + run identity as MLflow params (strings, bounded length)."""
    params = {"run_id": run.run_id, "scope": SCOPE, "run_dir": run.run_dir}
    for key, value in CONFIG.items():
        if isinstance(value, (tuple, list)):
            text = json.dumps(list(value))
        elif isinstance(value, dict):
            text = json.dumps(value)
        elif value is None:
            text = "None"
        else:
            text = str(value)
        params[str(key)] = text[:1000]
    return params


def collect_metrics(run):
    """Numeric metrics produced by the stages of this run."""
    metrics = {}

    training = _load_json(os.path.join(run.out_metrics, "training_metrics.json"))
    _flatten_numbers(training.get("results", {}), "train_", metrics)
    for key in ("epochs_done", "epochs_target"):
        number = _to_number(training.get(key))
        if number is not None:
            metrics[_metric_key(f"train_{key}")] = number

    for split, values in _load_json(os.path.join(run.out_metrics, "eval_metrics.json")).items():
        _flatten_numbers(values, f"eval_{split}_", metrics)

    quant = _load_json(os.path.join(run.out_metrics, "quantization_metrics.json"))
    _flatten_numbers(quant, "quant_", metrics)

    ground = _load_json(os.path.join(run.out_groundtruth, "compare_groundtruth.json"))
    for key in ("mean_iou", "n_compared"):
        number = _to_number(ground.get(key))
        if number is not None:
            metrics[_metric_key(f"gt_{key}")] = number

    return metrics


def _write_local_summary(run, params, metrics, tracking):
    os.makedirs(run.out_mlruns, exist_ok=True)
    path = os.path.join(run.out_mlruns, "run_metrics.json")
    payload = dict(run_id=run.run_id, run_dir=run.run_dir, tracking=tracking,
                   n_params=len(params), n_metrics=len(metrics),
                   params=params, metrics=metrics)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)
    return path


def log_run(run):
    """Stage 11 - log params/metrics/artifacts to MLflow (local fallback always)."""
    params = collect_params(run)
    metrics = collect_metrics(run)
    summary_path = None

    if not MLOPS_MLFLOW_ENABLED:
        tracking = dict(enabled=False, reason="MLOPS_MLFLOW_ENABLED=0")
        summary_path = _write_local_summary(run, params, metrics, tracking)
        log(f"[MLFLOW] disabled -> local summary {summary_path}")
        return dict(tracking, summary=summary_path, n_metrics=len(metrics))

    try:
        import mlflow
    except Exception as e:
        tracking = dict(enabled=False, reason=f"mlflow not installed: {e}")
        summary_path = _write_local_summary(run, params, metrics, tracking)
        log(f"[MLFLOW] not installed -> local summary {summary_path}")
        return dict(tracking, summary=summary_path, n_metrics=len(metrics))

    try:
        mlflow.set_tracking_uri(MLOPS_MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLOPS_MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=run.run_id) as active:
            mlflow.log_params(params)
            for key, value in metrics.items():
                mlflow.log_metric(key, float(value))
            mlflow.set_tag("pipeline", "aus_yolo_mlops")
            mlflow.set_tag("run_id", run.run_id)
            mlflow.set_tag("scope", SCOPE)
            # small text artifacts only (.pt/.onnx stay on disk / in MinIO)
            for folder, artifact_path in ((run.out_metrics, "metrics"),
                                          (run.out_groundtruth, "groundtruth"),
                                          (run.out_summary, "summary")):
                if not os.path.isdir(folder):
                    continue
                for name in sorted(os.listdir(folder)):
                    if name.lower().endswith((".json", ".csv")):
                        mlflow.log_artifact(os.path.join(folder, name),
                                            artifact_path=artifact_path)
            run_id = active.info.run_id
        tracking = dict(enabled=True, mlflow_run_id=run_id,
                         tracking_uri=MLOPS_MLFLOW_TRACKING_URI,
                         experiment=MLOPS_MLFLOW_EXPERIMENT)
        summary_path = _write_local_summary(run, params, metrics, tracking)
        log(f"[MLFLOW] logged run {run_id} ({len(metrics)} metrics) "
            f"-> {MLOPS_MLFLOW_TRACKING_URI}")
        return dict(tracking, summary=summary_path, n_metrics=len(metrics))
    except Exception as e:
        # Tracking must never take the pipeline down: keep the local summary.
        tracking = dict(enabled=False, error=str(e))
        summary_path = _write_local_summary(run, params, metrics, tracking)
        log(f"[MLFLOW] tracking failed ({e}) -> local summary kept {summary_path}")
        return dict(tracking, summary=summary_path, n_metrics=len(metrics))
