"""Smoke test for aus_yolo_mlops: imports, RunPaths, stage registry, CLI,
MLflow fallback and the storage gate. No network, no training.

  python tests/test_smoke.py
"""
import json
import os
import subprocess
import sys
import tempfile

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
WORK = tempfile.mkdtemp(prefix="mlops_work_")

env = dict(os.environ)
env.update({
    "AUS_WORK_DIR": WORK,
    "AUS_MOUNT_DRIVE": "0",
    "MLOPS_RUNS_DIR": os.path.join(WORK, "mlops_runs"),
    "MLOPS_MINIO_ENABLED": "0",
    "PYTHONPATH": PROJECT,
    "PIPELINE_ENV_FILE": os.path.join(PROJECT, "configs", "pipeline.env"),
})

# ---------------------------------------------------------------- run in-process
sys.path.insert(0, PROJECT)
os.environ.update({k: v for k, v in env.items() if k.startswith(("AUS_", "MLOPS_", "PIPELINE_"))})

import src.config as cfg                                  # noqa: E402

assert cfg.work_dir == WORK, cfg.work_dir
assert cfg.MLOPS_RUNS_DIR == os.path.join(WORK, "mlops_runs"), cfg.MLOPS_RUNS_DIR

run = cfg.RunPaths.build("run_001")
assert os.path.isdir(run.out_models) and os.path.isdir(run.out_groundtruth)
assert run.run_dir == os.path.join(WORK, "mlops_runs", "run_001"), run.run_dir
for bad in ("", "../evil", "a/b", "a\\b"):
    try:
        cfg.RunPaths.build(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"run_id {bad!r} must be rejected")
print("RunPaths OK")

import src.pipeline as pipeline                           # noqa: E402

assert pipeline.STAGE_NAMES == [
    "cleanup_previous_run", "fetch_dataset", "fetch_region", "prepare_data",
    "build_dataset", "train_segmentation", "export_onnx", "quantize_model",
    "finalize_results", "compare_groundtruth", "predict_area", "log_mlflow",
    "upload_outputs",
], pipeline.STAGE_NAMES
assert pipeline.PIPELINE_STAGES == pipeline.STAGE_NAMES[:11], pipeline.PIPELINE_STAGES
for name in pipeline.STAGE_NAMES:
    fn = pipeline.resolve_stage(name)
    assert callable(fn), name
print("stage registry OK (all 13 resolve)")

# every stage module imports (module-level cell code runs: config, helpers, ...)
import src.data, src.dataset, src.train, src.export                        # noqa: F401,E402
import src.quantize, src.evaluate, src.inference, src.track, src.storage    # noqa: F401,E402
print("all stage modules import OK")

# ------------------------------------------------------- MLflow fallback path
os.environ["MLOPS_MLFLOW_ENABLED"] = "1"
os.environ["MLOPS_MLFLOW_TRACKING_URI"] = "http://127.0.0.1:59999"   # nothing listens
import importlib
importlib.reload(cfg)
importlib.reload(src.track)
from src.track import log_run                                # noqa: E402

res = log_run(cfg.RunPaths.build("run_001"))
assert res["enabled"] is False, res
assert os.path.exists(res["summary"]), res
summary = json.load(open(res["summary"], encoding="utf-8"))
assert summary["run_id"] == "run_001"
print("MLflow fallback OK ->", res["summary"])

# ------------------------------------------------------------ storage gate
os.environ["MLOPS_MINIO_ENABLED"] = "0"
importlib.reload(cfg)
from src.storage import upload_run_outputs                   # noqa: E402

up = upload_run_outputs(cfg.RunPaths.build("run_001"))
assert up.get("skipped") is True, up
print("storage gate OK")

# ------------------------------------------------------------------- CLI
def cli(*args):
    return subprocess.run([PY, "run_cli.py", *args], cwd=PROJECT, env=env,
                          capture_output=True, text=True, timeout=300)

proc = cli("--help")
assert proc.returncode == 0, proc.stderr
proc = cli("list")
assert proc.returncode == 0, proc.stderr
assert "cleanup_previous_run" in proc.stdout and "upload_outputs" in proc.stdout
proc = cli("nonsense", "run_001")
assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
proc = cli("prepare_data")                      # missing run_id
assert proc.returncode == 2, proc.returncode
proc = cli("upload_outputs", "run_001")
assert proc.returncode == 0, proc.stderr
lines = proc.stdout.splitlines()
start = next(i for i, line in enumerate(lines) if line.startswith("{"))
out = json.loads("\n".join(lines[start:]))
assert out["stage"] == "upload_outputs" and out["ok"] is True, out
assert out["result"].get("skipped") is True, out
print("CLI OK (help/list/bad-stage/missing-run_id/upload_outputs)")

print("\nALL SMOKE TESTS PASSED")
