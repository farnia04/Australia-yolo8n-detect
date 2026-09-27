"""Functional test of the first four stages against a local fixture.

  python tests/test_stages.py

No network, no GPU: the fixture's national FlatGeobuf index makes
fetch_dataset take its "national index available" path, fetch_region rebuild
the city index through the fast extract, and prepare_data run the real
sampling / leakage-safe split code (zero block overlap is asserted).

The fixture directory must contain australia_buildings.fgb (real file) and
patch_histogram.npz; point at one with AUS_TEST_FIXTURE_DATA=<dir>.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.environ.get(
    "AUS_TEST_FIXTURE_DATA",
    r"C:\Users\hamid\AppData\Local\Temp\opencode\aus_city\data")
PY = sys.executable

work = tempfile.mkdtemp(prefix="mlops_stages_")
os.makedirs(os.path.join(work, "data"))
# Some fixtures store the national FGB as a directory holding the real file;
# promote it to the plain file path the pipeline expects.
_fgb = os.path.join(FIXTURE, "australia_buildings.fgb")
_inner = os.path.join(_fgb, "australia_buildings.fgb.fgb")
shutil.copy(_inner if os.path.isdir(_fgb) else _fgb,
            os.path.join(work, "data", "australia_buildings.fgb"))
shutil.copy(os.path.join(FIXTURE, "patch_histogram.npz"),
            os.path.join(work, "data", "patch_histogram.npz"))

env = dict(os.environ)
env.update({
    "AUS_WORK_DIR": work,
    "AUS_MOUNT_DRIVE": "0",
    "MLOPS_MINIO_ENABLED": "0",
    "MLOPS_MLFLOW_ENABLED": "0",
    "PYTHONPATH": PROJECT,
    "PIPELINE_ENV_FILE": os.path.join(PROJECT, "configs", "pipeline.env"),
})


def run_stage(stage, run_id="run_t1"):
    proc = subprocess.run([PY, "run_cli.py", stage, run_id], cwd=PROJECT, env=env,
                          capture_output=True, text=True, timeout=900)
    tail = "\n".join(proc.stdout.splitlines()[-14:])
    print(f"\n=== {stage} (rc={proc.returncode}) ===\n{tail}")
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:])
        raise SystemExit(f"stage {stage} failed")
    lines = proc.stdout.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("{"))
    return json.loads("\n".join(lines[start:]))


# --------------------------------------------------------------- 0. cleanup
result = run_stage("cleanup_previous_run")
assert result["ok"] is True, result
assert result["result"]["old_scope"] == "sydney", result
print("cleanup removed:", result["result"]["removed"])

# ---------------------------------------------------------------- 1. dataset
result = run_stage("fetch_dataset")
assert result["ok"] is True, result
raw_path = os.path.join(work, "mlops_runs", "run_t1", "inputs", "dataset",
                        "raw_dataset.json")
assert os.path.exists(raw_path), raw_path
raw = json.load(open(raw_path, encoding="utf-8"))
assert raw["index_ready"] is False and raw["national_ready"] is True, raw
print("raw_dataset.json:", raw)

# ------------------------------------------------------------------ 2. index
result = run_stage("fetch_region")
assert result["ok"] is True, result
assert result["result"]["index_patches"] > 0, result
assert result["result"]["buildings_indexed"] > 0, result
log_text = open(os.path.join(work, "logs", "pipeline.log"), encoding="utf-8").read()
assert "national index found -> extracting" in log_text, "expected fast extract"
print("index result:", result["result"])

# index metadata was recorded for future staleness checks
meta_path = os.path.join(work, "data", "sydney_dense_index_meta.json")
assert os.path.exists(meta_path), meta_path
meta = json.load(open(meta_path, encoding="utf-8"))
assert meta["scope"] == "sydney_dense", meta
assert meta["city_bbox"] == [151.07, -34.0, 151.3, -33.76], meta
print("index_meta.json:", meta)

# ------------------------------------------------------------------ 3. split
result = run_stage("prepare_data")
assert result["ok"] is True, result
counts = result["result"]["split_counts"]
overlap = result["result"]["block_overlap"]
print("split_counts:", counts, "| block_overlap:", overlap)
assert set(counts) == {"train", "val", "test"}, counts
assert all(v > 0 for v in counts.values()), counts
assert set(overlap.values()) == {0}, overlap
files = glob.glob(os.path.join(work, "splits", "sydney_dense", "*.txt"))
assert len(files) == 3, files
assert all(os.path.getsize(f) > 0 for f in files), files

# per-run log was mirrored into the run's own outputs/logs/
run_log = os.path.join(work, "mlops_runs", "run_t1", "outputs", "logs",
                       "pipeline.log")
assert os.path.exists(run_log), run_log
run_log_text = open(run_log, encoding="utf-8").read()
assert "[STAGE] >>> fetch_region" in run_log_text
assert "[MLOPS]" in run_log_text
print("per-run log OK ->", run_log)

print("\nSTAGE TESTS PASSED (cleanup_previous_run, fetch_dataset, "
      "fetch_region, prepare_data)")
