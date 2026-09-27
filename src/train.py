"""Training stage - CELL 11, resumable and CUDA-OOM safe.


Hoisted to module level (definitions only, no side effects):

  * checkpoint / OOM helper functions

  * train_args

  * build_batch_ladder()  (CELL 11's ladder block + prior-OOM history)

  * _train_from_or_resume()

The stage body is CELL 11's original flow (skip / resume / fresh).

"""

from src.config import *

from src.helpers import *

from ultralytics import YOLO


os.makedirs(CHECKPOINT_DIR, exist_ok=True)


RUN_DIR = os.path.join(RUNS_DIR, "aus_buildings_seg")
RESULTS_CSV = os.path.join(RUN_DIR, "results.csv")


def completed_epochs():
    """Best estimate of finished epochs. Ultralytics' strip_optimizer sets
    ckpt['epoch'] = -1 on a clean finish, so results.csv is authoritative;
    a non-stripped last.pt (interrupted run) gives a more current value."""
    done = 0
    if os.path.exists(RESULTS_CSV):
        try:
            with open(RESULTS_CSV, encoding="utf-8") as f:
                rows = [ln for ln in f.read().splitlines()[1:] if ln.strip()]
            if rows:
                done = int(rows[-1].split(",")[0])
        except Exception:
            pass
    if os.path.exists(LAST_PT):
        try:
            e = int(torch.load(LAST_PT, map_location="cpu", weights_only=False).get("epoch", -1))
            if e >= 0:                       # interrupted (not stripped) checkpoint
                done = max(done, e + 1)
        except Exception:
            pass
    return done


def ensure_target_epochs(path, target):
    """Ultralytics reads total epochs from the checkpoint. If the stored run was
    configured for fewer epochs, bump it so resume really trains to `target`."""
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ta = dict(ck.get("train_args", {}))
        if int(ck.get("epoch", -1)) >= 0 and int(ta.get("epochs", target)) != target:
            ta["epochs"] = target
            ck["train_args"] = ta
            torch.save(ck, path)
            return True
    except Exception as e:
        log(f"[WARN] could not update checkpoint epochs: {e}")
    return False


def is_resumable(path):
    """A checkpoint is resumable only if it still carries optimizer state
    (Ultralytics strips it, setting epoch=-1, on a clean finish)."""
    try:
        return int(torch.load(path, map_location="cpu", weights_only=False).get("epoch", -1)) >= 0
    except Exception:
        return False


def sync_checkpoints():
    for f in ("last.pt", "best.pt"):
        src = os.path.join(WEIGHTS_DIR, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(CHECKPOINT_DIR, f))
    log("[CKPT] mirrored last.pt/best.pt -> checkpoints/")


# ---- MEMORY FIX: CUDA-OOM-safe training for dense segmentation batches -------
# WHY THIS IS NEEDED: peak VRAM in a YOLOv8-seg step scales with
#   (objects in the batch) x (TaskAlignedAssigner cost matrix)
#   + (proto / mask loss tensors).
# Patches here hold hundreds to thousands of buildings, so batch=16 on a ~15 GB
# T4 overflows inside the assigner and later mid-epoch.
# WHAT IS TRADED: ONLY the micro-batch size. dataset, labels, spatial split,
# imgsz=640, model architecture, max_det, loss, thresholds and epochs are all
# untouched, and Ultralytics' gradient accumulation (nbs -> accumulate) keeps
# the EFFECTIVE batch size constant, so the optimizer updates - and therefore
# the final metrics - stay the same as the batch=16 run.
def is_cuda_oom(exc):
    """MEMORY FIX: PyTorch raises several OOM flavours (OutOfMemoryError is a
    RuntimeError subclass since 1.13; older builds raise plain RuntimeError)."""
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return ("out of memory" in msg
                or "cudamalloc failed" in msg
                or "hip out of memory" in msg
                or "cuda error: out of memory" in msg)
    return False


def free_gpu_memory():
    """MEMORY FIX: hand cached allocator blocks back after an OOM so the retry
    starts from a clean pool instead of a fragmented one."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def batch_candidates(start):
    """MEMORY FIX ladder: configured batch, then halve down to 2, then 1."""
    out, b = [], max(1, int(start))
    while True:
        if b not in out:
            out.append(b)
        if b <= 1:
            break
        b = max(1, b // 2)
    if 1 not in out:
        out.append(1)
    return out


def ensure_target_batch(path, target_batch):
    """MEMORY FIX: resume=True restores `batch` from the CHECKPOINT, so a
    memory-driven reduction only takes effect after patching train_args first.
    Weights, EMA, optimizer state and epoch are carried over untouched, and the
    write is atomic (temp + os.replace), so a crash mid-save can never destroy
    last.pt."""
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ta = dict(ck.get("train_args", {}))
        if int(ta.get("batch", -1)) == int(target_batch):
            return False
        ta["batch"] = int(target_batch)
        ck["train_args"] = ta
        tmp = path + ".tmp"
        torch.save(ck, tmp)
        os.replace(tmp, path)
        return True
    except Exception as e:
        log(f"[WARN] could not update checkpoint batch: {e}")
        return False


def ensure_target_device(path, target_device):
    """MEMORY FIX: resume=True also restores `device` from the CHECKPOINT, so a
    GPU-trained last.pt would ask for device=0 on a CPU-only runtime and
    Ultralytics raises `ValueError: Invalid CUDA 'device=0' requested`. Sync it
    to whatever THIS runtime resolved (0 with CUDA, 'cpu' otherwise). Weights /
    optimizer / epoch untouched; atomic write."""
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ta = dict(ck.get("train_args", {}))
        cur = ta.get("device", None)
        if str(cur) == str(target_device):
            return False
        ta["device"] = target_device
        ck["train_args"] = ta
        tmp = path + ".tmp"
        torch.save(ck, tmp)
        os.replace(tmp, path)
        return True
    except Exception as e:
        log(f"[WARN] could not update checkpoint device: {e}")
        return False


def run_with_oom_fallback(attempt, batch_list, label="train", remember=True):
    """MEMORY FIX: run attempt(batch); on CUDA OOM free GPU memory and retry the
    SAME code path with the next smaller batch. The checkpoint is never
    deleted, so every retry continues from the last saved epoch instead of
    restarting training from scratch. remember=True records the failed/good
    micro-batch in pipeline_state so the NEXT run does not repeat the OOM."""
    for i, b in enumerate(batch_list):
        try:
            result = attempt(b)
            if remember:
                record_batch_state(good=b)
            return result
        except Exception as e:
            if not is_cuda_oom(e):
                raise
            if remember:
                record_batch_state(failed=b)   # persist BEFORE any retry/raise
            free_gpu_memory()
            if i + 1 >= len(batch_list):
                log(f"[OOM] {label}: still out of memory at batch={b} -> stopping "
                    f"this attempt; last.pt/best.pt kept ({LAST_PT})")
                raise RuntimeError(
                    f"CUDA OOM persisted down to batch={b}; checkpoint preserved, "
                    "re-run this cell to resume from last.pt") from e
            log(f"[OOM] {label}: CUDA out of memory at batch={b} -> memory freed, "
                f"retrying with batch={batch_list[i + 1]}")
    raise RuntimeError(f"{label}: empty batch ladder")


train_args = dict(data=YAML_PATH, imgsz=yolo_imgsz, batch=batch, patience=patience,
                  device=device, project=RUNS_DIR, name="aus_buildings_seg",
                  seed=seed, workers=2, exist_ok=True,
                  # MEMORY FIX: keep AMP/mixed precision ON explicitly (FP16
                  # activations roughly halve activation memory).
                  amp=True)


def build_batch_ladder():

    """MEMORY FIX: build the micro-batch ladder (CELL 11 ladder block, wrapped

    in a function so every process rebuilds it from pipeline_state)."""

    BATCH_LADDER = batch_candidates(batch)      # MEMORY FIX: [4, 2, 1] when batch=4
    # MEMORY FIX: never re-enter a micro-batch that already OOM'd in a previous run
    # (covers the case where the runtime died before this ladder could run).
    _hist_fail = STATE.get("mem", {}).get("failed_batch")
    if _hist_fail:
        _below = [b for b in BATCH_LADDER if b < int(_hist_fail)]
        BATCH_LADDER = _below if _below else BATCH_LADDER[-1:]
        log(f"[MEM] prior CUDA OOM at batch={_hist_fail} -> ladder starts below it "
            f"(delete the \"mem\" key in {os.path.basename(STATE_PATH)} to reset)")
    log(f"[MEM] batch ladder: {BATCH_LADDER} | imgsz={yolo_imgsz} | amp=True")

    return BATCH_LADDER


def _train_from_or_resume(source_weights, b):
    """MEMORY FIX: the first attempt starts from `source_weights`; if an earlier
    attempt in this ladder already saved last.pt (OOM mid-epoch), resume that
    checkpoint instead so no finished epoch is ever thrown away."""
    if (resume and os.path.exists(LAST_PT) and is_resumable(LAST_PT)
            and completed_epochs() > 0):
        if ensure_target_epochs(LAST_PT, epochs):
            log(f"[TRAIN] extended checkpoint target to {epochs} epochs")
        if ensure_target_batch(LAST_PT, b):
            log(f"[TRAIN] checkpoint batch -> {b} (MEMORY FIX)")
        if ensure_target_device(LAST_PT, device):   # MEMORY FIX: CPU/GPU safe
            log(f"[TRAIN] checkpoint device -> {device} (MEMORY FIX)")
        log(f"[TRAIN] OOM retry -> resuming last.pt at epoch {completed_epochs()}")
        YOLO(LAST_PT).train(resume=True, device=device)
        return
    ta = dict(train_args)
    ta["batch"] = b
    YOLO(source_weights).train(epochs=epochs, **ta)


# ---- MLOps: checkpoint compatibility validation -------------------------

CHECKPOINT_META = os.path.join(CHECKPOINT_DIR, "checkpoint_meta.json")


def _dataset_fingerprint():
    """Config knobs that determine the training dataset content."""
    _bbox = (list(map(float, city_bbox))
             if restrict_to_city and city_bbox is not None else None)
    return dict(scope=SCOPE, city_bbox=_bbox, zoom=int(zoom),
                patch_tiles=int(patch_tiles), seed=int(seed),
                max_patches=max_patches, min_buildings=int(min_buildings),
                train_frac=float(train_frac), val_frac=float(val_frac),
                test_frac=float(test_frac))


def _checkpoint_fingerprint():
    """What the weights in checkpoints/ were produced from. Optimizer knobs
    (batch, patience, lr) are deliberately excluded: changing them on resume
    is safe, changing the data/architecture is not."""
    return dict(model=model, yolo_imgsz=int(yolo_imgsz),
                dataset=_dataset_fingerprint())


def _read_checkpoint_meta():
    try:
        with open(CHECKPOINT_META, encoding="utf-8") as _f:
            return json.load(_f)
    except Exception:
        return None


def _write_checkpoint_meta():
    _meta = _checkpoint_fingerprint()
    _tmp = CHECKPOINT_META + ".tmp"
    with open(_tmp, "w", encoding="utf-8") as _f:
        json.dump(_meta, _f, indent=2)
    os.replace(_tmp, CHECKPOINT_META)
    return _meta


def _fingerprint_diff(stored, current):
    """Human-readable list of changed keys between two fingerprint dicts."""
    _diff = []
    for _k in sorted(set(stored) | set(current)):
        _sv, _cv = stored.get(_k), current.get(_k)
        if isinstance(_sv, dict) and isinstance(_cv, dict):
            _diff.extend(f"dataset.{_s}" for _s in _fingerprint_diff(_sv, _cv))
        elif _sv != _cv:
            _diff.append(f"{_k}: {_sv!r} -> {_cv!r}")
    return _diff


def _assert_splits_match_config():
    """Fail fast when the on-disk splits were built for different config knobs.

    prepare_data regenerates splits whenever these change, so a mismatch here
    means the user changed the config and ran train alone -> training on stale
    splits would silently produce the wrong model.
    """
    if not os.path.exists(SPLIT_META):
        raise SystemExit(
            f"no splits for this config ({SPLIT_META} missing) -> "
            "run the prepare_data stage first")
    try:
        _on_disk = json.load(open(SPLIT_META, encoding="utf-8"))
    except Exception as _e:
        raise SystemExit(f"cannot read {SPLIT_META}: {_e}")
    _expected = _dataset_fingerprint()
    _diff = [f"{_k}: {_on_disk.get(_k, '<missing>')!r} -> {_expected[_k]!r}"
             for _k in _expected if _on_disk.get(_k) != _expected[_k]]
    if _diff:
        raise SystemExit(
            "splits on disk were built for a different config "
            f"({'; '.join(_diff)}) -> re-run the prepare_data stage first")


def validate_checkpoints():
    """Allow resume only from checkpoints produced by the CURRENT config.

    A bare last.pt is never enough: if the stored fingerprint (model, imgsz,
    dataset knobs) differs - or is missing (legacy/foreign weights) - the
    files are ARCHIVED to checkpoints/stale_<timestamp>/ (never deleted) and
    training starts from scratch. Returns True when usable checkpoints remain.
    """
    _have = [p for p in (LAST_PT, BEST_PT) if os.path.exists(p)]
    if not _have:
        return True
    _stored = _read_checkpoint_meta()
    _current = _checkpoint_fingerprint()
    if _stored is not None and _stored == _current:
        log(f"[CHECK] checkpoints match current config (model={_current['model']}, "
            f"scope={_current['dataset']['scope']}) -> resume allowed")
        return True
    import time as _time
    _ts = _time.strftime("%Y%m%d_%H%M%S")
    _dest = os.path.join(CHECKPOINT_DIR, f"stale_{_ts}")
    os.makedirs(_dest, exist_ok=True)
    for _p in (LAST_PT, BEST_PT, CHECKPOINT_META):
        if os.path.exists(_p):
            os.replace(_p, os.path.join(_dest, os.path.basename(_p)))
    if _stored is None:
        _why = "no checkpoint metadata (legacy/foreign weights - provenance unknown)"
    else:
        _why = "; ".join(_fingerprint_diff(_stored, _current)) or "metadata differs"
    log(f"[CHECK] checkpoints incompatible with current config ({_why}) -> "
        f"archived to {_dest}; training will start from scratch")
    return False



def stage_train_segmentation(run):

    """Stage 6 - train YOLOv8-seg from last.pt (CELL 11)."""

    # ---- MLOps: validate splits + checkpoints BEFORE the CELL 11 flow ----
    # A bare last.pt is NOT enough to resume: the weights must come from the
    # SAME dataset + model + input size (see validate_checkpoints).
    _assert_splits_match_config()   # fail fast: stale splits are never trained on
    validate_checkpoints()          # archives incompatible weights (never deletes)
    _write_checkpoint_meta()        # record THIS run's config for future runs

    # ---- MLOps: ladder + cache flush, exactly as CELL 11 did --------------

    BATCH_LADDER = build_batch_ladder()

    free_gpu_memory()                  # MEMORY FIX: empty cache before training

    # ---- CELL 11: train (skip / resume / fresh) ---------------------------

    done = completed_epochs() if resume else 0

    if resume and done >= epochs and os.path.exists(BEST_PT):
        log(f"[CHECK] training already complete: epoch {done}/{epochs} -> SKIP")
        set_stage("training", "complete", epoch=done, total=epochs)
    elif resume and done > 0 and os.path.exists(LAST_PT) and is_resumable(LAST_PT):
        if ensure_target_epochs(LAST_PT, epochs):
            log(f"[TRAIN] extended checkpoint target to {epochs} epochs")
        log(f"[TRAIN] existing checkpoint found: epoch {done}")
        log(f"[TRAIN] resuming from epoch {done}; target epoch: {epochs}")
        set_stage("training", "in_progress", epoch=done, total=epochs)

        def _resume_attempt(b):   # MEMORY FIX: patch batch+device, then resume SAME ckpt
            if ensure_target_batch(LAST_PT, b):
                log(f"[TRAIN] checkpoint batch -> {b} (MEMORY FIX)")
            if ensure_target_device(LAST_PT, device):   # MEMORY FIX: CPU/GPU safe
                log(f"[TRAIN] checkpoint device -> {device} (MEMORY FIX)")
            YOLO(LAST_PT).train(resume=True, device=device)  # restores other args from ckpt

        run_with_oom_fallback(_resume_attempt, BATCH_LADDER, label="resume")
        sync_checkpoints()
        done = completed_epochs()
        set_stage("training", "complete" if done >= epochs else "in_progress",
                  epoch=done, total=epochs)
    elif resume and done > 0 and os.path.exists(BEST_PT):
        # previous run finished cleanly (optimizer stripped) but target grew:
        # continue training from the best weights instead of throwing work away
        log(f"[TRAIN] clean finish at epoch {done}, target {epochs} -> continuing from best.pt")
        set_stage("training", "in_progress", epoch=done, total=epochs)

        def _continue_attempt(b):  # MEMORY FIX
            _train_from_or_resume(BEST_PT, b)

        run_with_oom_fallback(_continue_attempt, BATCH_LADDER, label="continue-from-best")
        sync_checkpoints()
        done = completed_epochs()
        set_stage("training", "complete" if done >= epochs else "in_progress",
                  epoch=done, total=epochs)
    else:
        log(f"[TRAIN] no usable checkpoint -> training from scratch; target epoch: {epochs}")
        set_stage("training", "in_progress", epoch=0, total=epochs)

        def _fresh_attempt(b):     # MEMORY FIX
            _train_from_or_resume(model, b)

        run_with_oom_fallback(_fresh_attempt, BATCH_LADDER, label="train-from-scratch")
        sync_checkpoints()
        done = completed_epochs()
        set_stage("training", "complete" if done >= epochs else "in_progress",
                  epoch=done, total=epochs)

    trained = YOLO(BEST_PT)
    log(f"[TRAIN] best weights: {BEST_PT}")

    # ---- MLOps: publish weights + training metrics for the next stages ------
    os.makedirs(run.out_models, exist_ok=True)
    os.makedirs(run.out_metrics, exist_ok=True)
    for _f in ("best.pt", "last.pt"):
        _src = os.path.join(WEIGHTS_DIR, _f)
        if os.path.exists(_src):
            shutil.copy(_src, os.path.join(run.out_models, _f))
    _csv_rows = {}
    try:
        if os.path.exists(RESULTS_CSV):
            _lines = [ln for ln in open(RESULTS_CSV, encoding="utf-8").read().splitlines()
                      if ln.strip()]
            _hdr = [c.strip() for c in _lines[0].split(",")]
            _last = [c.strip() for c in _lines[-1].split(",")]
            _csv_rows = dict(zip(_hdr, _last))
    except Exception as _e:
        log(f"[MLOPS] could not parse results.csv: {_e}")
    _payload = dict(run_dir=RUN_DIR, epochs_done=int(done), epochs_target=int(epochs),
                    best_pt=BEST_PT, last_pt=LAST_PT,
                    batch_ladder=list(BATCH_LADDER), results=_csv_rows)
    _tmp = os.path.join(run.out_metrics, "training_metrics.json.tmp")
    with open(_tmp, "w", encoding="utf-8") as f:
        json.dump(_payload, f, indent=2)
    os.replace(_tmp, os.path.join(run.out_metrics, "training_metrics.json"))
    log(f"[MLOPS] training artifacts -> {run.out_models}")
    return dict(best_pt=BEST_PT, last_pt=LAST_PT, epochs_done=int(done),
                batch_ladder=list(BATCH_LADDER), results_csv=RESULTS_CSV)
