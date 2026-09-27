"""FP32 ONNX export stage - CELL 14 (defs hoisted so quantize can reuse them)."""

from src.config import *

from src.helpers import *

from ultralytics import YOLO


import shutil
import onnx
import onnxruntime as ort
from collections import Counter

ONNX_IMGSZ = onnx_imgsz or yolo_imgsz


def export_fp32_onnx(weights_path, out_path, imgsz, opset):
    """Ultralytics YOLOv8-seg -> ONNX. simplify=True gives a cleaner graph that
    quantizes better; we retry without it if the simplifier is unavailable."""
    m = YOLO(weights_path)
    common = dict(format="onnx", imgsz=imgsz, opset=opset, dynamic=False,
                  half=False, device="cpu")
    try:
        exported = m.export(simplify=True, **common)
    except Exception as e:
        print(f"[export] simplify=True failed ({e}); retrying simplify=False")
        exported = m.export(simplify=False, **common)
    shutil.copy(str(exported), out_path)
    return out_path


def make_ort_session(path, cpu_only=True):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"] if cpu_only else ort.get_available_providers()
    return ort.InferenceSession(path, sess_options=so, providers=providers)


_EXPORT_META = os.path.join(ONNX_DIR, "_export_meta.json")


def _export_is_current():
    if not os.path.exists(ONNX_FP32) or not os.path.exists(_EXPORT_META):
        return False
    try:
        meta = json.load(open(_EXPORT_META))
    except Exception:
        return False
    return (os.path.exists(BEST_PT)
            and meta.get("opset") == onnx_opset and meta.get("imgsz") == ONNX_IMGSZ
            and meta.get("weights_mtime") == os.path.getmtime(BEST_PT))


def _onnx_runs(path):
    """Load + execute the graph on CPU with a dummy input; True if usable."""
    try:
        s = make_ort_session(path)
        x = np.zeros((1, 3, ONNX_IMGSZ, ONNX_IMGSZ), np.float32)
        s.run(None, {s.get_inputs()[0].name: x})
        return True
    except Exception as e:
        log(f"[WARN] ONNX validation failed for {os.path.basename(path)}: {e}")
        return False



def stage_export_onnx(run):

    """Stage 7 - export best.pt -> best_fp32.onnx (CELL 14)."""

    # ---- CELL 14: export flow ----------------------------------------------

    if quantization_enabled:
        if resume and _export_is_current() and _onnx_runs(ONNX_FP32):
            log("[CHECK] FP32 ONNX already valid and matches best.pt -> SKIP export")
        else:
            log("[EXPORT] exporting best.pt -> best_fp32.onnx (FP32, NOT quantized)")
            export_fp32_onnx(BEST_PT, ONNX_FP32, ONNX_IMGSZ, onnx_opset)
            json.dump({"opset": onnx_opset, "imgsz": ONNX_IMGSZ,
                       "weights_mtime": os.path.getmtime(BEST_PT)}, open(_EXPORT_META, "w"))
            if not _onnx_runs(ONNX_FP32):
                raise RuntimeError("exported FP32 ONNX does not run on CPU; export failed")
        set_stage("onnx_export", "complete")
        log(f"[CHECK] FP32 ONNX: {ONNX_FP32} ({os.path.getsize(ONNX_FP32)/1e6:.2f} MB) "
            f"(opset={onnx_opset}, imgsz={ONNX_IMGSZ})")

        _fp32_sess = make_ort_session(ONNX_FP32)
        FP32_INPUT_NAME = _fp32_sess.get_inputs()[0].name
        log(f"[CHECK] ONNX input : {[(i.name, i.shape, i.type) for i in _fp32_sess.get_inputs()]}")
        log(f"[CHECK] ONNX output: {[(o.name, o.shape, o.type) for o in _fp32_sess.get_outputs()]}")
    else:
        log("[CHECK] quantization_enabled=False -> skipping the ONNX / INT8 stage")

    # ---- MLOps: publish the FP32 graph next to the run artifacts ------------
    os.makedirs(run.out_models, exist_ok=True)
    _size_mb = None
    if os.path.exists(ONNX_FP32):
        shutil.copy(ONNX_FP32, os.path.join(run.out_models, os.path.basename(ONNX_FP32)))
        _size_mb = round(os.path.getsize(ONNX_FP32) / 1e6, 2)
        log(f"[MLOPS] FP32 ONNX -> {run.out_models} ({_size_mb} MB)")
    return dict(fp32_onnx=ONNX_FP32, size_mb=_size_mb,
                quantization_enabled=bool(quantization_enabled))
