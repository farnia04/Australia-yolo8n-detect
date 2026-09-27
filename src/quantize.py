"""INT8 quantization stage - CELL 15..18 in a single process.


Those four cells share function-local state (CALIB_PATHS, _fp32_sess,

_int8_sess, quant_status, quant_report), so they must run in one stage;

the prologue rebuilds what they read from CELL 14's module scope.

"""

from src.config import *

from src.helpers import *

from src.export import make_ort_session, _onnx_runs   # MLOps: shared ONNX helpers



def stage_quantize_model(run):

    """Stage 8 - INT8 calibration, quantization, CPU test, FP32-vs-INT8 metrics."""

    # ---- MLOps prologue: globals CELL 15..18 read ---------------------------
    import glob
    import onnx
    from collections import Counter
    ONNX_IMGSZ = onnx_imgsz or yolo_imgsz          # CELL 14's module constant
    assert os.path.exists(ONNX_FP32), (
        f"FP32 ONNX missing: {ONNX_FP32} -> run the export_onnx stage first")
    _fp32_sess = make_ort_session(ONNX_FP32)
    FP32_INPUT_NAME = _fp32_sess.get_inputs()[0].name

    # ---- CELL 15: INT8 calibration (train split only) ----------------------

    import cv2
    from onnxruntime.quantization import (quantize_static, QuantFormat, QuantType,
                                          CalibrationMethod, CalibrationDataReader)


    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
        """Resize + pad keeping aspect ratio (same as Ultralytics)."""
        shape = im.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = (new_shape[1] - new_unpad[0]) / 2.0
        dh = (new_shape[0] - new_unpad[1]) / 2.0
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        return cv2.copyMakeBorder(im, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=color)


    def preprocess_onnx(img_path, imgsz):
        """Return NCHW float32 RGB tensor in [0,1] with batch dim (1,3,H,W)."""
        im = cv2.imread(str(img_path))
        if im is None:
            raise FileNotFoundError(img_path)
        im = letterbox(im, (imgsz, imgsz))
        im = im[:, :, ::-1]                                   # BGR -> RGB
        im = np.ascontiguousarray(im.transpose(2, 0, 1), dtype=np.float32) / 255.0
        return im[None]


    class YoloCalibReader(CalibrationDataReader):
        """Feeds calibration images to onnxruntime.quantization."""

        def __init__(self, paths, imgsz, input_name):
            self.input_name = input_name
            self._data = [{input_name: preprocess_onnx(p, imgsz)} for p in paths]
            self._iter = iter(self._data)

        def get_next(self):
            return next(self._iter, None)

        def rewind(self):
            self._iter = iter(self._data)


    def calibration_paths(n):
        """Training images only; fall back to val (never test) if train is empty."""
        train = sorted(glob.glob(os.path.join(YOLO_DIR, "images", "train", "*.jpg")))
        if not train:
            train = sorted(glob.glob(os.path.join(YOLO_DIR, "images", "val", "*.jpg")))
            print("[calib] no train images found, using val split for calibration")
        if not train:
            raise RuntimeError("no images available for calibration")
        return train[:n]


    # persist the calibration set so it is identical on every restart (train only)
    if quantization_enabled:
        _reuse_calib = False
        if resume and os.path.exists(CALIB_JSON):
            try:
                _saved = json.load(open(CALIB_JSON))
                if (_saved.get("imgsz") == ONNX_IMGSZ
                        and _saved.get("n") == quant_calib_samples
                        and _saved.get("paths")
                        and all(os.path.exists(p) for p in _saved["paths"])):
                    CALIB_PATHS = _saved["paths"]
                    _reuse_calib = True
            except Exception:
                _reuse_calib = False
        if not _reuse_calib:
            CALIB_PATHS = calibration_paths(quant_calib_samples)
            tmp = CALIB_JSON + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"imgsz": ONNX_IMGSZ, "n": quant_calib_samples,
                           "paths": CALIB_PATHS}, f, indent=2)
            os.replace(tmp, CALIB_JSON)
        log(f"[CHECK] calibration set: {len(CALIB_PATHS)} train images "
            f"({'reused' if _reuse_calib else 'created'})")
    else:
        CALIB_PATHS = []

    # ---- CELL 16: static INT8 quantization + hard verification ------------

    _ACT = {"uint8": QuantType.QUInt8, "int8": QuantType.QInt8}[quant_activation_type]
    _WGT = {"int8": QuantType.QInt8, "uint8": QuantType.QUInt8}[quant_weight_type]


    def analyze_onnx_graph(path):
        """Inspect an ONNX graph and report the real quantization content."""
        g = onnx.load(path).graph
        op_hist = Counter(n.op_type for n in g.node)
        init_dtype = {t.name: t.data_type for t in g.initializer}
        int8_w = [t.name for t in g.initializer if t.data_type == onnx.TensorProto.INT8]
        uint8_w = [t.name for t in g.initializer if t.data_type == onnx.TensorProto.UINT8]
        float_w = [t.name for t in g.initializer if t.data_type == onnx.TensorProto.FLOAT]
        producer = {o: n for n in g.node for o in n.output}

        def weight_is_int8(weight_name):
            """True if a weight input resolves (directly or via DequantizeLinear)
            to an INT8 initializer."""
            if init_dtype.get(weight_name) == onnx.TensorProto.INT8:
                return True
            p = producer.get(weight_name)
            if p is not None and p.op_type == "DequantizeLinear" and p.input:
                return init_dtype.get(p.input[0]) == onnx.TensorProto.INT8
            return False

        conv_total = op_hist.get("Conv", 0)
        conv_int8 = sum(1 for n in g.node
                        if n.op_type == "Conv" and len(n.input) > 1
                        and weight_is_int8(n.input[1]))
        matmul_total = op_hist.get("MatMul", 0) + op_hist.get("Gemm", 0)
        matmul_int8 = sum(1 for n in g.node
                          if n.op_type in ("MatMul", "Gemm") and len(n.input) > 1
                          and weight_is_int8(n.input[1]))
        # op types that are genuinely left in FP32 (excludes compute ops that ORT
        # quantizes by wrapping them with Q/DQ pairs)
        qdq_and_compute = {"QuantizeLinear", "DequantizeLinear", "QLinearConv",
                           "QLinearMatMul", "ConvInteger", "MatMulInteger",
                           "DynamicQuantizeLinear", "Conv", "MatMul", "Gemm"}
        fp32_ops = sorted({op for op in op_hist if op not in qdq_and_compute})
        return dict(
            quantize_linear_nodes=op_hist.get("QuantizeLinear", 0),
            dequantize_linear_nodes=op_hist.get("DequantizeLinear", 0),
            int8_weight_initializers=len(int8_w),
            uint8_initializers=len(uint8_w),
            float_weight_initializers=len(float_w),
            conv_nodes=conv_total,
            conv_nodes_with_int8_weights=conv_int8,
            matmul_nodes=matmul_total,
            matmul_nodes_with_int8_weights=matmul_int8,
            op_type_histogram=dict(sorted(op_hist.items(), key=lambda kv: -kv[1])),
            fp32_remaining_op_types=fp32_ops,
        )


    def output_producer_names(path):
        """Names of the nodes that directly produce the graph outputs."""
        g = onnx.load(path).graph
        prod = {o: n.name for n in g.node for o in n.output}
        return [prod[o.name] for o in g.output if o.name in prod]


    def quantize_onnx(src, dst, calib_reader, nodes_to_exclude):
        """Static PTQ. Returns the QuantFormat string that succeeded."""
        common = dict(calibrate_method=CalibrationMethod.MinMax,
                      per_channel=quant_per_channel,
                      activation_type=_ACT, weight_type=_WGT,
                      nodes_to_exclude=nodes_to_exclude or None,
                      extra_options={"ActivationSymmetric": False,
                                     "WeightSymmetric": True})
        try:
            calib_reader.rewind()
            quantize_static(src, dst, calib_reader, quant_format=QuantFormat.QDQ, **common)
            return "QDQ"
        except Exception as e:
            print(f"[quant] QDQ failed: {e}\n[quant] falling back to QOperator")
            calib_reader.rewind()
            quantize_static(src, dst, calib_reader, quant_format=QuantFormat.QOperator, **common)
            return "QOperator"


    quant_status = {"attempted": False, "succeeded": False, "format": None}
    quant_report = None


    def _int8_is_current():
        """True if best_int8.onnx exists, was built from THIS best_fp32.onnx and
        THIS calibration set, and its meta is readable."""
        if not (os.path.exists(ONNX_INT8) and os.path.exists(QUANT_META)):
            return False
        try:
            m = json.load(open(QUANT_META))
        except Exception:
            return False
        return (m.get("fp32_mtime") == os.path.getmtime(ONNX_FP32)
                and m.get("calib") == CALIB_PATHS
                and m.get("opset") == onnx_opset
                and m.get("per_channel") == quant_per_channel)


    if quantization_enabled:
        quant_status["attempted"] = True
        if resume and _int8_is_current() and _onnx_runs(ONNX_INT8):
            _meta = json.load(open(QUANT_META))
            used_format = _meta.get("format", "QDQ")
            excluded_nodes = _meta.get("excluded_nodes", [])
            quant_report = analyze_onnx_graph(ONNX_INT8)
            quant_report["excluded_output_nodes"] = excluded_nodes
            _int8_sess = make_ort_session(ONNX_INT8)
            quant_status["succeeded"] = True
            quant_status["format"] = used_format
            log(f"[CHECK] INT8 model already valid -> SKIP quantization "
                f"({os.path.getsize(ONNX_INT8)/1e6:.2f} MB, format={used_format})")
            set_stage("quantization", "complete")
        else:
            log("[QUANTIZE] INT8 model not found/valid -> starting static quantization")
            # optional graph pre-processing (shape inference + symbolic shape fix)
            _pre = os.path.join(ONNX_DIR, "_best_fp32_pre.onnx")
            try:
                from onnxruntime.quantization.shape_inference import quant_pre_process
                quant_pre_process(ONNX_FP32, _pre, skip_symbolic_shape=False)
                _quant_src = _pre
                log("[QUANTIZE] quant_pre_process OK")
            except Exception as e:
                log(f"[QUANTIZE] quant_pre_process skipped ({e}); using raw FP32 graph")
                _quant_src = ONNX_FP32

            # the nodes that produce the graph outputs stay FP32 (see config comment)
            excluded_nodes = output_producer_names(_quant_src) if quant_exclude_output_nodes else []
            log(f"[QUANTIZE] keeping {len(excluded_nodes)} output node(s) in FP32: {excluded_nodes}")

            calib = YoloCalibReader(CALIB_PATHS, ONNX_IMGSZ, FP32_INPUT_NAME)
            used_format = quantize_onnx(_quant_src, ONNX_INT8, calib, excluded_nodes)

            quant_report = analyze_onnx_graph(ONNX_INT8)
            quant_report["excluded_output_nodes"] = excluded_nodes
            log("quantized graph report: " + json.dumps(
                {k: v for k, v in quant_report.items() if k != "op_type_histogram"}))
            log("quantized op histogram: " + json.dumps(
                {k: v for k, v in quant_report["op_type_histogram"].items()
                 if k in ("QuantizeLinear", "DequantizeLinear", "Conv", "MatMul", "Sigmoid",
                          "Resize", "Concat", "Mul", "Split", "MaxPool", "Add")}))

            # HARD verification: real INT8 ops must exist, and it must run on CPU
            if quant_report["quantize_linear_nodes"] == 0 or quant_report["int8_weight_initializers"] == 0:
                raise RuntimeError(
                    "INT8 quantization did NOT produce a quantized graph "
                    "(no QuantizeLinear / int8 weights). Refusing to label it INT8.")
            try:
                _int8_sess = make_ort_session(ONNX_INT8)
                _probe = preprocess_onnx(CALIB_PATHS[0], ONNX_IMGSZ)
                _oi = _int8_sess.run(None, {FP32_INPUT_NAME: _probe})
            except Exception as e:
                raise RuntimeError(
                    "Quantized ONNX is present but NOT runnable, so it is NOT usable INT8.\n"
                    f"  reason: {e}\n"
                    "  hints: use onnx_opset>=13 (needed for QDQ per-channel bias 'axis'),\n"
                    "         or set quant_per_channel=False, or retry with QOperator."
                ) from e

            # sanity: the INT8 head must not collapse (class scores must survive)
            _of = _fp32_sess.run(None, {FP32_INPUT_NAME: _probe})
            _o0f, _o0i = np.asarray(_of[0])[0], np.asarray(_oi[0])[0]
            _nc = _o0f.shape[0] - 4 - 32
            if _nc >= 1 and _o0f[4:4 + _nc].max() > 1e-3 and _o0i[4:4 + _nc].max() < 1e-4:
                raise RuntimeError(
                    "INT8 collapsed the class-score channels to 0. Keep "
                    "quant_exclude_output_nodes=True (or exclude the head in FP32).")

            # publish the quant meta atomically so the next run can skip
            tmp = QUANT_META + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"fp32_mtime": os.path.getmtime(ONNX_FP32),
                           "calib": CALIB_PATHS, "opset": onnx_opset,
                           "per_channel": quant_per_channel, "format": used_format,
                           "excluded_nodes": excluded_nodes}, f, indent=2)
            os.replace(tmp, QUANT_META)

            quant_status["succeeded"] = True
            quant_status["format"] = used_format
            log(f"[QUANTIZE] INT8 ONNX: {ONNX_INT8} ({os.path.getsize(ONNX_INT8)/1e6:.2f} MB) "
                f"format={used_format} -> runs on CPUExecutionProvider OK")
            set_stage("quantization", "complete")

    # ---- CELL 17: CPU inference test (onnxruntime) -------------------------

    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))


    def _nms_np(boxes, scores, iou_thres):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-9)
            order = order[1:][iou <= iou_thres]
        return np.array(keep, dtype=np.int64)


    def _crop_mask(masks, boxes):
        n, h, w = masks.shape
        ys = np.arange(h)[None, :, None]
        xs = np.arange(w)[None, None, :]
        x1, y1, x2, y2 = (boxes[:, k][:, None, None] for k in range(4))
        return masks * ((xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2))


    def postprocess_yolov8_seg(outputs, imgsz, conf_thres=0.25, iou_thres=0.45,
                               nc=1, nm=32):
        """Decode raw YOLOv8-seg ONNX outputs -> boxes/scores/classes/masks."""
        pred = proto = None
        for o in outputs:
            o = np.asarray(o)
            if o.ndim == 3 and o.shape[1] == 4 + nc + nm:
                pred = o[0]
            elif o.ndim == 4 and o.shape[1] == nm:
                proto = o[0]
        if pred is None or proto is None:
            raise ValueError(f"unexpected ONNX output shapes: {[np.asarray(o).shape for o in outputs]}")

        box = pred[:4].T                      # (N,4) cx,cy,w,h in input px
        cls_scores = pred[4:4 + nc].T         # (N,nc)
        coeffs = pred[4 + nc:].T              # (N,nm)
        conf = cls_scores.max(1)
        cls = cls_scores.argmax(1)
        m = conf >= conf_thres
        box, coeffs, conf, cls = box[m], coeffs[m], conf[m], cls[m]

        empty = dict(boxes=np.zeros((0, 4)), scores=np.zeros(0),
                     classes=np.zeros(0, dtype=int), masks=np.zeros((0, imgsz, imgsz)))
        if box.shape[0] == 0:
            return empty

        xyxy = np.empty_like(box)
        xyxy[:, 0] = box[:, 0] - box[:, 2] / 2
        xyxy[:, 1] = box[:, 1] - box[:, 3] / 2
        xyxy[:, 2] = box[:, 0] + box[:, 2] / 2
        xyxy[:, 3] = box[:, 1] + box[:, 3] / 2
        k = _nms_np(xyxy, conf, iou_thres)
        xyxy, coeffs, conf, cls = xyxy[k], coeffs[k], conf[k], cls[k]

        c, mh, mw = proto.shape
        masks = _sigmoid(coeffs @ proto.reshape(c, -1)).reshape(-1, mh, mw)
        masks = _crop_mask(masks, xyxy / (imgsz / mh))
        up = np.stack([cv2.resize(mm, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
                       for mm in masks]) if len(masks) else np.zeros((0, imgsz, imgsz))
        return dict(boxes=xyxy, scores=conf, classes=cls, masks=up)


    def run_onnx_cpu(session, input_name, img_path, imgsz):
        x = preprocess_onnx(img_path, imgsz)
        return session.run(None, {input_name: x}), x


    if quantization_enabled and cpu_inference_enabled:
        INT8_INPUT_NAME = _int8_sess.get_inputs()[0].name
        test_imgs_cpu = CALIB_PATHS[:min(4, len(CALIB_PATHS))]
        n_ok = 0
        for p in test_imgs_cpu:
            outs, _ = run_onnx_cpu(_int8_sess, INT8_INPUT_NAME, p, ONNX_IMGSZ)
            det = postprocess_yolov8_seg(outs, ONNX_IMGSZ)
            finite = all(np.isfinite(np.asarray(o)).all() for o in outs)
            if finite:
                n_ok += 1
                try:  # save a segmentation overlay (no GPU, no PyTorch needed)
                    im = cv2.imread(str(p))
                    h, w = im.shape[:2]
                    for m in det["masks"]:
                        mb = cv2.resize((m > 0.5).astype(np.uint8), (w, h),
                                        interpolation=cv2.INTER_NEAREST)
                        im[mb > 0] = (0.5 * im[mb > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
                    cv2.imwrite(os.path.join(PRED_DIR, "int8_" + os.path.basename(p)), im)
                except Exception as e:
                    log(f"[WARN] could not render prediction for {os.path.basename(p)}: {e}")
            log(f"[CPU] INT8 {os.path.basename(p)}: outputs={[np.asarray(o).shape for o in outs]} "
                f"finite={finite} detections={len(det['scores'])}")
        log(f"[CPU] INT8 inference test passed on {n_ok}/{len(test_imgs_cpu)} images "
            f"-> overlays in {PRED_DIR}")
        set_stage("cpu_inference", "complete")

    # ---- CELL 18: FP32 vs INT8 comparison + size + benchmark + metrics -----

    def mask_iou(a, b):
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return float(inter / union) if union else 0.0


    def best_match_iou(masks_a, masks_b, thresh=0.5):
        """Greedy best-match mean IoU between two mask sets (binary at `thresh`)."""
        if len(masks_a) == 0 or len(masks_b) == 0:
            return 0.0, 0
        A = masks_a > thresh
        B = masks_b > thresh
        ious = []
        for a in A:
            ious.append(max(mask_iou(a, b) for b in B))
        return float(np.mean(ious)), len(ious)


    def compare_fp32_int8(n_samples):
        val_imgs = sorted(glob.glob(os.path.join(YOLO_DIR, "images", "val", "*.jpg")))
        if not val_imgs:
            val_imgs = sorted(glob.glob(os.path.join(YOLO_DIR, "images", "test", "*.jpg")))
        imgs = val_imgs[:n_samples]
        if not imgs:
            return None
        fp32_sess = make_ort_session(ONNX_FP32)
        int8_sess = make_ort_session(ONNX_INT8)
        fi, ii = fp32_sess.get_inputs()[0].name, int8_sess.get_inputs()[0].name

        abs_errs, max_err, det_fp32, det_int8, ious, n = [], 0.0, 0, 0, [], 0
        for p in tqdm(imgs, desc="compare FP32 vs INT8", unit="img"):
            of = fp32_sess.run(None, {fi: preprocess_onnx(p, ONNX_IMGSZ)})
            oi = int8_sess.run(None, {ii: preprocess_onnx(p, ONNX_IMGSZ)})
            for a, b in zip(of, oi):
                a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
                if a.shape != b.shape:
                    continue
                d = np.abs(a - b)
                abs_errs.append(float(d.mean()))
                max_err = max(max_err, float(d.max()))
            df = postprocess_yolov8_seg(of, ONNX_IMGSZ)
            di = postprocess_yolov8_seg(oi, ONNX_IMGSZ)
            det_fp32 += len(df["scores"])
            det_int8 += len(di["scores"])
            miou, matched = best_match_iou(df["masks"], di["masks"])
            if matched:
                ious.append(miou)
            n += 1
        return dict(
            compared_samples=n,
            mean_absolute_error=float(np.mean(abs_errs)) if abs_errs else 0.0,
            max_absolute_error=float(max_err),
            detections_fp32=det_fp32,
            detections_int8=det_int8,
            mean_best_match_mask_iou=float(np.mean(ious)) if ious else 0.0,
        )


    # ============================
    # MODEL SIZE / COMPRESSION + CPU BENCHMARK + METRICS FILE
    # ============================
    def cpu_benchmark(n_runs, n_warmup):
        fp32_sess = make_ort_session(ONNX_FP32)
        int8_sess = make_ort_session(ONNX_INT8)
        paths = CALIB_PATHS[:min(4, len(CALIB_PATHS))]
        xs = [preprocess_onnx(p, ONNX_IMGSZ) for p in paths]
        fi, ii = fp32_sess.get_inputs()[0].name, int8_sess.get_inputs()[0].name

        def timeit(sess, name):
            for i in range(n_warmup):
                sess.run(None, {name: xs[i % len(xs)]})
            t0 = time.perf_counter()
            for i in range(n_runs):
                sess.run(None, {name: xs[i % len(xs)]})
            return (time.perf_counter() - t0) / n_runs * 1000.0

        return timeit(fp32_sess, fi), timeit(int8_sess, ii)


    if quantization_enabled and quant_status["succeeded"]:
        metrics = {
            "quantization_bits": quant_bits,
            "quantization_type": "static",
            "quantization_format": quant_status["format"],
            "weights_dtype": quant_weight_type,
            "activation_dtype": quant_activation_type,
            "per_channel": quant_per_channel,
            "calibration_samples": len(CALIB_PATHS),
            "calibration_source": "train split",
            "source_model_path": BEST_PT,
            "float_model_path": ONNX_FP32,
            "quantized_model_path": ONNX_INT8,
            "float_model_size_mb": round(os.path.getsize(ONNX_FP32) / 1e6, 4),
            "quantized_model_size_mb": round(os.path.getsize(ONNX_INT8) / 1e6, 4),
            "compression_ratio": round(os.path.getsize(ONNX_FP32) /
                                       os.path.getsize(ONNX_INT8), 4),
            "quantize_linear_nodes": quant_report["quantize_linear_nodes"],
            "dequantize_linear_nodes": quant_report["dequantize_linear_nodes"],
            "int8_weight_initializers": quant_report["int8_weight_initializers"],
            "uint8_initializers": quant_report["uint8_initializers"],
            "float_weight_initializers": quant_report["float_weight_initializers"],
            "conv_nodes": quant_report["conv_nodes"],
            "conv_nodes_with_int8_weights": quant_report["conv_nodes_with_int8_weights"],
            "matmul_nodes": quant_report["matmul_nodes"],
            "matmul_nodes_with_int8_weights": quant_report["matmul_nodes_with_int8_weights"],
            "excluded_output_nodes_fp32": quant_report["excluded_output_nodes"],
            "fp32_remaining_op_types": quant_report["fp32_remaining_op_types"],
            "cpu_inference_success": True,
        }

        cmp_res = compare_fp32_int8(onnx_compare_samples)
        if cmp_res:
            metrics.update(cmp_res)

        if cpu_inference_enabled:
            t_fp32, t_int8 = cpu_benchmark(onnx_bench_runs, onnx_bench_warmup)
            metrics.update(
                fp32_inference_ms=round(t_fp32, 2),
                int8_inference_ms=round(t_int8, 2),
                cpu_speedup=round(t_fp32 / t_int8, 3) if t_int8 else None,
                benchmark_runs=onnx_bench_runs,
            )
            print(f"CPU benchmark: FP32 {t_fp32:.2f} ms/img | INT8 {t_int8:.2f} ms/img "
                  f"| speedup {t_fp32 / t_int8:.2f}x")

        with open(QUANT_METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
        print("quantization metrics ->", QUANT_METRICS_PATH)
        print(json.dumps({k: v for k, v in metrics.items()
                          if k != "fp32_remaining_op_types"}, indent=2))
    else:
        print("quantization not completed -> no quantization_metrics.json written")

    # ---- MLOps: publish the INT8 graph + metrics ----------------------------
    os.makedirs(run.out_models, exist_ok=True)
    os.makedirs(run.out_metrics, exist_ok=True)
    _size_mb = None
    if os.path.exists(ONNX_INT8):
        shutil.copy(ONNX_INT8, os.path.join(run.out_models, os.path.basename(ONNX_INT8)))
        _size_mb = round(os.path.getsize(ONNX_INT8) / 1e6, 2)
    if os.path.exists(QUANT_METRICS_PATH):
        shutil.copy(QUANT_METRICS_PATH,
                    os.path.join(run.out_metrics, "quantization_metrics.json"))
        log(f"[MLOPS] quantization metrics -> {run.out_metrics}")
    return dict(int8_onnx=ONNX_INT8, size_mb=_size_mb,
                format=(quant_status or {}).get("format"),
                succeeded=bool((quant_status or {}).get("succeeded")),
                metrics_path=QUANT_METRICS_PATH)
