"""Evaluation stages: finalise a run (CELL 12/13/19) and compare the

predictions with the Kaggle ground truth (CELL 21).


stage_finalize_results starts with CELL 11b's split/manifest section (the

bridge): everything CELL 12..19 reads but that an earlier cell produced is

rebuilt from disk here, exactly like the notebook does after a restart.

"""

from src.config import *

from src.helpers import *

from ultralytics import YOLO

from src.train import run_with_oom_fallback, build_batch_ladder

from src.data import load_split_files

from src.dataset import interleave_patches



def stage_finalize_results(run):

    """Stage 9 - evaluate val+test (CELL 12), visualise (CELL 13), run summary (CELL 19)."""

    # ---- MLOps prologue: verify weights, rebuild the CELL 11b bridge --------
    assert os.path.exists(YAML_PATH), f"dataset.yaml missing: {YAML_PATH}"
    assert os.path.exists(BEST_PT), (
        f"best.pt missing: {BEST_PT} -> run the train_segmentation stage first")
    assert os.path.exists(LAST_PT), f"last.pt missing: {LAST_PT}"
    BATCH_LADDER = build_batch_ladder()

    # ---- CELL 11b section 5: split + block sets + download summary --------

    splits = load_split_files()
    assert splits, (f"no split files under {SPLITS_DIR} -> run CELL 7 on a "
                    "GPU runtime first")
    assert os.path.exists(HIST_PATH), f"patch histogram missing: {HIST_PATH}"

    _meta = {}
    if os.path.exists(SPLIT_META):
        _meta = json.load(open(SPLIT_META, encoding="utf-8"))
        log(f"[BRIDGE] split_meta: {json.dumps(_meta)}")

    # block size MUST be the value the split was generated with -> read it
    # back, never recompute it (a mismatch would fake a "leakage" alarm).
    block_patches_eff = int(_meta.get("block_patches", block_patches))
    min_buildings_eff = int(_meta.get("min_buildings", min_buildings))
    train_frac = float(_meta.get("train_frac", train_frac))
    val_frac = float(_meta.get("val_frac", val_frac))
    test_frac = float(_meta.get("test_frac", test_frac))

    def block_set(name):
        return {(p // block_patches_eff, q // block_patches_eff)
                for p, q, _ in splits[name]}

    tr_b, va_b, te_b = block_set("train"), block_set("val"), block_set("test")
    _ov = {"train&val": len(tr_b & va_b), "train&test": len(tr_b & te_b),
           "val&test": len(va_b & te_b)}
    log(f"[BRIDGE] block overlap: {json.dumps(_ov)}")
    assert not (tr_b & va_b) and not (tr_b & te_b) and not (va_b & te_b), \
        "block overlap -> split files on Drive are inconsistent"

    # building counts exactly as CELL 7 left them (same mask, same array)
    _h = np.load(HIST_PATH)
    px_all, py_all, cnt_all = _h["px"], _h["py"], _h["cnt"]
    _m = cnt_all >= min_buildings_eff
    px_all, py_all, cnt_all = px_all[_m], py_all[_m], cnt_all[_m]
    del _h, _m

    # patch work list exactly as CELL 9 built it
    all_patches = interleave_patches(
        splits,
        (round(train_frac * 10), round(val_frac * 10), round(test_frac * 10)))
    if build_limit is not None:
        all_patches = all_patches[:build_limit]

    manifest = load_manifest()
    built = {n: 0 for n in ("train", "val", "test")}
    _inv = collections.Counter()
    _failed = 0
    for _r in manifest.values():
        _st = _r.get("status")
        if _st == "complete" and _r.get("split") in built:
            built[_r["split"]] += 1
        elif _st == "invalid":
            _inv[_r.get("error", "")] += 1
        elif _st == "failed":
            _failed += 1
    # CELL 9's per-run `skipped` counter cannot be replayed from an
    # append-only manifest, so it is reported as 0 here; blank/no_polygons
    # are the real, deduplicated counts of rejected patches.
    skipped = collections.Counter(blank=_inv.get("blank imagery", 0),
                                  no_polygons=_inv.get("no polygons", 0),
                                  skipped=0, failed=_failed)
    log(f"[BRIDGE] patches: total={len(all_patches):,} built={json.dumps(built)} "
        f"skipped={json.dumps(dict(skipped))} "
        f"buildings_indexed={int(cnt_all.sum()):,}")

    # ---- CELL 12: evaluate on val and test --------------------------------

    best = os.path.join(RUNS_DIR, "aus_buildings_seg", "weights", "best.pt")
    trained = YOLO(best)

    for split in ("val", "test"):
        _val_out = {}

        def _val_attempt(b, _r=_val_out, _s=split):
            # MEMORY FIX: dense images can OOM validation as well. `batch` only
            # CHUNKS the evaluation (same images, same detections, same mAP) - it
            # never changes what is measured, so metrics are unaffected.
            _r["m"] = trained.val(data=YAML_PATH, split=_s, imgsz=yolo_imgsz,
                                  device=device, batch=b, project=RUNS_DIR,
                                  name=f"eval_{_s}", exist_ok=True)

        # remember=False: a val OOM must NOT shrink the training batch history
        run_with_oom_fallback(_val_attempt, BATCH_LADDER, label=f"eval_{split}",
                              remember=False)
        m = _val_out["m"]
        print(f"[{split}] mask mAP50={m.seg.map50:.4f} mAP50-95={m.seg.map:.4f} | "
              f"box mAP50={m.box.map50:.4f} mAP50-95={m.box.map:.4f}")

    # ---- CELL 13: visualize predictions on the test split -----------------

    import glob
    import matplotlib.pyplot as plt

    test_imgs = sorted(glob.glob(os.path.join(YOLO_DIR, "images", "test", "*.jpg")))[:6]
    if test_imgs:
        preds = trained.predict(test_imgs, imgsz=yolo_imgsz, conf=0.25, device=device)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        for ax, res in zip(axes.ravel(), preds):
            plotted = res.plot()[..., ::-1]  # BGR -> RGB
            ax.imshow(plotted)
            ax.axis("off")
        plt.tight_layout()
        out_png = os.path.join(RUNS_DIR, "test_predictions.png")
        plt.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.show()
        print("saved:", out_png)

    # ---- CELL 19: save a run summary --------------------------------------

    summary = dict(
        profile=PROFILE,
        config=CONFIG,
        patches_total=len(all_patches),
        built=built,
        skipped=dict(skipped),
        split_blocks={k: len(v) for k, v in (("train", tr_b), ("val", va_b), ("test", te_b))},
        buildings_indexed=int(cnt_all.sum()),
        artifacts=dict(work_dir=work_dir, best_pt=BEST_PT, fp32_onnx=ONNX_FP32,
                       int8_onnx=ONNX_INT8, metrics=QUANT_METRICS_PATH,
                       manifest=MANIFEST_PATH, state=STATE_PATH, splits=SPLITS_DIR,
                       predictions=PRED_DIR, checkpoints=CHECKPOINT_DIR),
    )
    with open(os.path.join(work_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log("[SUMMARY] " + json.dumps({k: v for k, v in summary.items()
                                   if k not in ("config", "artifacts")}))
    log(f"[DONE] all artifacts in {work_dir}")

    # ---- MLOps: publish evaluation metrics, figures and the summary ---------
    os.makedirs(run.out_metrics, exist_ok=True)
    os.makedirs(run.out_visualizations, exist_ok=True)
    os.makedirs(run.out_summary, exist_ok=True)
    _eval = {}
    for _split in ("val", "test"):
        _csv = os.path.join(RUNS_DIR, f"eval_{_split}", "results.csv")
        if not os.path.exists(_csv):
            continue
        try:
            _lines = [ln for ln in open(_csv, encoding="utf-8").read().splitlines()
                      if ln.strip()]
            _hdr = [c.strip() for c in _lines[0].split(",")]
            _last = [c.strip() for c in _lines[-1].split(",")]
            _eval[_split] = dict(zip(_hdr, _last))
        except Exception as _e:
            log(f"[MLOPS] could not parse {_csv}: {_e}")
    _tmp = os.path.join(run.out_metrics, "eval_metrics.json.tmp")
    with open(_tmp, "w", encoding="utf-8") as f:
        json.dump(_eval, f, indent=2)
    os.replace(_tmp, os.path.join(run.out_metrics, "eval_metrics.json"))
    _png = os.path.join(RUNS_DIR, "test_predictions.png")
    if os.path.exists(_png):
        shutil.copy(_png, os.path.join(run.out_visualizations, "test_predictions.png"))
    _summary_src = os.path.join(work_dir, "run_summary.json")
    if os.path.exists(_summary_src):
        shutil.copy(_summary_src, os.path.join(run.out_summary, "pipeline_summary.json"))
    log(f"[MLOPS] evaluation -> {run.out_metrics}")
    return dict(eval_metrics=_eval,
                summary=os.path.join(run.out_summary, "pipeline_summary.json"),
                run_summary=_summary_src)



def stage_compare_groundtruth(run):

    """Stage 10 - GREEN ground truth vs RED predictions (CELL 21), mean mask IoU."""

    # ---- MLOps prologue: verify inputs and load the trained model -----------
    assert os.path.exists(YAML_PATH), f"dataset.yaml missing: {YAML_PATH}"
    assert os.path.exists(BEST_PT), (
        f"best.pt missing: {BEST_PT} -> run the train_segmentation stage first")
    splits = load_split_files()
    assert splits, (
        f"no split files under {SPLITS_DIR} -> run the prepare_data stage first")
    trained = YOLO(BEST_PT)
    OUT_GT = run.out_groundtruth       # MLOps: figures + CSV/JSON land here
    os.makedirs(OUT_GT, exist_ok=True)

    # ---- CELL 21: compare predictions against the Kaggle ground truth ------

    EVAL_N = 10             # number of test patches compared
    EVAL_CONF = 0.25        # prediction confidence threshold (same as CELL 13)
    CROP = 384              # centre crop shown per image (px)

    import csv
    from PIL import ImageDraw
    import matplotlib.pyplot as plt


    def _gt_mask(p, q):
        """Rasterise the Kaggle ground truth for one patch with the same code path
        the training labels use, so the comparison is apples-to-apples."""
        bbox = patch_bbox_lonlat(p, q)
        ox, oy = p * patch_tiles, q * patch_tiles
        lines = []
        with fiona.open(FGB_PATH) as src:
            for feat in src.filter(bbox=bbox):
                try:
                    lines.extend(geom_to_yolo_lines(shape(feat["geometry"]), bbox, ox, oy))
                except Exception:
                    continue
        m = Image.new("L", (PATCH_PX, PATCH_PX), 0)
        draw = ImageDraw.Draw(m)
        for ln in lines:
            vals = [float(v) for v in ln.split()[1:]]
            pts = [(vals[i] * PATCH_PX, vals[i + 1] * PATCH_PX)
                   for i in range(0, len(vals) - 1, 2)]
            if len(pts) >= 3:
                draw.polygon(pts, fill=255)
        return np.asarray(m) > 0


    def _pred_mask(res):
        """Predicted building mask at ORIGINAL patch resolution (res.masks.xy is
        already in original-image pixels, so no scaling is involved)."""
        m = Image.new("L", (PATCH_PX, PATCH_PX), 0)
        draw = ImageDraw.Draw(m)
        if res.masks is not None and len(res.masks):
            for poly in res.masks.xy:
                pts = [(float(x), float(y)) for x, y in np.asarray(poly)]
                if len(pts) >= 3:
                    draw.polygon(pts, fill=255)
        return np.asarray(m) > 0


    def _blend(im, mask, color):
        """Half-blend `color` into `im` where `mask` is True (broadcast-safe)."""
        m = mask[..., None]
        im[m] = (0.5 * im[m] + 0.5 * np.array(color, dtype=np.float32)).astype(np.uint8)


    test_patches = list(splits["test"])
    assert test_patches, "no test patches -> run prepare_data first"
    _eval_rng = np.random.default_rng(0)          # deterministic sample
    _pick = sorted(_eval_rng.choice(len(test_patches),
                                    size=min(EVAL_N, len(test_patches)),
                                    replace=False).tolist())
    # metres per pixel at this latitude (same formula as CELL 8) -> real areas in m2
    patch_m = PATCH_PX * 156543.03 * math.cos(math.radians(-28.0)) / (2 ** zoom)
    mpp = patch_m / PATCH_PX

    results = []
    for _i, _k in enumerate(_pick):
        p, q, cnt = test_patches[_k]
        img = stitch_patch(p, q)
        if img is None:
            print(f"[GT] patch {p}_{q}: imagery unavailable -> skipped")
            continue
        rgb = np.asarray(img.convert("RGB"))
        gt = _gt_mask(p, q)
        res = trained.predict(img, imgsz=yolo_imgsz, conf=EVAL_CONF,
                              device=device, verbose=False)[0]
        pr = _pred_mask(res)

        inter = int(np.logical_and(gt, pr).sum())
        union = int(np.logical_or(gt, pr).sum())
        iou = inter / union if union else 1.0
        real_area = float(gt.sum()) * mpp * mpp
        det_area = float(pr.sum()) * mpp * mpp
        results.append(dict(patch=f"{p}_{q}", buildings=int(cnt),
                            real_area_m2=round(real_area, 1),
                            detected_area_m2=round(det_area, 1),
                            pixels_intersection=inter, pixels_union=union,
                            iou=round(float(iou), 4)))   # MLOps: collected for CSV/JSON/MLflow

        # ---- display: centre crop so 768 px -> CROP px of real zoom -------------
        h, w = gt.shape
        y0 = max(0, min(h - CROP, h // 2 - CROP // 2))
        x0 = max(0, min(w - CROP, w // 2 - CROP // 2))
        view = rgb[y0:y0 + CROP, x0:x0 + CROP].copy()
        _blend(view, gt[y0:y0 + CROP, x0:x0 + CROP], (0, 255, 0))   # GREEN = ground truth
        _blend(view, pr[y0:y0 + CROP, x0:x0 + CROP], (255, 0, 0))   # RED   = prediction

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(view)
        ax.set_title(f"{p}_{q}  real={real_area:,.0f} m2  det={det_area:,.0f} m2  "
                     f"IoU={iou:.3f}", fontsize=9)
        ax.axis("off")
        out_png = os.path.join(OUT_GT, f"compare_{_i:02d}_{p}_{q}.png")
        fig.savefig(out_png, dpi=120, bbox_inches="tight")   # MLOps: keep the figure on disk
        plt.close(fig)                                       # MLOps: free memory in batch runs
        print(f"[GT] {p}_{q}: real={real_area:,.0f} m2 detected={det_area:,.0f} m2 "
              f"IoU={iou:.3f}")

    mean_iou = float(np.mean([r["iou"] for r in results])) if results else None
    print(f"[GT] compared {len(results)} test patches | mean IoU = {mean_iou}")

    # --- MLOps: persist the comparison table next to the figures -----------------
    os.makedirs(OUT_GT, exist_ok=True)
    _csv_path = os.path.join(OUT_GT, "compare_groundtruth.csv")
    with open(_csv_path, "w", encoding="utf-8", newline="") as f:
        _w = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else
                            ["patch", "buildings", "real_area_m2", "detected_area_m2",
                             "pixels_intersection", "pixels_union", "iou"])
        _w.writeheader()
        for _r in results:
            _w.writerow(_r)
    _json_path = os.path.join(OUT_GT, "compare_groundtruth.json")
    with open(_json_path, "w", encoding="utf-8") as f:
        json.dump(dict(n_compared=len(results), mean_iou=mean_iou,
                       eval_n=EVAL_N, eval_conf=EVAL_CONF, crop_px=CROP,
                       results=results), f, indent=2)
    print("saved:", _csv_path)
    print("saved:", _json_path)

    # ---- MLOps: expose the headline metric to the pipeline result -----------
    _gt_json = os.path.join(OUT_GT, "compare_groundtruth.json")
    _gt = {}
    if os.path.exists(_gt_json):
        with open(_gt_json, encoding="utf-8") as f:
            _gt = json.load(f)
    log(f"[MLOPS] ground-truth comparison -> {OUT_GT} "
        f"(mean IoU {_gt.get('mean_iou')} over {_gt.get('n_compared')} patches)")
    return dict(mean_iou=_gt.get("mean_iou"), n_compared=_gt.get("n_compared"),
                csv_path=os.path.join(OUT_GT, "compare_groundtruth.csv"),
                figures_dir=OUT_GT)
