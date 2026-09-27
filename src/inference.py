"""Inference-only helpers (CELL 20) + the optional predict_area stage."""

from src.config import *

from src.helpers import *


def pixel_to_lonlat(X, Y, z, ox, oy):
    return tile_to_lonlat(ox + X / 256.0, oy + Y / 256.0, z)


def _load_seg_model(weights=None):
    from ultralytics import YOLO
    weights = weights or os.path.join(RUNS_DIR, "aus_buildings_seg", "weights", "best.pt")
    assert os.path.exists(weights), f"weights not found: {weights} (train first, or pass weights=...)"
    return YOLO(weights)


def predict_bbox(bbox, out_dir, weights=None, conf=0.25, imgsz=None, save_geojson=True):
    """bbox = (xmin, ymin, xmax, ymax) in lon/lat. Downloads imagery for every
    patch covering the bbox, predicts, and saves annotated patches + GeoJSON."""
    imgsz = imgsz or yolo_imgsz
    model = _load_seg_model(weights)
    xmin, ymin, xmax, ymax = bbox
    tx0, ty0 = lonlat_to_tile_f(xmin, ymax, zoom)   # top-left
    tx1, ty1 = lonlat_to_tile_f(xmax, ymin, zoom)   # bottom-right
    px0, py0 = int(tx0) // patch_tiles, int(ty0) // patch_tiles
    px1, py1 = int(tx1) // patch_tiles, int(ty1) // patch_tiles
    os.makedirs(out_dir, exist_ok=True)
    patches = [(px, py) for py in range(py0, py1 + 1) for px in range(px0, px1 + 1)]
    feats, n_patches, n_polys = [], 0, 0
    for px, py in tqdm(patches, desc="inference", unit="patch"):
        ox, oy = px * patch_tiles, py * patch_tiles
        needed = [(zoom, ox + dx, oy + dy)
                  for dy in range(patch_tiles) for dx in range(patch_tiles)]
        with cf.ThreadPoolExecutor(max_workers=download_workers) as ex:
            list(ex.map(lambda t: fetch_tile(*t), needed))
        img = stitch_patch(px, py)
        if img is None:
            continue
        res = model.predict(img, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        if res.masks is None or len(res.masks) == 0:
            continue
        n_patches += 1
        Image.fromarray(res.plot()[..., ::-1]).save(
            os.path.join(out_dir, f"pred_{px}_{py}.jpg"), quality=jpeg_quality)
        for poly in res.masks.xy:
            ring = [list(pixel_to_lonlat(float(X), float(Y), zoom, ox, oy)) for X, Y in poly]
            if len(ring) >= 3:
                feats.append({"type": "Feature",
                              "properties": {"patch": f"{px}_{py}"},
                              "geometry": {"type": "Polygon", "coordinates": [ring]}})
                n_polys += 1
    if save_geojson:
        with open(os.path.join(out_dir, "predictions.geojson"), "w") as f:
            json.dump({"type": "FeatureCollection", "features": feats}, f)
    print(f"inference done: {n_patches} patches with buildings, "
          f"{n_polys} polygons -> {out_dir}")
    return feats


def predict_images(paths, out_dir, weights=None, conf=0.25, imgsz=None):
    """Run the trained model on local image files and save the overlays."""
    imgsz = imgsz or yolo_imgsz
    model = _load_seg_model(weights)
    os.makedirs(out_dir, exist_ok=True)
    for p in tqdm(paths, desc="inference", unit="img"):
        res = model.predict(p, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        Image.fromarray(res.plot()[..., ::-1]).save(os.path.join(out_dir, os.path.basename(p)))
    print(f"inference done: {len(paths)} images -> {out_dir}")


# --- examples (uncomment and run after training) ----------------------------
# a small area of central Tehran:
# predict_bbox((51.38, 35.68, 51.43, 35.72),
#              out_dir=os.path.join(work_dir, "infer_tehran"), conf=0.25)
#
# or run on local images:
# predict_images(glob.glob("/content/my_images/*.jpg"),
#                out_dir=os.path.join(work_dir, "infer_local"))



def stage_predict_area(run):

    """Stage 11 - run the trained model on an arbitrary lon/lat bbox.


    Optional: set MLOPS_PREDICT_BBOX=xmin,ymin,xmax,ymax (e.g. a Tehran

    window) to exercise it; otherwise the stage reports skipped and the

    pipeline keeps going - the pipeline never depends on it.

    """

    if not MLOPS_PREDICT_BBOX:
        log("[MLOPS] MLOPS_PREDICT_BBOX not set -> predict_area SKIPPED")
        return dict(skipped=True, reason="MLOPS_PREDICT_BBOX not set")
    try:
        bbox = tuple(float(v) for v in MLOPS_PREDICT_BBOX.split(","))
        assert len(bbox) == 4
    except Exception as _e:
        raise ValueError(
            "MLOPS_PREDICT_BBOX must be xmin,ymin,xmax,ymax - got %r" % (MLOPS_PREDICT_BBOX,)
        ) from _e
    out_dir = os.path.join(run.out_predictions, "area")
    feats = predict_bbox(bbox, out_dir=out_dir, conf=0.25)
    log(f"[MLOPS] predict_area -> {out_dir} ({len(feats)} polygons)")
    return dict(bbox=list(bbox), out_dir=out_dir, polygons=len(feats))
