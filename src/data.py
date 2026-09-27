"""Data acquisition stages of the aus_yolo_mlops pipeline.


  cleanup_previous_run - narrowed CELL 2c (legacy scope + crash leftovers)

  fetch_dataset        - CELL 4 (env check) + CELL 5 (kagglehub download)

  fetch_region         - CELL 6 (FlatGeobuf index + histogram)

  prepare_data         - CELL 7 (sample + leakage-safe split) + CELL 8 (verify)


CELL 2c runs BEFORE CELL 5 on purpose: its `sydney_*` globs also match the

current scope (sydney_dense_*), so INDEX_READY must be recomputed after the

cleanup - exactly the notebook's cell order 2c -> 4 -> 5 -> 6.


Deliberate deviations from CELL 2c (see README, 'cleanup logic'):

  * checkpoints/, onnx/, predictions/, tile_cache/ are KEPT so a later run

    can resume from last.pt (every run's weights also live in

    mlops_runs/<run_id>/outputs/ + MinIO, so history is never lost).

  * the city index is NOT blindly wiped; staleness is handled by index

    metadata (invalidate_stale_index) instead.

  * logs/pipeline.log is KEPT (each run mirrors its own lines into

    mlops_runs/<run_id>/outputs/logs/pipeline.log).


CELL 7's save_split_files / load_split_files are hoisted to module level so

the build_dataset and finalize_results stages can import them.

"""

from src.config import *

from src.helpers import *

from src.helpers import _FionaGeometry   # MLOps: underscore names are not star-imported


# ---- CELL 7 hoisted: split file IO (used by CELL 7/9/11b/19) ----------

def save_split_files(splits, meta):
    for name, rows in splits.items():
        tmp = SPLIT_FILES[name] + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for p, q, c in rows:
                f.write(f"{p} {q} {c}\n")
        os.replace(tmp, SPLIT_FILES[name])
    tmp = SPLIT_META + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, SPLIT_META)


def load_split_files():
    out = {}
    for name, path in SPLIT_FILES.items():
        if not os.path.exists(path):
            return None
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    rows.append((int(parts[0]), int(parts[1]),
                                 int(parts[2]) if len(parts) > 2 else 0))
        if not rows:
            return None
        out[name] = rows
    return out


# ---- MLOps: city-index metadata (staleness validation) -----------------

def _index_meta_path():
    return os.path.join(DATA_DIR, f"{SCOPE}_index_meta.json")


def _index_fingerprint():
    """Config knobs that determine the city index content."""
    _bbox = (list(map(float, city_bbox))
             if restrict_to_city and city_bbox is not None else None)
    return dict(scope=SCOPE, city_bbox=_bbox, zoom=int(zoom),
                patch_tiles=int(patch_tiles),
                restrict_to_city=bool(restrict_to_city))


def _read_index_meta():
    try:
        with open(_index_meta_path(), encoding="utf-8") as _f:
            return json.load(_f)
    except Exception:
        return None


def _write_index_meta():
    _meta = _index_fingerprint()
    _tmp = _index_meta_path() + ".tmp"
    with open(_tmp, "w", encoding="utf-8") as _f:
        json.dump(_meta, _f, indent=2)
    os.replace(_tmp, _index_meta_path())
    return _meta


def _fingerprint_diff(stored, current):
    """Human-readable list of changed keys between two fingerprint dicts."""
    _diff = []
    for _k in sorted(set(stored) | set(current)):
        if stored.get(_k) != current.get(_k):
            _diff.append(f"{_k}: {stored.get(_k, '<missing>')!r} -> {current.get(_k)!r}")
    return _diff


def invalidate_stale_index():
    """Archive FGB+HIST when they exist but no longer match the current config.

    CELL 5/6 only test file existence, so without this a changed bbox/scope
    would silently reuse a stale index. Files are archived (never deleted) and
    the normal flow rebuilds them. Returns True when files were moved.
    """
    if not (os.path.exists(FGB_PATH) and os.path.exists(HIST_PATH)):
        return False
    _stored = _read_index_meta()
    _current = _index_fingerprint()
    if _stored is not None and _stored == _current:
        log(f"[CHECK] city index matches current config ({SCOPE}) -> reuse")
        return False
    import time as _time
    _ts = _time.strftime("%Y%m%d_%H%M%S")
    _dest = os.path.join(DATA_DIR, f"stale_index_{_ts}")
    os.makedirs(_dest, exist_ok=True)
    for _p in (FGB_PATH, HIST_PATH, _index_meta_path()):
        if os.path.exists(_p):
            if os.path.isdir(_p):
                shutil.move(_p, os.path.join(_dest, os.path.basename(_p)))
            else:
                os.replace(_p, os.path.join(_dest, os.path.basename(_p)))
    _why = ("no index metadata (legacy index)" if _stored is None
            else "; ".join(_fingerprint_diff(_stored, _current)) or "metadata differs")
    log(f"[CHECK] city index stale ({_why}) -> archived to {_dest}; will rebuild")
    return True



def stage_cleanup_previous_run(run):

    """Stage 1 - remove legacy-scope outputs + crash leftovers (narrowed CELL 2c)."""

    # ---- MLOps: narrowed CELL 2c -------------------------------------------
    # (see module docstring for what is deliberately NOT removed anymore)
    import glob as _glob

    _OLD_SCOPE = "sydney"          # the scope name the previous notebook produced
    _removed = []


    def _rm(path):
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            _removed.append("dir  " + path)
        elif os.path.exists(path):
            os.remove(path)
            _removed.append("file " + path)


    # legacy city-scope outputs (images/labels/split/runs of the old notebook)
    for _d in (os.path.join(work_dir, "yolo", _OLD_SCOPE),
               os.path.join(work_dir, "splits", _OLD_SCOPE),
               os.path.join(work_dir, "runs", _OLD_SCOPE)):
        _rm(_d)

    # legacy per-city index + histogram of the old scope only (the CURRENT
    # scope's index is validated by metadata, never blindly wiped)
    for _pat in (f"{_OLD_SCOPE}_*.fgb*", f"{_OLD_SCOPE}_*.npz*"):
        for _p in _glob.glob(os.path.join(DATA_DIR, _pat)):
            if os.path.basename(_p).startswith(SCOPE + "_"):
                continue
            _rm(_p)

    # legacy state / manifest / work_dir summary (the run-scoped copies live in
    # mlops_runs/<run_id>/; logs/pipeline.log is KEPT - each run mirrors its
    # own lines into mlops_runs/<run_id>/outputs/logs/pipeline.log)
    for _p in (os.path.join(work_dir, f"pipeline_state_{_OLD_SCOPE}.json"),
               os.path.join(work_dir, f"download_manifest_{_OLD_SCOPE}.jsonl"),
               os.path.join(work_dir, "run_summary.json")):
        _rm(_p)

    # crash leftovers: CELL 6/7/11 write "<name>.tmp*" + atomic replace, so any
    # surviving ".tmp" file is garbage from an interrupted run
    for _root in (DATA_DIR, os.path.join(work_dir, "splits")):
        if not os.path.isdir(_root):
            continue
        for _dp, _, _fns in os.walk(_root):
            for _fn in _fns:
                if ".tmp" in _fn:
                    _rm(os.path.join(_dp, _fn))

    if _removed:
        print("[CLEAN] removed old-run artifacts:")
        for _r in _removed:
            print("   " + _r)
    else:
        print("[CLEAN] nothing to remove (no old-run artifacts found)")
    print("[CLEAN] national index KEPT:",
          "australia_buildings.fgb" if os.path.exists(NATIONAL_FGB) else "(MISSING)")
    # ---- MLOps: report what the cleanup removed ----------------------------
    log(f"[MLOPS] cleanup removed {len(_removed)} entries "
        f"(old scope {_OLD_SCOPE!r})")
    return dict(old_scope=_OLD_SCOPE, removed=_removed)



def stage_fetch_dataset(run):

    """Stage 2 - environment check (CELL 4) + Kaggle download (CELL 5)."""

    # ---- MLOps: archive a stale city index BEFORE CELL 5 recomputes ----
    # INDEX_READY below so the existence check sees the validated state.
    invalidate_stale_index()

    # ---- CELL 4: environment check ----------------------------------------

    print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: no GPU -> training will be very slow. Runtime > Change runtime type > GPU")

    # ---- CELL 5: Kaggle dataset download (RESUMABLE) ----------------------

    import kagglehub

    DATASET_PATH = None
    INDEX_READY = os.path.exists(FGB_PATH) and os.path.exists(HIST_PATH)
    NATIONAL_READY = os.path.exists(NATIONAL_FGB) and os.path.exists(NATIONAL_HIST)

    if resume and (INDEX_READY or (restrict_to_city and NATIONAL_READY)):
        log("[CHECK] index available (city or national) -> SKIP Kaggle download")
        set_stage("dataset_download", "complete")
        if INDEX_READY:
            set_stage("index_build", "complete")
    else:
        if os.environ.get("AUS_DATASET_PATH"):
            DATASET_PATH = os.environ["AUS_DATASET_PATH"]
            log(f"[CHECK] using local dataset path: {DATASET_PATH}")
        else:
            kdir = os.path.expanduser("~/.kaggle")
            os.makedirs(kdir, exist_ok=True)
            if not os.path.exists(os.path.join(kdir, "kaggle.json")) and os.path.exists("/content/kaggle.json"):
                shutil.copy("/content/kaggle.json", os.path.join(kdir, "kaggle.json"))
                os.chmod(os.path.join(kdir, "kaggle.json"), 0o600)
            log("[DOWNLOAD] Kaggle dataset via kagglehub (raw GeoJSON, ~6.4 GB)")
            try:
                DATASET_PATH = kagglehub.dataset_download(dataset_handle)
            except Exception as e:
                raise SystemExit(
                    f"kagglehub failed: {e}\n"
                    "Upload your kaggle.json to /content and re-run this cell.")
            log(f"[DOWNLOAD] dataset at {DATASET_PATH}")
        set_stage("dataset_download", "complete")

    # ---- MLOps: persist what this stage resolved for fetch_region ----------
    os.makedirs(run.inputs_dataset, exist_ok=True)
    _resolved = dict(dataset_path=DATASET_PATH, index_ready=bool(INDEX_READY),
                     national_ready=bool(NATIONAL_READY), handle=dataset_handle)
    _tmp = os.path.join(run.inputs_dataset, "raw_dataset.json.tmp")
    with open(_tmp, "w", encoding="utf-8") as f:
        json.dump(_resolved, f, indent=2)
    os.replace(_tmp, os.path.join(run.inputs_dataset, "raw_dataset.json"))
    log(f"[MLOPS] dataset resolution -> {run.inputs_dataset}")
    return _resolved



def stage_fetch_region(run):

    """Stage 3 - FlatGeobuf index + histogram for the current scope (CELL 6)."""

    # ---- MLOps: rebuild the globals CELL 6 reads (resolved by fetch_dataset) ----
    _resolved_path = os.path.join(run.inputs_dataset, "raw_dataset.json")
    assert os.path.exists(_resolved_path), (
        f"missing {_resolved_path} -> run the fetch_dataset stage first")
    with open(_resolved_path, encoding="utf-8") as f:
        _resolved = json.load(f)
    DATASET_PATH = _resolved.get("dataset_path")
    INDEX_READY = bool(_resolved.get("index_ready"))

    # ---- MLOps: re-validate the index against the CURRENT config ---------
    # fetch_dataset validated it earlier, but the config may have changed since
    # (e.g. single-stage runs) -> a stale index must rebuild, never be reused.
    if INDEX_READY and _read_index_meta() != _index_fingerprint():
        log("[CHECK] city index changed since fetch_dataset -> rebuilding")
        INDEX_READY = False

    # ---- CELL 6: locate vector source -> FlatGeobuf + histogram -----------

    VECTOR_EXTS = (".geojson", ".shp", ".gpkg", ".fgb", ".parquet", ".json")


    def find_vector_sources(root):
        found = []
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn.lower().endswith(VECTOR_EXTS):
                    found.append(os.path.join(dp, fn))
        # prefer geojson/shp, and the biggest file (the national layer)
        found.sort(key=lambda p: (not p.lower().endswith((".geojson", ".shp")), -os.path.getsize(p)))
        return found


    def _city_box():
        return box(*city_bbox) if restrict_to_city else None


    def _add_feature(dst, hist, geom, g):
        """Count one building into the per-patch histogram and write it to the FGB."""
        cent = geom.centroid
        tx, ty = lonlat_to_tile_f(cent.x, cent.y, zoom)
        hist[(int(tx) // patch_tiles, int(ty) // patch_tiles)] += 1
        geom_out = _FionaGeometry.from_dict(g) if _FionaGeometry is not None else g
        dst.write({"type": "Feature", "properties": {}, "geometry": geom_out})


    def build_index(src_paths):
        """Build the FlatGeobuf + per-patch histogram for the current scope.

        FAST path: if restrict_to_city and the national index already exists, the
        city is extracted through its spatial index (seconds, no full scan).
        SLOW path: stream the raw GeoJSON once, keeping only the city bbox (or all
        polygons when restrict_to_city=False)."""
        hist = collections.Counter()
        out_schema = {"geometry": "MultiPolygon", "properties": {}}
        # unique temp names: avoids clashes with a stale/locked file from a crash
        fgb_tmp = FGB_PATH + f".tmp{os.getpid()}"
        cbox = _city_box()

        with fiona.open(fgb_tmp, "w", driver="FlatGeobuf",
                        schema=out_schema, crs="EPSG:4326") as dst:
            # ---- fast path: extract the city from the national index ------------
            if cbox is not None and os.path.exists(NATIONAL_FGB):
                log(f"[BUILD] national index found -> extracting {city_name} bbox (fast)")
                with fiona.open(NATIONAL_FGB) as src:
                    for feat in tqdm(src.filter(bbox=city_bbox),
                                     desc=f"extract {city_name}", unit="bldg"):
                        g = feat["geometry"]
                        if g is None:
                            continue
                        if g["type"] == "Polygon":
                            g = {"type": "MultiPolygon", "coordinates": [g["coordinates"]]}
                        _add_feature(dst, hist, shape(g), g)
            # ---- slow path: stream raw sources (national build / no index) ------
            else:
                assert src_paths, "no vector source and no national index available"
                for src_path in src_paths:
                    with fiona.open(src_path) as src:
                        crs = src.crs or "EPSG:4326"
                        try:
                            crs_epsg = crs.to_epsg()
                        except Exception:
                            crs_epsg = None
                        if crs_epsg not in (None, 4326):
                            from pyproj import Transformer
                            from shapely.ops import transform as shp_transform
                            tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                            reproject = lambda geom: shp_transform(
                                lambda x, y, z=None: tr.transform(x, y), geom)
                        else:
                            reproject = None
                        try:
                            total = len(src)
                        except Exception:
                            total = None
                        for feat in tqdm(src, total=total,
                                         desc=os.path.basename(src_path), unit="bldg"):
                            g = feat["geometry"]
                            if g is None:
                                continue
                            if g["type"] == "Polygon":
                                g = {"type": "MultiPolygon", "coordinates": [g["coordinates"]]}
                            try:
                                geom = shape(g)
                            except Exception:
                                continue
                            if reproject is not None:
                                geom = reproject(geom)
                                g = mapping(geom)
                            if cbox is not None and not geom.intersects(cbox):
                                continue
                            _add_feature(dst, hist, geom, g)

        px = np.fromiter((k[0] for k in hist), dtype=np.int64, count=len(hist))
        py = np.fromiter((k[1] for k in hist), dtype=np.int64, count=len(hist))
        cnt = np.fromiter(hist.values(), dtype=np.int64, count=len(hist))
        # publish the index atomically: a crash mid-pass leaves only the .tmp file,
        # which is discarded and rebuilt next run (stage-level resume)
        os.replace(fgb_tmp, FGB_PATH)
        hist_tmp = HIST_PATH + f".tmp{os.getpid()}.npz"
        np.savez_compressed(hist_tmp, px=px, py=py, cnt=cnt)
        os.replace(hist_tmp, HIST_PATH)
        return px, py, cnt


    if resume and INDEX_READY:
        d = np.load(HIST_PATH)
        px_all, py_all, cnt_all = d["px"], d["py"], d["cnt"]
        scope = city_name if restrict_to_city else "Australia"
        log(f"[CHECK] loaded cached {scope} index: {len(px_all):,} patches, "
            f"{int(cnt_all.sum()):,} buildings -> SKIP index build")
        set_stage("index_build", "complete")
    else:
        # discard any partial index left by an interrupted run (best effort)
        for _d, _base in ((os.path.dirname(FGB_PATH), os.path.basename(FGB_PATH)),
                          (os.path.dirname(HIST_PATH), os.path.basename(HIST_PATH))):
            for _fn in os.listdir(_d):
                if _fn.startswith(_base + ".tmp"):
                    try:
                        os.remove(os.path.join(_d, _fn))
                    except OSError:
                        pass
        if restrict_to_city and os.path.exists(NATIONAL_FGB):
            px_all, py_all, cnt_all = build_index(None)
        else:
            sources = find_vector_sources(DATASET_PATH) if DATASET_PATH else []
            log(f"[CHECK] vector sources found: {len(sources)}")
            for s in sources[:10]:
                log(f"    {s} ({os.path.getsize(s)/1e9:.2f} GB)")
            assert sources, "no vector file found in the downloaded dataset"
            px_all, py_all, cnt_all = build_index(sources)
        log(f"[BUILD] index ready: {len(px_all):,} patches, {int(cnt_all.sum()):,} buildings")
        set_stage("index_build", "complete")
        # optionally free ~2 GB by dropping the national index once the city is out
        if restrict_to_city and not keep_national_index and os.path.exists(NATIONAL_FGB):
            for f in (NATIONAL_FGB, NATIONAL_HIST):
                try:
                    os.remove(f)
                except OSError:
                    pass
            log("[BUILD] removed national index (keep_national_index=False)")

    _write_index_meta()   # MLOps: record what config built this index
    log(f"[MLOPS] index -> {FGB_PATH} ({len(px_all):,} patches, "
        f"{int(cnt_all.sum()):,} buildings)")
    return dict(fgb_path=FGB_PATH, hist_path=HIST_PATH,
                index_meta=_index_meta_path(),
                index_patches=int(len(px_all)),
                buildings_indexed=int(cnt_all.sum()))



def stage_prepare_data(run):

    """Stage 4 - sample patches + leakage-safe split (CELL 7), verify it (CELL 8)."""

    # ---- MLOps: rebuild the globals CELL 7 reads (CELL 6 histogram) ---------
    assert os.path.exists(HIST_PATH), (
        f"patch histogram missing: {HIST_PATH} -> run the fetch_region stage first")
    _h = np.load(HIST_PATH)
    px_all, py_all, cnt_all = _h["px"], _h["py"], _h["cnt"]
    del _h

    # ---- CELL 7: sample patches (city scope) + leakage-safe split ---------

    rng = np.random.default_rng(seed)
    mask = cnt_all >= min_buildings
    px_all, py_all, cnt_all = px_all[mask], py_all[mask], cnt_all[mask]
    log(f"[CHECK] patches with >= {min_buildings} buildings: {len(px_all):,}")




    # spatial blocks + an ADAPTIVE chunk size. A fixed 40 blocks/chunk collapses a
    # single city into 1-2 chunks, and the 8/1/1 pattern would put them all in
    # train (val/test empty). Sizing the chunk to len(blocks)//12 guarantees that
    # val and test chunks actually exist, whatever the scope.
    # keep blocks fine enough that even a small/coarse scope still has many blocks
    # (a big block_patches collapses a city at zoom 16 into ~1 block -> no val/test)
    auto_block = max(1, int(math.sqrt(max(1, len(px_all)) / 100.0)))
    block_patches_eff = max(1, min(block_patches, auto_block))
    block_x = (px_all // block_patches_eff).astype(np.int64)
    block_y = (py_all // block_patches_eff).astype(np.int64)
    blocks = sorted(set(zip(block_x.tolist(), block_y.tolist())), key=lambda b: morton(*b))
    chunk_blocks = max(1, min(split_chunk_blocks, max(1, len(blocks) // 12)))
    n_chunks = (len(blocks) + chunk_blocks - 1) // chunk_blocks
    log(f"[CHECK] blocks: {len(blocks):,} | chunks: {n_chunks:,} "
        f"(block_patches: config={block_patches} -> effective={block_patches_eff}, "
        f"chunk_blocks={chunk_blocks})")


    SPLIT_META_EXPECTED = dict(seed=seed, max_patches=max_patches, block_patches=block_patches_eff,
                               split_chunk_blocks=chunk_blocks, buffer_blocks=buffer_blocks,
                               region_patches=region_patches, min_buildings=min_buildings,
                               train_frac=train_frac, val_frac=val_frac, test_frac=test_frac,
                               zoom=zoom, patch_tiles=patch_tiles,
                               scope=SCOPE,
                               city_bbox=list(city_bbox) if restrict_to_city else None)

    loaded = load_split_files() if resume else None
    meta_ok = False
    if loaded is not None and os.path.exists(SPLIT_META):
        try:
            meta_ok = json.load(open(SPLIT_META)) == SPLIT_META_EXPECTED
        except Exception:
            meta_ok = False

    if loaded is not None and meta_ok:
        splits = loaded
        log("[CHECK] spatial split already exists and matches config -> SKIP regeneration")
        set_stage("split_generation", "complete")
    else:
        # The split changed vs the cached one. Old images/labels for this scope may
        # now belong to a different split, which would leak the same patch into two
        # splits -> wipe them (and the manifest) so everything is rebuilt cleanly.
        if resume and loaded is not None:
            for sub in ("images", "labels"):
                shutil.rmtree(os.path.join(YOLO_DIR, sub), ignore_errors=True)
            if os.path.exists(MANIFEST_PATH):
                os.remove(MANIFEST_PATH)
            log("[CHECK] split changed -> cleared cached images/labels/manifest for this scope")

        block_chunk = {b: i // chunk_blocks for i, b in enumerate(blocks)}

        if n_chunks >= 10:
            def chunk_split(chunk):          # scattered 8/1/1 pattern
                r = chunk % 10
                return "train" if r < 8 else ("val" if r == 8 else "test")
        else:
            def chunk_split(chunk):          # too few chunks to scatter -> contiguous
                frac = (chunk + 0.5) / n_chunks
                if frac < train_frac:
                    return "train"
                if frac < train_frac + val_frac:
                    return "val"
                return "test"

        block_split = {b: chunk_split(c) for b, c in block_chunk.items()}

        def conflicting(bx, by, split):
            # Only the TRAIN split is pushed away from val/test; val/test are always
            # kept so even a small city still has a usable evaluation set. This
            # leaves a buffer gap (dropped train blocks) around val/test, so train
            # patches are never spatially adjacent to val/test patches.
            if split != "train":
                return False
            for dx in range(-buffer_blocks, buffer_blocks + 1):
                for dy in range(-buffer_blocks, buffer_blocks + 1):
                    if block_split.get((bx + dx, by + dy)) in ("val", "test"):
                        return True
            return False

        splits = {"train": [], "val": [], "test": []}
        for name, frac in (("train", train_frac), ("val", val_frac), ("test", test_frac)):
            idx = np.array([i for i, (bx, by) in enumerate(zip(block_x, block_y))
                            if block_split[(int(bx), int(by))] == name
                            and not conflicting(int(bx), int(by), name)], dtype=np.int64)
            if len(idx) == 0:
                log(f"[CHECK] {name}: no patches available")
                continue
            if max_patches is None:
                chosen = idx                       # FULL run: keep every valid patch
            else:
                target = int(round(max_patches * frac))
                chosen = idx[stratified_sample(px_all[idx], py_all[idx], cnt_all[idx], target, rng)]
            for i in chosen:
                splits[name].append((int(px_all[i]), int(py_all[i]), int(cnt_all[i])))
            log(f"[CHECK] {name:5s}: {len(chosen):7d} patches "
                f"(from {len(idx):,} candidates), "
                f"{sum(c for _, _, c in splits[name]):,} buildings")
        save_split_files(splits, SPLIT_META_EXPECTED)
        set_stage("split_generation", "complete")

    # ---- CELL 8: verify there is no leakage between splits ---------------

    def block_set(name):
        return {(p // block_patches_eff, q // block_patches_eff) for p, q, _ in splits[name]}


    tr_b, va_b, te_b = block_set("train"), block_set("val"), block_set("test")
    print("block overlap train&val :", len(tr_b & va_b))
    print("block overlap train&test:", len(tr_b & te_b))
    print("block overlap val&test  :", len(va_b & te_b))
    assert not (tr_b & va_b) and not (tr_b & te_b) and not (va_b & te_b), "block overlap!"

    patch_m = PATCH_PX * 156543.03 * math.cos(math.radians(-28.0)) / (2 ** zoom)
    print(f"approx patch size: {patch_m:.1f} m | block: {block_patches_eff*patch_m/1000:.2f} km")


    def centers(name):
        return np.array([[(p + 0.5) * patch_m, (q + 0.5) * patch_m] for p, q, _ in splits[name]])


    for other in ("val", "test"):
        if len(splits["train"]) and len(splits[other]):
            a, b = centers("train"), centers(other)
            sub = b[:: max(1, len(b) // 500)]  # subsample to bound memory
            dd = np.sqrt(((a[:, None, :] - sub[None, :, :]) ** 2).sum(-1)).min()
            print(f"min train<->{other} patch distance: {dd/1000:.2f} km "
                  f"(expect >= ~{(buffer_blocks + 1) * block_patches_eff * patch_m / 1000:.1f} km)")

    _summary = dict(split_counts={k: len(v) for k, v in splits.items()},
                    block_overlap={"train&val": len(tr_b & va_b),
                                   "train&test": len(tr_b & te_b),
                                   "val&test": len(va_b & te_b)},
                    buildings_indexed=int(cnt_all.sum()),
                    split_meta=SPLIT_META, splits_dir=SPLITS_DIR)
    log(f"[MLOPS] split -> {_summary['split_counts']}")
    return _summary
