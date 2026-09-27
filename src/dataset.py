"""YOLO dataset build stage: imagery download + rasterised labels."""

from src.config import *

from src.helpers import *

from src.data import load_split_files


# ---- CELL 9 hoisted: balanced patch ordering (also used by CELL 11b/19) ----

def interleave_patches(splits, weights=(8, 1, 1)):
    """Order patches as train,val,test repeated proportionally, so a PARTIAL run
    still contains a balanced train/val/test set (val/test are not left to last)."""
    names = ["train", "val", "test"]
    pools = {n: list(splits[n]) for n in names}
    pos = {n: 0 for n in names}
    pattern = [n for n, w in zip(names, weights) for _ in range(max(1, int(w)))]
    out, remaining = [], sum(len(p) for p in pools.values())
    while remaining > 0:
        moved = False
        for n in pattern:
            if pos[n] < len(pools[n]):
                p, q, _c = pools[n][pos[n]]
                pos[n] += 1
                out.append((n, p, q))
                remaining -= 1
                moved = True
        if not moved:
            break
    return out



def stage_build_dataset(run):

    """Stage 5 - imagery preflight (CELL 4b), YOLO dataset (CELL 9), dataset.yaml (CELL 10)."""

    splits = load_split_files()
    assert splits, (
        f"no split files under {SPLITS_DIR} -> run the prepare_data stage first")

    # ---- CELL 4b: imagery provider preflight -------------------------------

    _probe_tx, _probe_ty = lonlat_to_tile_f(151.2153, -33.8568, zoom)  # Sydney Opera House
    _ox, _oy = int(_probe_tx), int(_probe_ty)
    _probe = _session().get(TILE_URL.format(z=zoom, y=_oy, x=_ox), timeout=30)
    print(f"provider={imagery_provider} status={_probe.status_code} bytes={len(_probe.content)}")
    if _probe.status_code != 200 or not _probe.content.startswith(b"\xff\xd8"):
        print("!! imagery provider did NOT return a JPEG.")
        print("   ESRI is blocked in some countries (CloudFront). On Colab it works.")
        print("   Otherwise set imagery_provider='eox_s2cloudless' (lower resolution).")

    # ---- CELL 9: download imagery + build the YOLO dataset ----------------

    for name in ("train", "val", "test"):
        os.makedirs(os.path.join(YOLO_DIR, "images", name), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DIR, "labels", name), exist_ok=True)

    all_patches = interleave_patches(
        splits, (round(train_frac * 10), round(val_frac * 10), round(test_frac * 10)))
    if build_limit is not None:
        all_patches = all_patches[:build_limit]
        log(f"[CHECK] build_limit={build_limit:,} -> balanced subset "
            f"(full split has {sum(len(v) for v in splits.values()):,} patches)")
    n_tr = sum(1 for n, _, _ in all_patches if n == "train")
    n_va = sum(1 for n, _, _ in all_patches if n == "val")
    n_te = sum(1 for n, _, _ in all_patches if n == "test")
    log(f"[CHECK] patches to build: {len(all_patches):,} (train/val/test = {n_tr}/{n_va}/{n_te})")

    manifest = load_manifest() if resume else {}
    log(f"[CHECK] manifest rows loaded: {len(manifest):,}")


    def valid_image(path):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return False
        try:
            with Image.open(path) as im:
                im.verify()
            return True
        except Exception:
            return False


    def valid_label(path):
        return os.path.exists(path) and os.path.getsize(path) > 0


    def patch_tile_list(p, q):
        ox, oy = p * patch_tiles, q * patch_tiles
        return [(zoom, ox + dx, oy + dy)
                for dy in range(patch_tiles) for dx in range(patch_tiles)]


    def drop_tiles(tiles):
        if keep_tile_cache:
            return
        for z, tx, ty in tiles:
            fp = os.path.join(CACHE_DIR, f"{z}_{tx}_{ty}.jpg")
            try:
                if os.path.exists(fp):
                    os.remove(fp)
            except OSError:
                pass


    stats = collections.Counter()
    set_stage("patch_generation", "in_progress")

    with cf.ThreadPoolExecutor(max_workers=download_workers) as ex, fiona.open(FGB_PATH) as src:
        for start in range(0, len(all_patches), patch_batch):
            patch_chunk = all_patches[start:start + patch_batch]
            todo = []
            for name, p, q in patch_chunk:
                pid = f"{p}_{q}"
                img_path = os.path.join(YOLO_DIR, "images", name, pid + ".jpg")
                lbl_path = os.path.join(YOLO_DIR, "labels", name, pid + ".txt")
                rec = manifest.get(pid)
                # (a) already complete AND valid -> SKIP (no network at all)
                if (resume and rec and rec.get("status") == "complete"
                        and rec.get("split") == name
                        and valid_image(img_path) and valid_label(lbl_path)):
                    stats["already_existing"] += 1
                    continue
                # (b) deterministically empty patch -> SKIP forever (never retry)
                if (resume and rec and rec.get("status") == "invalid"
                        and rec.get("error") in ("blank imagery", "no polygons")
                        and rec.get("split") == name):
                    stats["skipped"] += 1
                    continue
                todo.append((name, p, q, pid, img_path, lbl_path))

            # parallel tile fetch for the whole batch (only missing tiles hit network)
            tiles = [t for (_n, p, q, *_rest) in todo for t in patch_tile_list(p, q)]
            if tiles:
                list(ex.map(lambda t: fetch_tile(*t), tiles))

            # stitch + rasterize sequentially (fiona handle is single-threaded)
            new_records = []
            for name, p, q, pid, img_path, lbl_path in todo:
                patch_tiles_list = patch_tile_list(p, q)
                try:
                    img = stitch_patch(p, q)
                    if img is None:
                        raise RuntimeError("missing tiles")
                    if np.asarray(img, dtype=np.float32).std() < blank_std_threshold:
                        rec = dict(patch_id=pid, split=name, source=imagery_provider,
                                   status="invalid", error="blank imagery")
                        stats["blank"] += 1
                    else:
                        ox, oy = p * patch_tiles, q * patch_tiles
                        bbox = patch_bbox_lonlat(p, q)
                        lines = []
                        for feat in src.filter(bbox=bbox):
                            try:
                                lines.extend(geom_to_yolo_lines(shape(feat["geometry"]),
                                                                bbox, ox, oy))
                            except Exception:
                                continue
                        if not lines:
                            rec = dict(patch_id=pid, split=name, source=imagery_provider,
                                       status="invalid", error="no polygons")
                            stats["no_polygons"] += 1
                        else:
                            img.save(img_path, quality=jpeg_quality)
                            tmp_lbl = lbl_path + ".tmp"
                            with open(tmp_lbl, "w", encoding="utf-8") as f:
                                f.write("\n".join(lines))
                            os.replace(tmp_lbl, lbl_path)      # label published atomically
                            rec = dict(patch_id=pid, split=name, source=imagery_provider,
                                       status="complete", image_path=img_path,
                                       label_path=lbl_path,
                                       file_size=os.path.getsize(img_path),
                                       n_polygons=len(lines), validated=True, error="")
                            stats["downloaded"] += 1
                except Exception as e:
                    rec = dict(patch_id=pid, split=name, source=imagery_provider,
                               status="failed", error=str(e)[:200])
                    stats["failed"] += 1
                    log(f"[FAIL] patch {pid}: {e}")
                finally:
                    drop_tiles(patch_tiles_list)
                manifest[pid] = rec
                new_records.append(rec)

            append_manifest(new_records)          # incremental flush every batch
            done = start + len(patch_chunk)
            log(f"[DOWNLOAD] {done:,}/{len(all_patches):,} processed "
                f"(downloaded={stats['downloaded']:,} skipped={stats['already_existing']:,} "
                f"failed={stats['failed']:,})")

    # rebuild the {train,val,test} -> count map used by later cells
    built = {n: 0 for n in ("train", "val", "test")}
    for r in manifest.values():
        if r.get("status") == "complete" and r.get("split") in built:
            built[r["split"]] += 1
    skipped = collections.Counter({k: stats[k] for k in
                                   ("blank", "no_polygons", "skipped", "failed")})

    complete = sum(1 for r in manifest.values() if r.get("status") == "complete")
    log(f"[SUMMARY] total={len(all_patches):,} successful={complete:,} "
        f"failed={stats['failed']:,} skipped={stats['blank']+stats['no_polygons']+stats['skipped']:,} "
        f"downloaded={stats['downloaded']:,} already_existing={stats['already_existing']:,}")
    set_stage("patch_generation", "complete" if stats["failed"] == 0 else "in_progress",
              failed=stats["failed"], complete=complete)
    set_stage("label_generation", "complete" if stats["failed"] == 0 else "in_progress",
              failed=stats["failed"])

    # ---- CELL 10: write dataset.yaml --------------------------------------

    with open(YAML_PATH, "w") as f:
        f.write(f"path: {YOLO_DIR}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write("names:\n  0: building\n")
    print(open(YAML_PATH).read())

    log(f"[MLOPS] dataset -> {YOLO_DIR} (train/val/test = "
        f"{built['train']}/{built['val']}/{built['test']})")
    return dict(built=built, skipped=dict(skipped),
                patches_total=len(all_patches), yaml_path=YAML_PATH,
                yolo_dir=YOLO_DIR)
