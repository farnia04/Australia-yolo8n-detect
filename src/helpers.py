# ========================================================================

# src/helpers.py - CELL 3 (imports, tile math, tile downloader, labels)

# ========================================================================

from src.config import *


# ============================================================================
# CELL 3 - imports, tile math and helpers
# ============================================================================
import io
import json
import math
import time
import hashlib
import threading
import collections
import concurrent.futures as cf

import numpy as np
import requests
import fiona
try:  # fiona >= 1.9 needs Geometry objects; older versions accept plain dicts
    from fiona.model import Geometry as _FionaGeometry
except Exception:
    _FionaGeometry = None
from shapely.geometry import shape, box, mapping
from PIL import Image
from tqdm.auto import tqdm

# Imagery tile providers (XYZ / Web Mercator). NOTE: ESRI is served through
# CloudFront and blocks some countries (e.g. Iran) -> use a Colab/US IP for the
# default, or switch provider. All providers share the same XYZ scheme, so the
# geometry math is identical.
IMAGERY_PROVIDERS = {
    "esri": {
        "url": ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Imagery/MapServer/tile/{z}/{y}/{x}"),
        "referer": "https://www.arcgis.com/",
    },
    "eox_s2cloudless": {  # free Sentinel-2 cloudless (10 m) fallback
        "url": ("https://tiles.maps.eox.at/wmts/1.0.0/"
                "s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg"),
        "referer": "https://s2maps.eu/",
    },
}
_provider = IMAGERY_PROVIDERS[imagery_provider]
TILE_URL = _provider["url"]
TILE_REFERER = _provider.get("referer", "")

# --- Web Mercator / XYZ tile math -------------------------------------------
def lonlat_to_tile_f(lon, lat, z):
    """Fractional XYZ tile coordinates of a lon/lat point."""
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def tile_to_lonlat(x, y, z):
    """lon/lat of the top-left corner of tile (x, y)."""
    n = 2.0 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lon, lat


def lonlat_to_pixel(lon, lat, z, ox, oy):
    """Pixel coordinates inside a patch whose origin tile is (ox, oy)."""
    tx, ty = lonlat_to_tile_f(lon, lat, z)
    return (tx - ox) * 256.0, (ty - oy) * 256.0


def patch_bbox_lonlat(px, py):
    """lon/lat bounding box of a patch (patch = patch_tiles x patch_tiles tiles)."""
    ox, oy = px * patch_tiles, py * patch_tiles
    lon0, lat0 = tile_to_lonlat(ox, oy, zoom)                    # top-left
    lon1, lat1 = tile_to_lonlat(ox + patch_tiles, oy + patch_tiles, zoom)  # bottom-right
    return (lon0, lat1, lon1, lat0)                              # xmin, ymin, xmax, ymax


# --- tile downloader with on-disk cache + thread-local sessions --------------
_tls = threading.local()


def _session():
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36"),
            "Accept": "image/avif,image/webp,image/jpeg,image/png,*/*",
        }
        if TILE_REFERER:
            headers["Referer"] = TILE_REFERER
        s.headers.update(headers)
        _tls.s = s
    return s


def fetch_tile(z, x, y, retries=4):
    fp = os.path.join(CACHE_DIR, f"{z}_{x}_{y}.jpg")
    if os.path.exists(fp) and os.path.getsize(fp) > 0:
        return fp
    url = TILE_URL.format(z=z, y=y, x=x)
    for attempt in range(retries):
        try:
            r = _session().get(url, timeout=30)
            if r.status_code == 200 and r.content and len(r.content) > 100:
                with open(fp, "wb") as f:
                    f.write(r.content)
                return fp
        except Exception:
            pass
        time.sleep(0.5 * (attempt + 1))  # back off on failures / rate limits
    return None


def stitch_patch(px, py):
    """Assemble the patch image from cached tiles; return PIL image or None."""
    ox, oy = px * patch_tiles, py * patch_tiles
    canvas = Image.new("RGB", (PATCH_PX, PATCH_PX), (0, 0, 0))
    ok = 0
    for dy in range(patch_tiles):
        for dx in range(patch_tiles):
            fp = os.path.join(CACHE_DIR, f"{zoom}_{ox + dx}_{oy + dy}.jpg")
            if not os.path.exists(fp):
                continue
            canvas.paste(Image.open(fp).convert("RGB"), (dx * 256, dy * 256))
            ok += 1
    if ok < patch_tiles * patch_tiles:
        return None
    return canvas


# --- geometry -> YOLO segmentation polygon labels ---------------------------
def geom_to_yolo_lines(geom, bbox, ox, oy):
    """Clip a building geometry to the patch bbox and return YOLO seg lines."""
    clipped = geom.intersection(box(*bbox))
    if clipped.is_empty:
        return []
    parts = [clipped] if clipped.geom_type == "Polygon" else list(getattr(clipped, "geoms", []))
    lines = []
    for part in parts:
        if part.geom_type != "Polygon":
            continue
        part = part.simplify(2e-6, preserve_topology=True)  # ~0.2 m at these latitudes
        if part.is_empty or part.geom_type != "Polygon":
            continue
        ring = list(part.exterior.coords)
        pts, seen = [], set()
        for lon, lat in ring:
            X, Y = lonlat_to_pixel(lon, lat, zoom, ox, oy)
            X = min(max(X, 0.0), PATCH_PX)
            Y = min(max(Y, 0.0), PATCH_PX)
            key = (round(X, 1), round(Y, 1))
            if key in seen:
                continue
            seen.add(key)
            pts.append((X / PATCH_PX, Y / PATCH_PX))
        if len(pts) < 3:
            continue
        # shoelace area in pixels
        area = 0.0
        for i in range(len(pts)):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % len(pts)]
            area += x0 * y1 - x1 * y0
        area = abs(area) * 0.5 * (PATCH_PX ** 2)
        if area < min_poly_area_px:
            continue
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
        lines.append("0 " + coords)
    return lines


# --- Morton order + spatial split helpers -----------------------------------
def _part1by1(n):
    n &= 0x0000FFFF
    n = (n | (n << 8)) & 0x00FF00FF
    n = (n | (n << 4)) & 0x0F0F0F0F
    n = (n | (n << 2)) & 0x33333333
    n = (n | (n << 1)) & 0x55555555
    return n


def morton(x, y):
    return (_part1by1(x) << 1) | _part1by1(y)


def stratified_sample(px, py, counts, n, rng):
    """Pick ~n patches spread over region_patches-sized cells, weighted by count."""
    if n <= 0 or len(px) == 0:
        return np.zeros(0, dtype=np.int64)
    region = (px // region_patches).astype(np.int64) * 100000 + (py // region_patches)
    order = np.argsort(region, kind="stable")
    region_s, px_s, py_s, cnt_s = region[order], px[order], py[order], counts[order]
    bounds = np.searchsorted(region_s, np.unique(region_s), side="left").tolist()
    bounds.append(len(region_s))
    n_regions = len(bounds) - 1
    per_region = max(1, int(math.ceil(n / max(1, n_regions))))
    picked = []
    for i in range(n_regions):
        lo, hi = bounds[i], bounds[i + 1]
        k = min(per_region, hi - lo)
        local = np.arange(lo, hi)
        w = cnt_s[lo:hi].astype(np.float64)
        w = w / w.sum() if w.sum() > 0 else None
        sel = rng.choice(local, size=k, replace=False, p=w)
        picked.append(sel)
    sel = np.concatenate(picked) if picked else np.zeros(0, dtype=np.int64)
    if len(sel) > n:
        sel = rng.choice(sel, size=n, replace=False)
    return order[sel]


print("helpers ready")
