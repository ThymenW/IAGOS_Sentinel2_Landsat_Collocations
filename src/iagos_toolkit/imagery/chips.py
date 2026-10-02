"""Cut small Sentinel-2 image chips around collocated aircraft (and random negatives).

Each chip is a ``2 * half_size`` square on the 10 m grid holding B02, B03, B04 and B10
(cirrus, 60 m, nearest-neighbour upsampled to 10 m). The jp2 files are downloaded
through :class:`pycontrails.datalib.sentinel.Sentinel`, only the chip windows are decoded,
and the files are removed from disk again (~400 full tiles would take ~130 GB).
"""

from __future__ import annotations

import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.transform
from pycontrails.core import cache
from pycontrails.datalib import sentinel
from pycontrails.datalib.leo_utils.sentinel_metadata import read_image_coordinates
from rasterio.windows import Window
from scipy import ndimage as ndi

logger = logging.getLogger(__name__)

VISIBLE_BANDS = ("B02", "B03", "B04")
CIRRUS_BAND = "B10"
RESOLUTION = 10.0  # m, grid of the stored chips

# Chips dropped after visual inspection (name -> reason). Skipped by :func:`load_chips`.
EXCLUDED_CHIPS = {
    "neg_0467_2": "random negative that contains a real aircraft",
    "neg_0540_1": "random negative that contains a real aircraft",
}
DETECTOR_MARGIN = 500.0  # m, extra window around a chip used for the detector-border distance


def load_sentinel_collocations(csv_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return usable Sentinel rows and the excluded rows (with an ``exclude_reason``)."""
    df = pd.read_csv(csv_path, comment="#")
    df = df[df["satellite"] == "Sentinel"].copy()
    df["row"] = df.index

    reason = pd.Series("", index=df.index)
    reason[df["out_of_frame_flag"] == True] = "out_of_frame_flag"  # noqa: E712
    for i, r in df.iterrows():
        xmin, xmax, ymin, ymax = json.loads(r["extent"])
        if not (np.isfinite(r["x"]) and np.isfinite(r["y"])):
            reason[i] = reason[i] or "x/y not finite"
        elif not (xmin <= r["x"] <= xmax and ymin <= r["y"] <= ymax):
            reason[i] = reason[i] or "x/y outside extent"

    df["exclude_reason"] = reason
    return df[reason == ""].drop(columns="exclude_reason"), df[reason != ""]


def _pixel_axes(granule_meta: str, band: str) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre x/y axes. pycontrails gives upper-left corners, so shift by half a pixel."""
    x, y = read_image_coordinates(granule_meta, band)
    res = x[1] - x[0]
    return x + res / 2, y - res / 2


class TileChipper:
    """Cut chips from one Sentinel-2 tile.

    The jp2 files are downloaded with pycontrails, but only the chip windows are decoded
    (rasterio/OpenJPEG). Decoding full 10980 x 10980 tiles takes ~1 min per granule.
    """

    def __init__(self, base_url: str, granule_id: str, cachestore: cache.CacheStore | None = None):
        handler = sentinel.Sentinel(
            base_url, granule_id, bands=VISIBLE_BANDS, cachestore=cachestore
        )
        granule_meta, safe_meta = handler._get_meta()
        self.src, self.scale, self.axes = {}, {}, {}
        for b in (*VISIBLE_BANDS, CIRRUS_BAND):
            self.src[b] = rasterio.open(handler._get_jp2(b))
            self.scale[b] = sentinel._read_band_reflectance_rescaling(safe_meta, b)
            self.axes[b] = _pixel_axes(granule_meta, b)
        self.xc, self.yc = self.axes[VISIBLE_BANDS[0]]
        self.crs = str(handler.get_crs())

    def close(self) -> None:
        for src in self.src.values():
            src.close()

    def _read(self, band: str, i0: int, i1: int, j0: int, j1: int) -> np.ndarray:
        dn = self.src[band].read(1, window=Window(j0, i0, j1 - j0, i1 - i0)).astype("float32")
        gain, offset = self.scale[band]
        return np.where(dn == 0, np.nan, (dn + offset) / gain).astype("float32")

    def chip(self, x: float, y: float, half_size: float = 1000.0) -> dict | None:
        """Return a chip whose centre pixel contains (x, y); None if it leaves the tile or hits nodata."""
        n = int(round(2 * half_size / RESOLUTION))
        xc, yc = self.xc, self.yc
        i0 = int(np.floor((yc[0] + RESOLUTION / 2 - y) / RESOLUTION)) - n // 2
        j0 = int(np.floor((x - (xc[0] - RESOLUTION / 2)) / RESOLUTION)) - n // 2
        if i0 < 0 or j0 < 0 or i0 + n > len(yc) or j0 + n > len(xc):
            return None
        out = {b: self._read(b, i0, i0 + n, j0, j0 + n) for b in VISIBLE_BANDS}
        xs, ys = xc[j0 : j0 + n], yc[i0 : i0 + n]

        # nearest-neighbour cirrus value for every 10 m pixel centre
        cxc, cyc = self.axes[CIRRUS_BAND]
        cres = cxc[1] - cxc[0]
        jj = np.floor((xs - (cxc[0] - cres / 2)) / cres).astype(int)
        ii = np.floor((cyc[0] + cres / 2 - ys) / cres).astype(int)
        if jj.min() < 0 or ii.min() < 0 or jj.max() >= len(cxc) or ii.max() >= len(cyc):
            return None
        cir = self._read(CIRRUS_BAND, ii.min(), ii.max() + 1, jj.min(), jj.max() + 1)
        out[CIRRUS_BAND] = cir[np.ix_(ii - ii.min(), jj - jj.min())]

        if any(np.isnan(v).any() for v in out.values()):
            return None
        out.update(x=xs, y=ys)
        return out

    def sample_negatives(
        self,
        n: int,
        avoid_xy: np.ndarray,
        rng: np.random.Generator,
        min_dist: float = 5000.0,
        half_size: float = 1000.0,
        max_tries: int = 500,
    ) -> list[tuple[float, float, dict]]:
        """Random non-overlapping chips at least ``min_dist`` from every point in ``avoid_xy``."""
        out = []
        lo_x, hi_x = self.xc[0] + half_size, self.xc[-1] - half_size
        lo_y, hi_y = self.yc[-1] + half_size, self.yc[0] - half_size
        for _ in range(max_tries):
            if len(out) == n:
                break
            x, y = rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y)
            if np.hypot(avoid_xy[:, 0] - x, avoid_xy[:, 1] - y).min() < min_dist:
                continue
            if any(np.hypot(x - ox, y - oy) < 2 * half_size for ox, oy, _ in out):
                continue
            c = self.chip(x, y, half_size)
            if c is not None:
                out.append((x, y, c))
        return out


class DetectorMask:
    """Which Sentinel-2 detector imaged each 10 m pixel of a tile (B03 detector footprints).

    Uses the ``MSK_DETFOO_B03`` file in the granule's QI_DATA folder: a jp2 raster for
    processing baseline >= 04.00, GML polygons before. GML footprints of neighbouring detectors
    overlap; ``overlap`` then counts how many footprints cover each pixel (None for jp2).
    The GML polygons are rasterised onto the tile's own 10 m grid (pycontrails'
    ``parse_high_res_detector_mask`` places them at the top-left corner of the tile instead).
    """

    def __init__(self, base_url: str, granule_id: str, cachestore: cache.CacheStore | None = None):
        handler = sentinel.Sentinel(base_url, granule_id, bands=["B03"], cachestore=cachestore)
        granule_meta, _ = handler._get_meta()
        self.xc, self.yc = _pixel_axes(granule_meta, "B03")
        files = handler.fs.ls(f"{base_url}/GRANULE/{granule_id}/QI_DATA/")
        masks = [f for f in files if "MSK_DETFOO_B03" in f and f.endswith((".jp2", ".gml"))]
        if not masks:
            raise FileNotFoundError(f"no B03 detector mask for {granule_id}")
        url = "gs://" + sorted(masks, key=lambda f: not f.endswith(".jp2"))[0].removeprefix("gs://")
        sink = handler.cachestore.path(url.removeprefix(sentinel.GCP_STRIP_PREFIX))
        if not handler.cachestore.exists(sink):
            handler.fs.get(url, sink)
        self.format = Path(sink).suffix[1:]
        shape = (len(self.yc), len(self.xc))
        if self.format == "jp2":
            with rasterio.open(sink) as src:
                self.detector = src.read(1)
            self.overlap = None
        else:
            import geopandas as gpd

            gdf = gpd.read_file(sink)
            gdf["det"] = gdf["gml_id"].str.extract(r"-B\d+-(\d+)-")[0].astype(int)
            transform = rasterio.transform.from_origin(
                self.xc[0] - RESOLUTION / 2, self.yc[0] + RESOLUTION / 2, RESOLUTION, RESOLUTION
            )
            self.detector = np.zeros(shape, np.uint8)
            self.overlap = np.zeros(shape, np.uint8)
            for det, g in gdf.groupby("det"):
                m = rasterio.features.rasterize(
                    [(geom, 1) for geom in g.geometry], out_shape=shape, transform=transform,
                    fill=0, dtype="uint8",
                )
                self.overlap += m
                self.detector[m > 0] = det

    def window(self, xs: np.ndarray, ys: np.ndarray, margin: float = DETECTOR_MARGIN) -> dict:
        """Detector id, overlap count and distance (m) to the nearest detector border on a chip grid.

        The border distance is computed on a window ``margin`` larger than the chip, so borders
        just outside the chip still count; it saturates at ``margin``.
        """
        m = int(round(margin / RESOLUTION))
        j0 = int(round((xs[0] - self.xc[0]) / RESOLUTION))
        i0 = int(round((self.yc[0] - ys[0]) / RESOLUTION))
        n, k = len(ys), len(xs)
        pad = np.pad(self.detector, m, mode="edge")
        det = pad[i0 : i0 + n + 2 * m, j0 : j0 + k + 2 * m]
        # a border pixel has a 4-neighbour from another (valid) detector
        border = np.zeros(det.shape, bool)
        for ax in (0, 1):
            diff = np.diff(det.astype(int), axis=ax) != 0
            valid = (np.delete(det, -1, axis=ax) > 0) & (np.delete(det, 0, axis=ax) > 0)
            d = diff & valid
            if ax == 0:
                border[:-1] |= d; border[1:] |= d
            else:
                border[:, :-1] |= d; border[:, 1:] |= d
        dist = (ndi.distance_transform_edt(~border) * RESOLUTION if border.any()
                else np.full(det.shape, np.inf))
        out = {"detector": det[m:-m, m:-m],
               "det_border_m": np.minimum(dist[m:-m, m:-m], margin).astype("float32")}
        if self.overlap is not None:
            out["det_overlap"] = np.pad(self.overlap, m, mode="edge")[i0 + m : i0 + m + n, j0 + m : j0 + m + k]
        return out


def add_detector_info(
    df: pd.DataFrame, chip_dir: str | Path, cache_dir: str | Path, max_workers: int = 6
) -> pd.DataFrame:
    """Add detector id / border distance (/ overlap) arrays to every cached chip of ``df``'s granules.

    Only the small B03 detector-mask file and the tile metadata are downloaded. Chips that
    already contain ``det_border_m`` are skipped. Returns a log DataFrame.
    """
    chip_dir, cache_dir = Path(chip_dir), Path(cache_dir)

    def one(rows: pd.DataFrame) -> dict:
        r0 = rows.iloc[0]
        paths = [chip_dir / f"pos_{r:04d}.npz" for r in rows["row"]]
        paths += sorted(chip_dir.glob(f"neg_{r0['row']:04d}_*.npz"))
        paths = [p for p in paths if p.exists()]
        todo = []
        for p in paths:
            with np.load(p) as z:
                if "det_border_m" not in z.files:
                    todo.append(p)
        if not todo:
            return {"granule": r0["scene_id"], "status": "cached"}
        store = cache_dir / r0["scene_id"]
        try:
            mask = DetectorMask(r0["base_url"], r0["scene_id"], cache.DiskCacheStore(cache_dir=store, allow_clear=True))
            for p in todo:
                with np.load(p) as z:
                    d = {k: z[k] for k in z.files}
                d.update(mask.window(d["x"], d["y"]))
                np.savez_compressed(p, **d)
            return {"granule": r0["scene_id"], "status": f"ok ({mask.format})"}
        except Exception as e:  # noqa: BLE001
            return {"granule": r0["scene_id"], "status": f"error: {type(e).__name__}: {e}"}
        finally:
            shutil.rmtree(store, ignore_errors=True)

    groups = [g for _, g in df.groupby("scene_id")]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        log = list(ex.map(one, groups))
    return pd.DataFrame(log)


def near_detector_border(chip: dict, max_dist: float = 200.0, x: float | None = None, y: float | None = None) -> bool:
    """True if the chip (or the point x, y in it) lies within ``max_dist`` m of a detector border.

    Without ``x``/``y`` the whole chip counts: any pixel closer than ``max_dist``. Chips without
    detector information (see :func:`add_detector_info`) return False.
    """
    if "det_border_m" not in chip:
        return False
    dist = chip["det_border_m"]
    if x is None or y is None:
        return bool(dist.min() < max_dist)
    i = int(np.argmin(np.abs(chip["y"] - y)))
    j = int(np.argmin(np.abs(chip["x"] - x)))
    return bool(dist[i, j] < max_dist)


def drop_near_detector_border(chips: list[dict], max_dist: float = 200.0) -> tuple[list[dict], list[str]]:
    """Remove chips that contain (or lie within ``max_dist`` m of) a Sentinel-2 detector border.

    Near detector borders the data are less reliable: the same aircraft can appear twice, and
    small co-registration offsets between detectors create colour fringes. The test uses only
    the chip itself (:func:`near_detector_border` over all its pixels), never the label.
    Returns the kept chips and the names of the dropped ones.
    """
    kept, dropped = [], []
    for c in chips:
        (dropped.append(c["name"]) if near_detector_border(c, max_dist) else kept.append(c))
    return kept, dropped


def _save(path: Path, chip: dict, dtype: str = "float32", **meta) -> None:
    np.savez_compressed(
        path, **{k: np.asarray(v, dtype=dtype) for k, v in chip.items() if k.startswith("B")},
        x=chip["x"], y=chip["y"], **meta,
    )


def _process_granule(
    rows: pd.DataFrame, out_dir: Path, cache_dir: Path, n_neg: int, half_size: float, seed: int,
    dtype: str = "float32",
) -> list[dict]:
    """Download one granule, write its positive and negative chips, delete the tile."""
    r0 = rows.iloc[0]
    granule = r0["scene_id"]
    log = []
    store_dir = cache_dir / granule
    tile = None
    try:
        tile = TileChipper(r0["base_url"], granule, cache.DiskCacheStore(cache_dir=store_dir, allow_clear=True))
        avoid = rows[["x", "y"]].to_numpy(float)
        for _, r in rows.iterrows():
            c = tile.chip(r["x"], r["y"], half_size)
            if c is None:
                log.append({"row": r["row"], "granule": granule, "status": "chip crosses tile edge/nodata"})
                continue
            _save(out_dir / f"pos_{r['row']:04d}.npz", c, dtype, label=1, row=r["row"],
                  truth_x=r["x"], truth_y=r["y"], granule=granule, crs=tile.crs)
            log.append({"row": r["row"], "granule": granule, "status": "ok"})
        # seed per granule so a rerun of a single granule gives the same negatives
        rng = np.random.default_rng([seed, int(rows["row"].iloc[0])])
        negs = tile.sample_negatives(n_neg * len(rows), avoid, rng, half_size=half_size)
        for k, (x, y, c) in enumerate(negs):
            _save(out_dir / f"neg_{r0['row']:04d}_{k}.npz", c, dtype, label=0, row=r0["row"],
                  truth_x=np.nan, truth_y=np.nan, centre_x=x, centre_y=y, granule=granule, crs=tile.crs)
        if len(negs) < n_neg * len(rows):
            log.append({"row": r0["row"], "granule": granule,
                        "status": f"only {len(negs)} negatives found"})
        (out_dir / f".done_{r0['row']:04d}").touch()
    except Exception as e:  # noqa: BLE001 -- keep going, report at the end
        log.append({"row": r0["row"], "granule": granule, "status": f"error: {type(e).__name__}: {e}"})
    finally:
        if tile is not None:
            tile.close()
        shutil.rmtree(store_dir, ignore_errors=True)
    return log


def build_chip_cache(
    df: pd.DataFrame,
    out_dir: str | Path,
    cache_dir: str | Path,
    n_neg: int = 3,
    half_size: float = 1000.0,
    seed: int = 0,
    max_workers: int = 3,
    dtype: str = "float32",
) -> pd.DataFrame:
    """Write ``pos_<row>.npz`` and ``neg_<row>_<k>.npz`` chips for every granule in ``df``.

    Granules that were processed before (marker ``.done_<row>`` or existing negatives) are
    skipped. ``dtype`` sets the stored band precision (float16 halves the size of big chips).
    Returns a log DataFrame.
    """
    out_dir, cache_dir = Path(out_dir), Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    todo = [
        rows for _, rows in df.groupby("scene_id")
        # negatives are named after the granule's first row, so they mark a processed granule
        # even when its positive chip could not be cut (tile edge / nodata)
        if not (out_dir / f".done_{rows['row'].iloc[0]:04d}").exists()
        and not any(out_dir.glob(f"neg_{rows['row'].iloc[0]:04d}_*.npz"))
    ]
    logger.info("%d granules to process", len(todo))
    log = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_process_granule, rows, out_dir, cache_dir, n_neg, half_size, seed, dtype)
                for rows in todo]
        for k, f in enumerate(as_completed(futs), 1):
            res = f.result()
            log.extend(res)
            logger.info("[%d/%d] %s: %s", k, len(todo), res[0]["granule"], res[-1]["status"])
    return pd.DataFrame(log)


def load_chips(chip_dir: str | Path, exclude: dict[str, str] | None = None) -> list[dict]:
    """Load every cached chip as a dict of arrays/scalars, skipping ``exclude`` (default
    :data:`EXCLUDED_CHIPS`)."""
    exclude = EXCLUDED_CHIPS if exclude is None else exclude
    chips = []
    for p in sorted(Path(chip_dir).glob("*.npz")):
        if p.stem in exclude:
            continue
        with np.load(p, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        for k in ("label", "row", "truth_x", "truth_y", "granule", "crs"):
            d[k] = d[k].item()
        d["name"] = p.stem
        chips.append(d)
    return chips
