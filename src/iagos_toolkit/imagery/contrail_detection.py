"""Did the aircraft form a contrail? Line filters on Sentinel-2 chips around a known aircraft.

Chips (see :mod:`iagos_toolkit.imagery.chips`) are 6 x 6 km windows on the 10 m grid centred on
the manual aircraft label. Every method uses two priors:

* the aircraft position (for now the manual label ``truth_x``/``truth_y``),
* the aircraft heading. Plume elements emitted at different times all drift with the same wind,
  so a contrail line points along the *air-relative* direction of flight, i.e. the heading.
  It extends behind the aircraft, at ``heading + 180``; wind errors give a few degrees scatter.

Image convention: arrays are (row, col) with row 0 at the north edge; a direction is a unit
vector ``(drow, dcol)``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyproj
import rasterio.features
import rasterio.transform
from scipy import ndimage as ndi
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

from iagos_toolkit.imagery import aircraft_detection as ad

Chip = dict[str, np.ndarray]
RES = 10.0  # m per pixel

CSV_FIELDS = ("heading", "longitude", "latitude", "altitude", "contrail_formation", "contrail_label",
              "eastward_wind_iagos", "northward_wind_iagos", "ground_speed_AC", "sensing_time")


# --- metadata --------------------------------------------------------------------------------
def attach_metadata(chips: list[Chip], csv: pd.DataFrame) -> list[Chip]:
    """Copy the collocation CSV fields of each chip's row into the chip dict (in place)."""
    for c in chips:
        row = csv.loc[c["row"]]
        for f in CSV_FIELDS:
            c[f] = row[f]
        c["formation"] = bool(row["contrail_formation"])
    return chips


def _parse_geojson(label) -> dict | None:
    if not isinstance(label, str) or not label.strip():
        return None
    g = json.loads(label)
    if isinstance(g, str):  # some labels are double-encoded
        g = json.loads(g)
    return g


# --- geometry --------------------------------------------------------------------------------
def grid_convergence(chip: Chip) -> float:
    """Meridian convergence (deg) at the aircraft: true north minus grid north, clockwise positive."""
    f = pyproj.Proj(chip["crs"]).get_factors(chip["longitude"], chip["latitude"])
    return float(f.meridian_convergence)


def behind_direction(chip: Chip, heading: float | None = None, convergence_sign: float = 1.0) -> np.ndarray:
    """Unit (drow, dcol) vector pointing from the aircraft back along its heading.

    The heading is a true-north azimuth; the image grid is rotated by the meridian convergence.
    ``convergence_sign`` lets the caller flip the convention (checked against the labels).
    """
    heading = chip["heading"] if heading is None else heading
    az = np.deg2rad(heading + 180.0 - convergence_sign * grid_convergence(chip))
    return np.array([-np.cos(az), np.sin(az)])


def aircraft_pixel(chip: Chip) -> tuple[int, int]:
    """(row, col) of the manual aircraft label."""
    return (int(np.argmin(np.abs(chip["y"] - chip["truth_y"]))),
            int(np.argmin(np.abs(chip["x"] - chip["truth_x"]))))


def label_mask(chip: Chip) -> np.ndarray | None:
    """Rasterise the hand-drawn contrail polygon onto the chip grid (None if there is none)."""
    g = _parse_geojson(chip.get("contrail_label"))
    if g is None:
        return None
    tr = pyproj.Transformer.from_crs("EPSG:4326", chip["crs"], always_xy=True).transform
    geom = shp_transform(tr, shape(g))
    t = rasterio.transform.from_origin(chip["x"][0] - RES / 2, chip["y"][0] + RES / 2, RES, RES)
    shp = (len(chip["y"]), len(chip["x"]))
    return rasterio.features.rasterize([(geom, 1)], out_shape=shp, transform=t, fill=0, dtype="uint8").astype(bool)


def mask_direction(mask: np.ndarray, origin: tuple[int, int]) -> np.ndarray | None:
    """Principal direction of a mask (unit (drow, dcol)), oriented away from ``origin``."""
    ij = np.argwhere(mask).astype(float)
    if len(ij) < 5:
        return None
    centred = ij - ij.mean(0)
    v = np.linalg.svd(centred, full_matrices=False)[2][0]
    if np.dot(ij.mean(0) - np.asarray(origin, float), v) < 0:
        v = -v
    return v


# --- line filters ----------------------------------------------------------------------------
def hessian(a: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale-normalised Gaussian second derivatives (Hrr, Hrc, Hcc) of ``a``."""
    a = a.astype("float32")
    s2 = sigma**2
    return (s2 * ndi.gaussian_filter(a, sigma, order=(2, 0)),
            s2 * ndi.gaussian_filter(a, sigma, order=(1, 1)),
            s2 * ndi.gaussian_filter(a, sigma, order=(0, 2)))


def ridge_across(h: tuple[np.ndarray, ...], u: np.ndarray) -> np.ndarray:
    """Second derivative across a line with direction ``u`` (drow, dcol).

    Negative for a bright line, positive for a dark one; static background varying slower
    than the scale ``sigma`` gives ~0.
    """
    hrr, hrc, hcc = h
    nr, nc = -u[1], u[0]  # normal
    return nr * nr * hrr + 2 * nr * nc * hrc + nc * nc * hcc


def _ray_points(origin, angles_rad, dists_px):
    """Sample coordinates (2, n_angles, n_dists) of rays from ``origin``."""
    dr, dc = -np.cos(angles_rad), np.sin(angles_rad)  # azimuth convention, like behind_direction
    return np.stack([origin[0] + dr[:, None] * dists_px[None], origin[1] + dc[:, None] * dists_px[None]])


def angular_profile(
    a: np.ndarray,
    origin: tuple[int, int],
    sigma: float,
    r0: float = 200.0,
    lengths: tuple[float, ...] = (500.0, 1000.0, 2000.0, 2800.0),
    step_deg: float = 2.0,
    tolerance_px: int = 2,
) -> dict:
    """Mean oriented ridge response along rays from ``origin`` for every azimuth.

    For each azimuth (deg, clockwise from grid north) the ridge filter is steered along the ray
    and sampled from ``r0`` to each length in ``lengths`` (m). Each sample takes the strongest
    response within ``tolerance_px`` across the ray, so a slightly curved or offset contrail is
    not lost. Returns ``angles`` and ``bright``/``dark`` arrays of shape (n_angles, n_lengths).
    """
    h = hessian(a, sigma)
    angles = np.arange(0.0, 360.0, step_deg)
    rad = np.deg2rad(angles)
    d = np.arange(r0, max(lengths) + RES, RES) / RES
    bright = np.zeros((len(angles), len(lengths)), "float32")
    dark = np.zeros_like(bright)
    for k, t in enumerate(rad):
        u = np.array([-np.cos(t), np.sin(t)])
        n = np.array([-u[1], u[0]])
        r = ridge_across(h, u)
        best_b = np.full(len(d), -np.inf, "float32")
        best_d = np.full(len(d), -np.inf, "float32")
        for off in range(-tolerance_px, tolerance_px + 1):
            pr = origin[0] + u[0] * d + n[0] * off
            pc = origin[1] + u[1] * d + n[1] * off
            v = ndi.map_coordinates(r, [pr, pc], order=1, mode="constant", cval=np.nan)
            best_b = np.fmax(best_b, -v)
            best_d = np.fmax(best_d, v)
        for m, L in enumerate(lengths):
            sel = d * RES <= L
            bright[k, m] = np.nanmean(best_b[sel])
            dark[k, m] = np.nanmean(best_d[sel])
    return {"angles": angles, "lengths": np.asarray(lengths), "bright": bright, "dark": dark}


def profile_score(prof: dict, expected_az: float, search: float = 25.0, exclude: float = 40.0,
                  polarities: tuple[str, ...] = ("bright", "dark")) -> dict:
    """How much the rear direction stands out in an angular profile.

    For every length and polarity the profile is standardised with the median/MAD of the
    angles further than ``exclude`` deg from ``expected_az``; the score is the largest z within
    ``search`` deg of it. Returns score, best azimuth, length (m) and polarity.
    """
    ang = prof["angles"]
    diff = (ang - expected_az + 180) % 360 - 180
    near, far = np.abs(diff) <= search, np.abs(diff) > exclude
    best = {"score": -np.inf, "azimuth": np.nan, "length": np.nan, "polarity": ""}
    for pol in polarities:
        p = prof[pol]
        for m, L in enumerate(prof["lengths"]):
            col = p[:, m]
            ref = col[far]
            med = np.nanmedian(ref)
            mad = 1.4826 * np.nanmedian(np.abs(ref - med))
            z = (col - med) / max(mad, 1e-9)
            k = int(np.nanargmax(np.where(near, z, -np.inf)))
            if z[k] > best["score"]:
                best = {"score": float(z[k]), "azimuth": float(ang[k]), "length": float(L), "polarity": pol}
    return best


def expected_azimuth(chip: Chip) -> float:
    """Grid azimuth (deg) from the aircraft back along its heading."""
    b = behind_direction(chip)
    return float(np.degrees(np.arctan2(b[1], -b[0])) % 360)


def corridor_contrast(a: np.ndarray, origin, expected_az: float, size: int = 15, r0: float = 200.0,
                      length: float = 1000.0, half_width: float = 50.0, step_deg: float = 10.0) -> float:
    """Baseline: mean |top-hat| in a strip behind the aircraft, z-scored against other directions."""
    th = ad.tophat(a.astype("float32"), size)
    vals = []
    angles = np.arange(0, 360, step_deg)
    d = np.arange(r0, length + RES, RES) / RES
    offs = np.arange(-half_width, half_width + RES, RES) / RES
    for t in np.deg2rad(angles):
        u = np.array([-np.cos(t), np.sin(t)]); n = np.array([-u[1], u[0]])
        pr = origin[0] + u[0] * d[:, None] + n[0] * offs[None]
        pc = origin[1] + u[1] * d[:, None] + n[1] * offs[None]
        v = ndi.map_coordinates(th, [pr.ravel(), pc.ravel()], order=1, mode="constant", cval=np.nan)
        vals.append(np.nanmean(np.abs(v)))
    vals = np.asarray(vals)
    diff = (angles - expected_az + 180) % 360 - 180
    near, far = np.abs(diff) <= 25, np.abs(diff) > 40
    med = np.median(vals[far]); mad = 1.4826 * np.median(np.abs(vals[far] - med))
    return float((vals[near].max() - med) / max(mad, 1e-9))


# --- input images ----------------------------------------------------------------------------
def clahe(a: np.ndarray, clip_limit: float = 0.03, kernel_frac: float = 1 / 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation, as pycontrails builds its RGB.

    pycontrails (``leo_utils.vis``) min-max normalises a band and applies
    ``skimage.exposure.equalize_adapthist(clip_limit=0.03)`` with a window of 1/8 of the image.
    Applied to a chip, the window is local (750 m for a 6 km chip); it lifts faint structure
    inside bright cloud, where a contrail has little contrast in the raw reflectance.
    """
    from skimage import exposure

    a = a.astype("float32")
    n = (a - a.min()) / max(float(a.max() - a.min()), 1e-9)
    return exposure.equalize_adapthist(n, clip_limit=clip_limit, kernel_size=max(int(a.shape[0] * kernel_frac), 8)).astype("float32")


def image(chip: Chip, name: str) -> np.ndarray:
    """Input image for the line filters. Cached in the chip.

    ``B03`` / ``B10``: raw reflectance. ``g_minus_b``: B03 - B02. ``g_minus_maxbr``:
    B03 - max(B02, B04). ``*_clahe``: the same after :func:`clahe` of each band. Subtracting blue
    or red removes white background (cloud) that is equally bright in all bands; a contrail
    survives because band parallax shifts its blue copy sideways.
    """
    key = ("_img", name)
    if key in chip:
        return chip[key]
    f = lambda b: chip[b].astype("float32")
    if name in ("B02", "B03", "B04", "B10"):
        out = f(name)
    elif name == "B10_clahe":
        out = clahe(f("B10"))
    elif name == "g_minus_b":
        out = f("B03") - f("B02")
    elif name == "g_minus_maxbr":
        out = f("B03") - np.maximum(f("B02"), f("B04"))
    elif name == "g_minus_b_clahe":
        out = clahe(f("B03")) - clahe(f("B02"))
    else:
        raise ValueError(f"unknown image {name}")
    chip[key] = out
    return out


# --- per-chip features -----------------------------------------------------------------------
# Tuned with granule-level 2-fold CV on the 405 aircraft chips (notebooks/contrail_formation.ipynb).
CIRRUS_RAW = {"image": "B10", "sigma": 3.0, "r0": 300.0, "length": 1000.0, "polarities": ("bright", "dark")}
GREEN = {"image": "B03", "sigma": 1.0, "r0": 200.0, "length": 1000.0, "polarities": ("bright",)}
CIRRUS = {"image": "B10_clahe", "sigma": 3.0, "r0": 200.0, "length": 1000.0, "polarities": ("bright",)}
GREEN_MINUS_BLUE = {"image": "g_minus_b_clahe", "sigma": 1.0, "r0": 200.0, "length": 1000.0, "polarities": ("bright",)}
GREEN_WEIGHT = 0.25          # v1: log(1 + cirrus_raw) + 0.25 log(1 + green)
GMB_WEIGHT = 0.5             # final: log(1 + cirrus) + 0.5 log(1 + green_minus_blue)


def line_score(chip: Chip, cfg: dict, origin=None, expected_az=None) -> dict:
    """Angular-profile score of one band configuration (see :func:`angular_profile`).

    ``r0`` skips the aircraft itself: in B10 its 60 m blob, recorded at a slightly different
    time, would otherwise look like the start of a contrail.
    """
    origin = aircraft_pixel(chip) if origin is None else origin
    expected_az = expected_azimuth(chip) if expected_az is None else expected_az
    prof = angular_profile(image(chip, cfg["image"]), origin, cfg["sigma"], r0=cfg["r0"], lengths=(cfg["length"],))
    return profile_score(prof, expected_az, polarities=cfg["polarities"])


def rgb_line_offsets(chip: Chip, azimuth: float, origin=None, r0: float = 300.0, length: float = 1000.0,
                     sigma: float = 1.0, max_offset: int = 12) -> dict:
    """Line strength and sideways offset of a line along ``azimuth`` in B02, B03 and B04.

    A contrail is high, so band-to-band parallax shifts its blue, green and red copies sideways
    (median ~7 px between blue and red in this dataset); ground lines are not shifted. For each
    band the ridge response is averaged along the line at lateral offsets of +-``max_offset``
    px; the best offset and its strength (in units of the band's ridge MAD) are returned.
    ``rgb_min`` = strength of the weakest band, i.e. the line is visible in all three.
    """
    origin = aircraft_pixel(chip) if origin is None else origin
    t = np.deg2rad(azimuth)
    u = np.array([-np.cos(t), np.sin(t)]); n = np.array([-u[1], u[0]])
    d = np.arange(r0, length, RES) / RES
    offs = np.arange(-max_offset, max_offset + 1)
    out = {}
    best = {}
    for b in ("B02", "B03", "B04"):
        r = ridge_across(hessian(chip[b], sigma), u)
        scale = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), 1e-9)
        prof = np.array([np.nanmean(ndi.map_coordinates(r, [origin[0] + u[0] * d + n[0] * k, origin[1] + u[1] * d + n[1] * k],
                                                       order=1, mode="constant", cval=np.nan)) for k in offs]) / scale
        for pol, sgn in (("bright", -1.0), ("dark", 1.0)):
            q = sgn * prof
            k = int(np.nanargmax(q))
            best[(b, pol)] = (float(q[k]), int(offs[k]))
    for pol in ("bright", "dark"):
        strengths = [best[(b, pol)][0] for b in ("B02", "B03", "B04")]
        out[pol] = {"rgb_min": min(strengths), "offsets": [best[(b, pol)][1] for b in ("B02", "B03", "B04")]}
    pol = max(out, key=lambda k: out[k]["rgb_min"])
    return {"rgb_min": out[pol]["rgb_min"], "polarity": pol, "offsets_px": out[pol]["offsets"]}


def combined_score(cirrus: float, visible: float, weight: float = GMB_WEIGHT) -> float:
    """Formation score: log(1 + cirrus z) + weight * log(1 + visible-band line z)."""
    return float(np.log1p(max(cirrus, 0.0)) + weight * np.log1p(max(visible, 0.0)))


def formation_features(chip: Chip) -> dict:
    """All formation scores for one chip (one row of the evaluation table)."""
    o = aircraft_pixel(chip)
    az = expected_azimuth(chip)
    out = {"chip": chip["name"], "granule": chip["granule"], "formation": chip["formation"], "expected_az": az}
    out["corridor"] = corridor_contrast(chip["B03"], o, az)
    for name, cfg in (("green", GREEN), ("green_s2", {**GREEN, "sigma": 2.0}),
                      ("green_minus_blue_raw", {**GREEN_MINUS_BLUE, "image": "g_minus_b", "polarities": ("bright", "dark")}),
                      ("green_minus_maxbr", {**GREEN_MINUS_BLUE, "image": "g_minus_maxbr", "polarities": ("bright", "dark")}),
                      ("green_minus_blue", GREEN_MINUS_BLUE),
                      ("cirrus_raw", CIRRUS_RAW), ("cirrus_s6", {**CIRRUS_RAW, "sigma": 6.0}), ("cirrus", CIRRUS)):
        p = line_score(chip, cfg, o, az)
        out[name] = p["score"]
        out[f"{name}_az"] = p["azimuth"]
        out[f"{name}_pol"] = p["polarity"]
    rgb = rgb_line_offsets(chip, out["cirrus_az"], o)
    out["rgb_min"] = rgb["rgb_min"]
    out["rgb_offsets_px"] = rgb["offsets_px"]
    out["combined_v1"] = combined_score(out["cirrus_raw"], out["green"], GREEN_WEIGHT)
    out["combined"] = combined_score(out["cirrus"], out["green_minus_blue"], GMB_WEIGHT)
    out["b10_background"] = float(np.median(chip["B10"].astype("float32")))
    return out


# --- segmentation ----------------------------------------------------------------------------
def segment_contrail(
    chip: Chip,
    azimuth: float,
    polarity: str | None = None,
    sigma: float = 1.0,
    z_high: float = 3.0,
    z_low: float = 1.5,
    wedge: float = 15.0,
    r_min: float = 150.0,
    seed_range: tuple[float, float] = (150.0, 800.0),
    image_name: str = "g_minus_b",
) -> dict:
    """Segment the contrail behind the aircraft in the green band (``image_name``, see :func:`image`).

    1. Oriented ridge response of B03 for lines along ``azimuth`` (max over +-10 deg, so mildly
       bent contrails are kept), signed by ``polarity`` and robustly standardised.
    2. Seeds: pixels above ``z_high`` on the ray at ``azimuth`` between ``seed_range`` (m).
    3. Hysteresis: connected pixels above ``z_low`` inside a +-``wedge`` deg sector behind the
       aircraft and further than ``r_min`` from it; only components containing a seed are kept.
       Short, fading contrails stop early; long line-like ones keep growing.

    With ``polarity=None`` both bright and dark lines are tried and the larger segment is kept
    (a contrail over bright cloud can be darker than its surroundings).

    Defaults were chosen on the correctly detected contrails to balance coverage of and
    precision against the hand-drawn polygons (median 0.67 / 0.97).

    Returns the mask, its length along the azimuth (m), mean width (m), polarity and the z-map.
    """
    if polarity is None:
        a = segment_contrail(chip, azimuth, "bright", sigma, z_high, z_low, wedge, r_min, seed_range, image_name)
        b = segment_contrail(chip, azimuth, "dark", sigma, z_high, z_low, wedge, r_min, seed_range, image_name)
        return a if a["mask"].sum() >= b["mask"].sum() else b
    o = aircraft_pixel(chip)
    h = hessian(image(chip, image_name), sigma)
    sign = -1.0 if polarity == "bright" else 1.0
    resp = None
    for da in (-10.0, 0.0, 10.0):
        t = np.deg2rad(azimuth + da)
        r = sign * ridge_across(h, np.array([-np.cos(t), np.sin(t)]))
        resp = r if resp is None else np.maximum(resp, r)
    z = ad.robust_z(resp, border=5)

    rr, cc = np.mgrid[: z.shape[0], : z.shape[1]]
    dr, dc = rr - o[0], cc - o[1]
    dist = np.hypot(dr, dc) * RES
    pix_az = np.degrees(np.arctan2(dc, -dr)) % 360
    in_wedge = (np.abs((pix_az - azimuth + 180) % 360 - 180) <= wedge) & (dist >= r_min)

    t = np.deg2rad(azimuth)
    u = np.array([-np.cos(t), np.sin(t)])
    along = dr * u[0] + dc * u[1]
    across = np.abs(-dr * u[1] + dc * u[0])
    seed_zone = (along * RES >= seed_range[0]) & (along * RES <= seed_range[1]) & (across <= 4)
    low = (z > z_low) & in_wedge
    lab, _ = ndi.label(low, structure=np.ones((3, 3)))
    seeds = np.unique(lab[seed_zone & (z > z_high) & low])
    mask = np.isin(lab, seeds[seeds > 0])
    length = float(along[mask].max() * RES - r_min) if mask.any() else 0.0
    width = float(mask.sum() * RES * RES / max(length, RES)) if mask.any() else 0.0
    return {"mask": mask, "length_m": max(length, 0.0), "width_m": width, "polarity": polarity, "z": z}


def segment_vs_label(seg: np.ndarray, lab: np.ndarray | None, tol_px: int = 3) -> dict:
    """Overlap diagnostics with a ``tol_px`` tolerance buffer (labels are hand-drawn)."""
    if lab is None or not lab.any():
        return {"coverage": np.nan, "seg_precision": np.nan, "label_px": 0}
    st = np.ones((2 * tol_px + 1, 2 * tol_px + 1), bool)
    cov = float((lab & ndi.binary_dilation(seg, st)).sum() / lab.sum())
    prec = float((seg & ndi.binary_dilation(lab, st)).sum() / seg.sum()) if seg.any() else np.nan
    return {"coverage": cov, "seg_precision": prec, "label_px": int(lab.sum())}


# --- binary evaluation -----------------------------------------------------------------------
def binary_pr(score: np.ndarray, truth: np.ndarray) -> pd.DataFrame:
    """Precision/recall at every distinct threshold (accept when score >= threshold)."""
    order = np.argsort(-score, kind="stable")
    s, t = score[order], truth[order].astype(bool)
    tp = np.cumsum(t); n = np.arange(1, len(s) + 1)
    last = np.r_[s[1:] != s[:-1], True]
    return pd.DataFrame({"threshold": s[last], "precision": tp[last] / n[last], "recall": tp[last] / max(t.sum(), 1)})


def best_binary_threshold(score, truth) -> dict:
    pr = binary_pr(np.asarray(score, float), np.asarray(truth))
    pr["f1"] = 2 * pr.precision * pr.recall / (pr.precision + pr.recall).clip(lower=1e-9)
    pr["minpr"] = pr[["precision", "recall"]].min(axis=1)
    return pr.sort_values(["minpr", "f1"], ascending=False).iloc[0].to_dict()


def binary_eval(score, truth, threshold: float) -> dict:
    score, truth = np.asarray(score, float), np.asarray(truth, bool)
    acc = score >= threshold
    tp = (acc & truth).sum()
    return {"precision": tp / max(acc.sum(), 1), "recall": tp / max(truth.sum(), 1),
            "specificity": ((~acc) & ~truth).sum() / max((~truth).sum(), 1)}


def cross_validate_binary(df: pd.DataFrame, score_col: str, truth_col: str = "formation", seed: int = 0) -> pd.DataFrame:
    """Granule-level 2-fold CV of the threshold on ``score_col``."""
    in_a = df["granule"].isin(ad.granule_split(df["granule"], seed=seed))
    rows = []
    for fold, (tune, test) in enumerate([(in_a, ~in_a), (~in_a, in_a)]):
        b = best_binary_threshold(df.loc[tune, score_col], df.loc[tune, truth_col])
        rows.append({"fold": fold, "threshold": b["threshold"],
                     **binary_eval(df.loc[test, score_col], df.loc[test, truth_col], b["threshold"])})
    return pd.DataFrame(rows)


def _features_from_file(args) -> dict:
    path, meta = args
    with np.load(path, allow_pickle=False) as z:
        c = {k: z[k] for k in z.files}
    for k in ("label", "row", "truth_x", "truth_y", "granule", "crs"):
        c[k] = c[k].item()
    c["name"] = path.split("/")[-1][:-4]
    c.update(meta)
    f = formation_features(c)
    f["chip_size_m"] = float(len(c["x"]) * RES)
    return f


def run_features(chip_dir, csv: pd.DataFrame, n_jobs: int = 8) -> pd.DataFrame:
    """:func:`formation_features` for every ``pos_*.npz`` in ``chip_dir`` (workers load the files)."""
    from multiprocessing import Pool
    from pathlib import Path

    jobs = []
    for p in sorted(Path(chip_dir).glob("pos_*.npz")):
        row = csv.loc[int(p.stem[4:8])]
        meta = {f: row[f] for f in CSV_FIELDS}
        meta["formation"] = bool(row["contrail_formation"])
        jobs.append((str(p), meta))
    with Pool(n_jobs) as pool:
        return pd.DataFrame(pool.map(_features_from_file, jobs, chunksize=4))


def _segment_from_file(args) -> dict:
    path, meta, azimuth = args
    with np.load(path, allow_pickle=False) as z:
        c = {k: z[k] for k in z.files}
    for k in ("label", "row", "truth_x", "truth_y", "granule", "crs"):
        c[k] = c[k].item()
    c["name"] = path.split("/")[-1][:-4]
    c.update(meta)
    seg = segment_contrail(c, azimuth)
    lab = label_mask(c)
    o = aircraft_pixel(c)
    lab_len = np.nan
    if lab is not None and lab.any():
        t = np.deg2rad(azimuth)
        ij = np.argwhere(lab)
        lab_len = float(((ij[:, 0] - o[0]) * -np.cos(t) + (ij[:, 1] - o[1]) * np.sin(t)).max() * RES)
    return {"chip": c["name"], "seg_px": int(seg["mask"].sum()), "seg_length_m": seg["length_m"],
            "seg_width_m": seg["width_m"], "seg_polarity": seg["polarity"], "label_length_m": lab_len,
            **segment_vs_label(seg["mask"], lab)}


def run_segmentation(chip_dir, csv: pd.DataFrame, azimuths: dict[str, float], n_jobs: int = 8) -> pd.DataFrame:
    """:func:`segment_contrail` along the given azimuth (per chip name) for every chip in ``chip_dir``."""
    from multiprocessing import Pool
    from pathlib import Path

    jobs = []
    for p in sorted(Path(chip_dir).glob("pos_*.npz")):
        if p.stem not in azimuths:
            continue
        row = csv.loc[int(p.stem[4:8])]
        meta = {f: row[f] for f in CSV_FIELDS}
        meta["formation"] = bool(row["contrail_formation"])
        jobs.append((str(p), meta, azimuths[p.stem]))
    with Pool(n_jobs) as pool:
        return pd.DataFrame(pool.map(_segment_from_file, jobs, chunksize=4))


# --- straight-line contrail model ------------------------------------------------------------
def _line_z(chip: Chip, image_name: str, sigma: float, azimuth: float, polarity: str) -> np.ndarray:
    """Robust z of the oriented ridge response for lines along ``azimuth`` (signed by polarity)."""
    t = np.deg2rad(azimuth)
    r = ridge_across(hessian(image(chip, image_name), sigma), np.array([-np.cos(t), np.sin(t)]))
    return ad.robust_z(-r if polarity == "bright" else r, border=5)


def fit_contrail_line(
    chip: Chip,
    azimuth: float,
    image_name: str = "g_minus_b",
    sigma: float = 1.0,
    refine: float = 6.0,
    r_fit: tuple[float, float] = (150.0, 1000.0),
    z_end: float = 1.0,
    smooth_m: float = 200.0,
    max_gap_m: float = 200.0,
    z_free: tuple[float, float] = (3.0, 1.5),
) -> dict:
    """Model the contrail as a straight, thin rectangle starting at the aircraft.

    1. Direction: the azimuth (from the cirrus profile) is refined within +-``refine`` deg
       (0.5 deg steps) by the mean ridge z along the line between ``r_fit`` (m), for both
       polarities; the line is anchored at the aircraft.
    2. Length: the ridge z along the line (best of +-1 px across) is smoothed over
       ``smooth_m``; the contrail ends where it stays below ``z_end`` for more than ``max_gap_m``.
    3. Width: full width at half maximum of the mean cross-track z profile, 20-150 m.

    ``continuity`` is the share of the straight line between ``r_fit`` where the ridge z is at
    least ``z_end``: a contrail is a continuous straight line from the aircraft, a curved cloud
    edge or coastline only touches the line here and there.

    Also returns ``straightness``: the share of the free hysteresis segmentation
    (:func:`segment_contrail`, thresholds ``z_free``) that lies inside the rectangle widened by
    30 m. A contrail is straight; segments that curve along cloud edges, coasts or ridges are not.
    """
    o = aircraft_pixel(chip)
    H, W = chip["B03"].shape
    d_fit = np.arange(*r_fit, RES) / RES
    best = (-np.inf, azimuth, "bright", None)
    for pol in ("bright", "dark"):
        zmaps = {}
        for da in np.arange(-refine, refine + 0.25, 0.5):
            az = azimuth + da
            key = round(az / 3.0)  # one filtered map per 3 deg is close enough for the refinement
            if key not in zmaps:
                zmaps[key] = _line_z(chip, image_name, sigma, key * 3.0, pol)
            z = zmaps[key]
            t = np.deg2rad(az)
            u = np.array([-np.cos(t), np.sin(t)]); n = np.array([-u[1], u[0]])
            v = np.max([ndi.map_coordinates(z, [o[0] + u[0] * d_fit + n[0] * k, o[1] + u[1] * d_fit + n[1] * k],
                                            order=1, mode="constant", cval=np.nan) for k in (-1, 0, 1)], 0)
            m = np.nanmean(v)
            if m > best[0]:
                best = (m, az, pol, None)
    _, az, pol, _ = best
    z = _line_z(chip, image_name, sigma, az, pol)
    t = np.deg2rad(az)
    u = np.array([-np.cos(t), np.sin(t)]); n = np.array([-u[1], u[0]])
    d_all = np.arange(0, 4300, RES) / RES
    along = np.max([ndi.map_coordinates(z, [o[0] + u[0] * d_all + n[0] * k, o[1] + u[1] * d_all + n[1] * k],
                                        order=1, mode="constant", cval=np.nan) for k in (-1, 0, 1)], 0)
    inside = np.isfinite(along)
    k = max(int(smooth_m / RES), 1)
    sm = ndi.uniform_filter1d(np.nan_to_num(along, nan=-5.0), k)
    start = int(r_fit[0] / RES)
    gap = int(max_gap_m / RES)
    end = start
    below = 0
    for i in range(start, len(sm)):
        if not inside[i]:
            break
        if sm[i] >= z_end:
            end, below = i, 0
        else:
            below += 1
            if below > gap:
                break
    length = float(end * RES) if end > start else 0.0

    # width from the cross-track profile over the found length
    offs = np.arange(-15, 16)
    if length > 0:
        dl = d_all[start:end + 1]
        prof = np.array([np.nanmean(ndi.map_coordinates(z, [o[0] + u[0] * dl + n[0] * q, o[1] + u[1] * dl + n[1] * q],
                                                        order=1, mode="constant", cval=np.nan)) for q in offs])
        pk = np.nanmax(prof)
        width = float(np.clip((prof >= 0.5 * pk).sum() * RES, 20.0, 150.0)) if pk > 0 else 20.0
    else:
        width = 0.0

    rr, cc = np.mgrid[:H, :W]
    a_al = (rr - o[0]) * u[0] + (cc - o[1]) * u[1]
    a_ac = (rr - o[0]) * n[0] + (cc - o[1]) * n[1]
    rect = (a_al >= 0) & (a_al * RES <= length) & (np.abs(a_ac) * RES <= width / 2)

    free = segment_contrail(chip, az, pol, sigma, z_free[0], z_free[1], image_name=image_name)["mask"]
    band = (a_al >= 0) & (a_al * RES <= max(length, r_fit[1])) & (np.abs(a_ac) * RES <= width / 2 + 30.0)
    straightness = float((free & band).sum() / free.sum()) if free.any() else 0.0
    # straightness test on a fixed stretch: share of the straight line from r_fit[0] to r_fit[1]
    # (within the chip) where the ridge response is present
    fixed = along[int(r_fit[0] / RES):int(r_fit[1] / RES) + 1]
    fixed = fixed[np.isfinite(fixed)]
    continuity = float(np.mean(fixed >= z_end)) if fixed.size else 0.0
    return {"mask": rect, "azimuth": float(az), "polarity": pol, "length_m": length, "width_m": width,
            "straightness": straightness, "continuity": continuity, "free_mask": free,
            "line_mean_z": float(best[0])}


# --- detector seams --------------------------------------------------------------------------
def seam_check(chip: Chip, azimuth: float, near_m: float = 250.0, path_m: float = 1500.0,
               duplicate_threshold: float | None = 25.2) -> dict:
    """Is this chip affected by a Sentinel-2 detector seam?

    Flagged when

    * the aircraft lies within ``near_m`` of a detector border,
    * a border crosses the first ``path_m`` of the contrail path behind it, or
    * the aircraft detector finds a duplicate of the aircraft (a colour-reversed copy on the
      neighbouring detector, :func:`iagos_toolkit.imagery.aircraft_detection.detect`) in the
      central 2 x 2 km, with both copies scoring at least ``duplicate_threshold``.

    Near seams the aircraft and its contrail can appear twice or offset between neighbouring
    detectors, so the formation decision is unreliable. Chips without detector information
    are never flagged.
    """
    if "det_border_m" not in chip:
        return {"near_seam": False, "seam_reason": ""}
    o = aircraft_pixel(chip)
    bd, det = chip["det_border_m"], chip["detector"]
    t = np.deg2rad(azimuth)
    d = np.arange(0, path_m + RES, RES) / RES
    rr = np.clip(np.round(o[0] - np.cos(t) * d).astype(int), 0, det.shape[0] - 1)
    cc = np.clip(np.round(o[1] + np.sin(t) * d).astype(int), 0, det.shape[1] - 1)
    reasons = []
    if bd[o] < near_m:
        reasons.append(f"aircraft {bd[o]:.0f} m from a detector seam")
    ids = set(det[rr, cc].tolist()) - {0}
    if len(ids) > 1:
        reasons.append(f"seam crosses the contrail path (detectors {sorted(ids)})")
    if duplicate_threshold is not None:
        h = 100
        i0, j0 = max(o[0] - h, 0), max(o[1] - h, 0)
        sl = (slice(i0, i0 + 2 * h), slice(j0, j0 + 2 * h))
        crop = {k: chip[k][sl].astype("float32") for k in ("B02", "B03", "B04", "B10")}
        crop.update(detector=det[sl], det_border_m=bd[sl], x=chip["x"][sl[1]], y=chip["y"][sl[0]], name=chip.get("name", ""))
        if min(crop["B03"].shape) > 2 * ad.BORDER + 10:
            dup = ad.detect(crop, threshold=duplicate_threshold, warn=False)
            if dup["duplicate"]:
                sep = np.hypot(dup["x2"] - dup["x"], dup["y2"] - dup["y"])
                reasons.append(f"aircraft duplicated {sep:.0f} m away on detectors {dup['detector']}|{dup['detector2']}")
    return {"near_seam": bool(reasons), "seam_reason": "; ".join(reasons)}


def _line_fit_from_file(args) -> dict:
    path, meta, azimuth = args
    with np.load(path, allow_pickle=False) as z:
        c = {k: z[k] for k in z.files}
    for k in ("label", "row", "truth_x", "truth_y", "granule", "crs"):
        c[k] = c[k].item()
    c["name"] = path.split("/")[-1][:-4]
    c.update(meta)
    r = fit_contrail_line(c, azimuth)
    lab = label_mask(c)
    out = {"chip": c["name"], "line_az": r["azimuth"], "line_polarity": r["polarity"], "line_length_m": r["length_m"],
           "line_width_m": r["width_m"], "straightness": r["straightness"], "continuity": r["continuity"],
           "line_mean_z": r["line_mean_z"], **seam_check(c, r["azimuth"]), **segment_vs_label(r["mask"], lab)}
    if lab is not None and lab.any():
        o = aircraft_pixel(c); t = np.deg2rad(r["azimuth"]); ij = np.argwhere(lab)
        out["label_length_m"] = float(((ij[:, 0] - o[0]) * -np.cos(t) + (ij[:, 1] - o[1]) * np.sin(t)).max() * RES)
    return out


def run_line_fits(chip_dir, csv: pd.DataFrame, azimuths: dict[str, float], n_jobs: int = 8) -> pd.DataFrame:
    """:func:`fit_contrail_line` + :func:`seam_check` + label diagnostics for every chip."""
    from multiprocessing import Pool
    from pathlib import Path

    jobs = []
    for p in sorted(Path(chip_dir).glob("pos_*.npz")):
        if p.stem not in azimuths:
            continue
        row = csv.loc[int(p.stem[4:8])]
        meta = {f: row[f] for f in CSV_FIELDS}
        meta["formation"] = bool(row["contrail_formation"])
        jobs.append((str(p), meta, azimuths[p.stem]))
    with Pool(n_jobs) as pool:
        return pd.DataFrame(pool.map(_line_fit_from_file, jobs, chunksize=4))


# --- final decision ----------------------------------------------------------------------------
MIN_CONTINUITY = 0.4   # share of the straight line 150 m-1 km behind the aircraft with a ridge response
FORMATION_THRESHOLD = 2.2  # pooled τ on `combined` (2-fold CV medians 2.1-2.3)


def final_score(df: pd.DataFrame) -> pd.Series:
    """`combined`, or -inf where the straight-line fit is not continuous enough to be a contrail."""
    return df["combined"].where(df["continuity"] >= MIN_CONTINUITY, -np.inf)


def flag_seams(df: pd.DataFrame, warn: bool = True) -> pd.DataFrame:
    """Split off chips near a detector seam (``near_seam``) and warn that they should not be used."""
    seams = df[df["near_seam"]]
    if warn and len(seams):
        import warnings

        warnings.warn(
            f"{len(seams)} chips lie near a Sentinel-2 detector seam (aircraft/contrail may be duplicated "
            f"or offset between detectors). Their formation result is unreliable and they are excluded: "
            + ", ".join(seams["chip"]), stacklevel=2,
        )
    return df[~df["near_seam"]]


def detect_contrail(chip: Chip, threshold: float = FORMATION_THRESHOLD, warn: bool = True) -> dict:
    """Full pipeline for one chip: formation decision, straight-rectangle segment, seam check.

    ``formation`` is True when the combined score clears ``threshold`` and the straight-line
    fit is continuous (``continuity >= MIN_CONTINUITY``). Near a detector seam the result is
    returned but ``use`` is False and a warning is issued.
    """
    f = formation_features(chip)
    line = fit_contrail_line(chip, f["cirrus_az"])
    seam = seam_check(chip, line["azimuth"])
    formation = bool(f["combined"] >= threshold and line["continuity"] >= MIN_CONTINUITY)
    if seam["near_seam"] and warn:
        import warnings

        warnings.warn(f"{chip.get('name', 'chip')}: {seam['seam_reason']}; do not use this formation result", stacklevel=2)
    return {"formation": formation, "use": not seam["near_seam"], "score": f["combined"],
            "segment": line["mask"] if formation else np.zeros_like(line["mask"]),
            "length_m": line["length_m"] if formation else 0.0, "width_m": line["width_m"] if formation else 0.0,
            "azimuth": line["azimuth"], "continuity": line["continuity"], "straightness": line["straightness"],
            **seam, "features": f}
