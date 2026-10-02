"""Simple image-processing detectors for aircraft in Sentinel-2 chips.

Every method is a *response map*: a function of a chip (dict with 2-D arrays ``B02``,
``B03``, ``B04``, ``B10`` on a common 10 m grid plus 1-D ``x``/``y`` pixel-centre axes) that
returns a 2-D array, robustly standardised so values are comparable between chips.
:func:`detect` takes its arg-max as the aircraft location and its value as the score;
thresholding the score is left to the caller. No method uses the chip centre or any distance
prior; a ``BORDER``-pixel frame where the filters are unreliable is excluded from the arg-max.

Background:

* Sentinel-2 records B02, B03 and B04 a fraction of a second apart, so a moving aircraft
  appears as three displaced blobs (blue and red ~60-150 m on opposite sides of green), while
  static clouds and ground are co-located in all bands. Comparing green with the other visible
  bands removes most of the background. The manual labels sit on the green blob.
* Over dark surfaces the aircraft is *brighter* than its surroundings, over bright cloud it
  is *darker* (white paint ~0.4 vs thick cloud ~0.8 reflectance), so the green signal must be
  treated as signed.
* The cirrus band (B10) mostly sees objects high in the atmosphere; it confirms the aircraft but
  is acquired at yet another time (blob offset from green) and at 60 m, so it is only used as
  a capped, down-weighted term.
* Where neighbouring detectors overlap, the same aircraft can appear twice, with the blue/red
  order reversed in the copy. :func:`detect` warns about such duplicates.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from scipy import ndimage as ndi

Chip = dict[str, np.ndarray]
VISIBLE = ("B02", "B03", "B04")
BORDER = 10  # px (100 m) excluded from the arg-max: median/gaussian filters are unreliable there


def robust_z(a: np.ndarray, border: int = BORDER) -> np.ndarray:
    """Standardise with median and MAD (scaled to sigma) of the interior of ``a``."""
    inner = a[border:-border, border:-border]
    med = np.median(inner)
    mad = 1.4826 * np.median(np.abs(inner - med))
    return (a - med) / max(mad, 1e-9)


def tophat(a: np.ndarray, size: int) -> np.ndarray:
    """Remove the local background with a median filter of ``size`` pixels."""
    return a - ndi.median_filter(a, size=size, mode="reflect")


def _visible_tophats(chip: Chip, size: int = 7, sigma: float = 1.0) -> dict[str, np.ndarray]:
    """Smoothed top-hats of the visible bands, cached in the chip (they are reused a lot)."""
    key = ("_tophats", size, sigma)
    if key not in chip:
        chip[key] = {b: ndi.gaussian_filter(tophat(chip[b], size), sigma) for b in VISIBLE}
    return chip[key]


def band_excess(chip: Chip, band: str, size: int = 7, sigma: float = 1.0) -> np.ndarray:
    """Top-hat of ``band`` minus the brighter top-hat of the other two visible bands.

    Positive where ``band`` is brighter than both others: static features cancel, the moving
    (bright) aircraft does not.
    """
    th = _visible_tophats(chip, size, sigma)
    return th[band] - np.maximum(*[th[b] for b in VISIBLE if b != band])


def band_deficit(chip: Chip, band: str, size: int = 7, sigma: float = 1.0) -> np.ndarray:
    """Darker top-hat of the other two visible bands minus that of ``band``.

    Positive where ``band`` is darker than both others: the aircraft over bright cloud.
    """
    th = _visible_tophats(chip, size, sigma)
    return np.minimum(*[th[b] for b in VISIBLE if b != band]) - th[band]


def cirrus_response(chip: Chip, size: int = 19, sigma: float = 2.0) -> np.ndarray:
    """Small-scale B10 anomaly. ``size`` = 190 m keeps the ~60 m aircraft, drops cumulus blobs."""
    return ndi.gaussian_filter(tophat(chip["B10"], size), sigma)


# --- response maps ---------------------------------------------------------------------------
def green_tophat(chip: Chip) -> np.ndarray:
    """Brightest small-scale feature in B03."""
    return robust_z(ndi.gaussian_filter(tophat(chip["B03"], 15), 1.0))


def cirrus_anomaly(chip: Chip) -> np.ndarray:
    """Brightest small-scale feature in B10."""
    return robust_z(cirrus_response(chip))


def green_minus_red(chip: Chip) -> np.ndarray:
    """Top-hat of B03 - B04: removes background that is equally bright in green and red."""
    return robust_z(ndi.gaussian_filter(tophat(chip["B03"] - chip["B04"], 7), 1.0))


def green_minus_blue(chip: Chip) -> np.ndarray:
    """Top-hat of B03 - B02: removes background that is equally bright in green and blue."""
    return robust_z(ndi.gaussian_filter(tophat(chip["B03"] - chip["B02"], 7), 1.0))


def green_excess(chip: Chip) -> np.ndarray:
    """Green blob without a co-located blue or red counterpart (bright aircraft only)."""
    return robust_z(band_excess(chip, "B03"))


def green_signed(chip: Chip) -> np.ndarray:
    """Green excess *or* green deficit, whichever is stronger (bright or dark aircraft)."""
    return np.maximum(robust_z(band_excess(chip, "B03")), robust_z(band_deficit(chip, "B03")))


def green_cirrus_capped(chip: Chip, weight: float = 0.75, cap: float = 30.0) -> np.ndarray:
    """Green excess plus a capped, down-weighted cirrus term.

    The cap stops bright cumulus tops (huge B10 anomalies, little green excess) from being
    detected on the cirrus term alone.
    """
    return green_excess(chip) + weight * np.minimum(cirrus_anomaly(chip), cap)


def signed_green_cirrus_capped(chip: Chip, weight: float = 0.75, cap: float = 30.0) -> np.ndarray:
    """Signed green plus a capped, down-weighted cirrus term (final method).

    ``weight`` and ``cap`` were tuned with granule-level 2-fold cross-validation
    (notebooks/aircraft_detection.ipynb).
    """
    return green_signed(chip) + weight * np.minimum(cirrus_anomaly(chip), cap)


METHODS: dict[str, Callable[[Chip], np.ndarray]] = {
    "green_tophat": green_tophat,
    "cirrus_anomaly": cirrus_anomaly,
    "green_minus_red": green_minus_red,
    "green_minus_blue": green_minus_blue,
    "green_excess": green_excess,
    "green_signed": green_signed,
    "green_cirrus_capped": green_cirrus_capped,
    "signed_green_cirrus_capped": signed_green_cirrus_capped,
}

# A detector = a response map plus optional rejection rules (see :func:`detect`).
DETECTORS: dict[str, dict] = {name: {"response": name} for name in METHODS}
DETECTORS["signed_green_cirrus_capped_sized"] = {
    "response": "signed_green_cirrus_capped",
    "max_cirrus_area": 250,  # px at 10 m (0.025 km2); tuned with 2-fold CV
    "min_border_dist": 20,  # m; detections on a detector seam are rejected
}
FINAL_METHOD = "signed_green_cirrus_capped_sized"


def response(chip: Chip, detector: str) -> np.ndarray:
    """Response map used by ``detector`` (a key of :data:`DETECTORS` or :data:`METHODS`)."""
    name = DETECTORS[detector]["response"] if detector in DETECTORS else detector
    return METHODS[name](chip)


def cirrus_blob_area(chip: Chip, i: int, j: int, search: int = 6, frac: float = 0.5) -> int:
    """Size (px at 10 m) of the B10 object at (i, j).

    The B10 top-hat maximum within ``search`` px of (i, j) is taken as the object's peak; the
    area is the connected region above ``frac`` x that peak. An aircraft (~60 m, smeared by
    the 60 m pixels and the smoothing) gives ~50-160 px; low cumulus clouds, which are also
    moving and show a slight colour offset, are usually much larger.
    """
    ct = cirrus_response(chip)
    i0, j0 = max(i - search, 0), max(j - search, 0)
    sub = ct[i0 : i + search + 1, j0 : j + search + 1]
    a, b = np.unravel_index(np.argmax(sub), sub.shape)
    ci, cj = i0 + a, j0 + b
    if ct[ci, cj] <= 0:
        return 0
    lab, _ = ndi.label(ct >= frac * ct[ci, cj])
    return int((lab == lab[ci, cj]).sum())


# --- detection -------------------------------------------------------------------------------
def _argmax(response: np.ndarray, exclude: np.ndarray | None = None, border: int = BORDER):
    r = response.copy()
    r[:border], r[-border:], r[:, :border], r[:, -border:] = -np.inf, -np.inf, -np.inf, -np.inf
    if exclude is not None:
        r[exclude] = -np.inf
    i, j = np.unravel_index(np.argmax(r), r.shape)
    return int(i), int(j), float(r[i, j])


def ghost_direction(chip: Chip, i: int, j: int, radius: int = 20) -> np.ndarray | None:
    """Unit vector (row, col) from the blue to the red ghost of the aircraft at (i, j).

    Uses the excess maps for a bright aircraft and the deficit maps for a dark one.
    """
    bright = band_excess(chip, "B03")[i, j] >= band_deficit(chip, "B03")[i, j]
    f = band_excess if bright else band_deficit
    i0, i1, j0, j1 = max(i - radius, 0), i + radius + 1, max(j - radius, 0), j + radius + 1
    pts = []
    for b in ("B02", "B04"):
        sub = f(chip, b)[i0:i1, j0:j1]
        a, c = np.unravel_index(np.argmax(sub), sub.shape)
        pts.append(np.array([i0 + a, j0 + c], float))
    v = pts[1] - pts[0]
    n = np.hypot(*v)
    return v / n if n > 0 else None


def detect(
    chip: Chip,
    method: str = FINAL_METHOD,
    threshold: float | None = None,
    max_cirrus_area: float | None = ...,
    min_border_dist: float | None = ...,
    min_ratio: float = 0.8,
    sep_range: tuple[float, float] = (250.0, 2000.0),
    warn: bool = True,
) -> dict:
    """Locate the aircraft with detector ``method`` and check for a duplicate.

    Returns ``x``, ``y``, ``score`` (the response at the arg-max) and diagnostics:

    * ``cirrus_area``: size of the B10 object at the detection (:func:`cirrus_blob_area`).
      If it exceeds ``max_cirrus_area`` (default: the detector's setting in
      :data:`DETECTORS`), the object is too large to be an aircraft: ``rejected`` is set and
      ``score`` becomes -inf (``score_raw`` keeps the value).
    * ``detector`` / ``border_dist``: Sentinel-2 detector id and distance (m) to the nearest
      detector border at the detection, if the chip has detector information
      (:func:`iagos_toolkit.imagery.chips.add_detector_info`). Detections closer than
      ``min_border_dist`` (default: the detector's setting) to a border are rejected the same way.
    * ``x2``, ``y2``, ``score2``, ``detector2``: the strongest peak at least
      ``sep_range[0]`` m away, ``ghost_cos`` (cosine between the blue->red ghost directions
      of both peaks; -1 = reversed) and ``duplicate``.

    ``duplicate`` is True when the second peak scores at least ``min_ratio`` x the first (and
    >= ``threshold`` if given), lies within ``sep_range`` m, shows the reversed blue/red order,
    and - when detector information is available - lies on a different detector. That is what
    a copy of the same aircraft from the neighbouring detector looks like; a
    :class:`UserWarning` is issued then.
    """
    opts = DETECTORS.get(method, {})
    if max_cirrus_area is ...:
        max_cirrus_area = opts.get("max_cirrus_area")
    if min_border_dist is ...:
        min_border_dist = opts.get("min_border_dist")
    resp = response(chip, method)
    i, j, score = _argmax(resp)
    yy, xx = np.mgrid[: resp.shape[0], : resp.shape[1]]
    near = np.hypot(yy - i, xx - j) * 10.0 < sep_range[0]
    i2, j2, score2 = _argmax(resp, exclude=near)
    sep = 10.0 * np.hypot(i2 - i, j2 - j)

    has_det = "detector" in chip
    det1 = int(chip["detector"][i, j]) if has_det else 0
    det2 = int(chip["detector"][i2, j2]) if has_det else 0
    border_dist = float(chip["det_border_m"][i, j]) if "det_border_m" in chip else np.nan

    area = cirrus_blob_area(chip, i, j)
    rejected = ""
    if max_cirrus_area is not None and area > max_cirrus_area:
        rejected = "cirrus object too large"
    elif min_border_dist is not None and border_dist < min_border_dist:
        rejected = "near detector border"

    ghost_cos = np.nan
    if score2 >= min_ratio * score and sep <= sep_range[1]:
        g1, g2 = ghost_direction(chip, i, j), ghost_direction(chip, i2, j2)
        if g1 is not None and g2 is not None:
            ghost_cos = float(np.dot(g1, g2))
    other_detector = (det1 != det2) if (has_det and det1 > 0 and det2 > 0) else True
    duplicate = bool(
        ghost_cos < -0.5 and other_detector and not rejected
        and (threshold is None or score2 >= threshold)
    )
    if duplicate and warn:
        warnings.warn(
            f"{chip.get('name', 'chip')}: possible duplicate of the aircraft {sep:.0f} m away "
            f"(score {score2:.0f} vs {score:.0f}, reversed blue/red order"
            + (f", detectors {det1}/{det2}" if has_det else "")
            + "); the detection may be the copy rather than the labelled aircraft",
            stacklevel=2,
        )
    return {
        "x": float(chip["x"][j]), "y": float(chip["y"][i]),
        "score": -np.inf if rejected else score, "score_raw": score, "rejected": rejected,
        "cirrus_area": area, "detector": det1, "border_dist": border_dist,
        "x2": float(chip["x"][j2]), "y2": float(chip["y"][i2]), "score2": score2,
        "detector2": det2, "ghost_cos": ghost_cos, "duplicate": duplicate,
    }


# --- evaluation ------------------------------------------------------------------------------
def evaluate(results, threshold: float, tol: float = 100.0) -> dict[str, float]:
    """Precision/recall/specificity over positive and negative chips.

    ``results`` is a DataFrame with columns ``label`` (1/0), ``score`` and ``dist`` (distance
    to the truth in m; NaN for negatives).
    """
    acc = results["score"] >= threshold
    pos = results["label"] == 1
    tp = (acc & pos & (results["dist"] <= tol)).sum()
    fp = (acc & ~(pos & (results["dist"] <= tol))).sum()
    n_pos, n_neg = pos.sum(), (~pos).sum()
    fp_neg = (acc & ~pos).sum()
    return {
        "threshold": threshold,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(n_pos, 1),
        "specificity": 1 - fp_neg / max(n_neg, 1),
        "tp": int(tp), "fp": int(fp), "fp_on_negatives": int(fp_neg),
        "loc_recall": float((results.loc[pos, "dist"] <= tol).mean()),
        "median_err_m": float(results.loc[pos, "dist"].median()),
    }


def pr_curve(results, tol: float = 100.0):
    """Precision/recall/specificity at every distinct score threshold (vectorised)."""
    import pandas as pd

    order = np.argsort(-results["score"].to_numpy(), kind="stable")
    score = results["score"].to_numpy()[order]
    pos = (results["label"].to_numpy() == 1)[order]
    hit = pos & (results["dist"].to_numpy()[order] <= tol)
    tp, n_acc, fp_neg = np.cumsum(hit), np.arange(1, len(score) + 1), np.cumsum(~pos)
    last = np.r_[score[1:] != score[:-1], True]  # last index of every distinct threshold
    return pd.DataFrame({
        "threshold": score[last],
        "precision": tp[last] / n_acc[last],
        "recall": tp[last] / max(pos.sum(), 1),
        "specificity": 1 - fp_neg[last] / max((~pos).sum(), 1),
    })


def best_threshold(results, tol: float = 100.0) -> dict[str, float]:
    """Threshold maximising min(precision, recall) (ties: higher F1)."""
    pr = pr_curve(results, tol)
    pr["f1"] = 2 * pr.precision * pr.recall / (pr.precision + pr.recall).clip(lower=1e-9)
    pr["minpr"] = pr[["precision", "recall"]].min(axis=1)
    return pr.sort_values(["minpr", "f1"], ascending=False).iloc[0].to_dict()


def _run_one(args) -> list[dict]:
    chip, names = args
    rows = []
    for name in names:
        d = detect(chip, name, warn=False)
        dist = np.hypot(d["x"] - chip["truth_x"], d["y"] - chip["truth_y"]) if chip["label"] == 1 else np.nan
        rows.append({"chip": chip["name"], "granule": chip["granule"], "label": chip["label"],
                     "method": name, **d, "dist": dist})
    return rows


def run_detectors(chips: list[Chip], methods: list[str] | None = None, n_jobs: int = 1):
    """Run :func:`detect` with the named detectors (default: all of :data:`DETECTORS`) on every chip.

    Returns a long DataFrame with one row per chip x method. The ``duplicate`` column ignores
    the score threshold; use :func:`flag_duplicates` once a threshold is chosen.
    """
    import pandas as pd

    names = list(DETECTORS) if methods is None else list(methods)
    jobs = [(c, names) for c in chips]
    if n_jobs == 1:
        out = map(_run_one, jobs)
    else:
        from multiprocessing import Pool

        with Pool(n_jobs) as pool:
            out = pool.map(_run_one, jobs, chunksize=16)
    return pd.DataFrame([r for rows in out for r in rows])


def granule_split(granules, frac_dev: float = 0.5, seed: int = 0) -> set[str]:
    """Seeded random subset of granule ids used as the dev (tuning) split."""
    g = np.array(sorted(set(granules)))
    rng = np.random.default_rng(seed)
    return set(rng.choice(g, size=int(round(frac_dev * len(g))), replace=False))


def cross_validate(results, seed: int = 0, tol: float = 100.0):
    """Granule-level 2-fold CV of the score threshold for one method.

    The threshold is tuned on one half (max min(P, R)) and evaluated on the other, then the
    halves swap. Returns one row per fold.
    """
    import pandas as pd

    in_a = results["granule"].isin(granule_split(results["granule"], seed=seed))
    rows = []
    for fold, (tune, test) in enumerate([(in_a, ~in_a), (~in_a, in_a)]):
        b = best_threshold(results[tune], tol)
        t = evaluate(results[test], b["threshold"], tol)
        rows.append({"fold": fold, "threshold": b["threshold"], "tune_P": b["precision"],
                     "tune_R": b["recall"], **{k: t[k] for k in ("precision", "recall", "specificity",
                                                                 "tp", "fp", "fp_on_negatives")}})
    return pd.DataFrame(rows)


def flag_duplicates(results, threshold: float):
    """Set ``duplicate`` (second peak also >= threshold) and warn once per flagged chip."""
    results = results.copy()
    results["duplicate"] = results["duplicate"] & (results["score2"] >= threshold) & (results["score"] >= threshold)
    for _, r in results[results["duplicate"]].iterrows():
        sep = np.hypot(r.x2 - r.x, r.y2 - r.y)
        warnings.warn(f"{r.chip} ({r.method}): possible duplicate aircraft {sep:.0f} m away "
                      f"(score {r.score2:.0f} vs {r.score:.0f}, reversed blue/red order)", stacklevel=2)
    return results
