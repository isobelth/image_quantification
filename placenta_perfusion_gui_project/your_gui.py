from dataclasses import dataclass
from heapq import heappop, heappush
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import warnings

from IPython.display import display
import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
from liffile import LifFile
from magicgui import magicgui
from magicgui.widgets import Container, TextEdit
from matplotlib.colors import to_rgba
from napari.utils.notifications import show_error, show_warning
from qtpy.QtWidgets import QApplication, QFileDialog
from scipy.ndimage import gaussian_filter
from skimage import exposure
from scipy.ndimage import binary_fill_holes
from skimage.draw import line as sk_line
from skimage.filters import median, sato, threshold_li, threshold_minimum, threshold_otsu, threshold_triangle
from skimage.measure import label
from skimage.morphology import binary_closing, binary_dilation, binary_erosion, binary_opening, disk, remove_small_holes, remove_small_objects, skeletonize
warnings.filterwarnings("ignore")


def _running_in_notebook() -> bool:
    """Best-effort check: if running under an IPython kernel, don't quit Python on window close."""
    if "ipykernel" in sys.modules:
        return True
    try:
        from IPython import get_ipython

        ip = get_ipython()
        return ip is not None and "IPKernelApp" in getattr(ip, "config", {})
    except Exception:
        return False


# -------------------------
# Processing + segmentation
# -------------------------

@dataclass(frozen=True)
class RidgeParams:
    ratio_cutoff: float = 0.95 

    # edge mode fallbacks
    edge_extend_gap: int = 30

    # ridge mode settings
    sato_sigmas: Tuple[float, ...] = (2, 3, 4, 5, 6, 7, 8)
    ridge_close_disk: int = 9
    ridge_min_size: int = 600
    ridge_keep_components: int = 3
    ridge_extend_gap: int = 50
    edge_margin: int = 30
    step_cost: float = 100.0


class PlacentaPerfusionProcessor:
    _NEIGH8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def __init__(self, params: Optional[RidgeParams] = None):
        self.params = params or RidgeParams()

    @staticmethod
    def label_colormap(color, alpha=1.0):
        rgba = np.array(to_rgba(color))
        rgba[3] = alpha
        return {0: np.array([0.0, 0.0, 0.0, 0.0]), 1: rgba}

    def vertical_brightness_ratio(self, img: np.ndarray, eps: float = 1e-6) -> float:
        """Abs(log(median(top)/median(bottom))). Used to gate edge vs ridge mode."""
        y = img.shape[0] // 2
        a = float(np.median(img[:y]))
        b = float(np.median(img[y:]))
        return float(abs(np.log((a + eps) / (b + eps))))

    def fit_line(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Fast least-squares y=a*x+b, avoiding np.polyfit overhead."""
        n = x.size
        if n < 2:
            return 0.0, float(y[0]) if n else 0.0

        x = x.astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)

        xm, ym = x.mean(), y.mean()
        dx = x - xm
        denom = float(np.dot(dx, dx))
        if denom == 0.0:
            return 0.0, float(ym)

        a = float(np.dot(dx, y - ym) / denom)
        b = float(ym - a * xm)
        return a, b

    def extend_path_to_edges(self, path_mask: np.ndarray, k: int = 5, max_gap: int = 30, edge_tol: int = 0) -> np.ndarray:
        """
        Extend a path to x=0 and/or x=W-1 if its endpoints are within max_gap of those edges.
        Uses k extreme points and a linear fit to extrapolate.
        """
        m = path_mask.astype(bool).copy()
        H, W = m.shape
        ys, xs = np.nonzero(m)
        n = xs.size
        if n < 2:
            return m

        x_min, x_max = int(xs.min()), int(xs.max())
        left_gap, right_gap = x_min, (W - 1) - x_max

        if left_gap <= edge_tol and right_gap <= edge_tol:
            return m
        if left_gap > max_gap and right_gap > max_gap:
            return m

        kk = min(k, n)

        if left_gap <= max_gap:
            idx = np.argpartition(xs, kk - 1)[:kk]
            a, b = self.fit_line(xs[idx], ys[idx])
            x0 = x_min
            y0 = int(np.clip(np.rint(a * x0 + b), 0, H - 1))
            y_edge = int(np.clip(np.rint(b), 0, H - 1))  # x=0
            rr, cc = sk_line(y_edge, 0, y0, x0)
            m[rr, cc] = True

        if right_gap <= max_gap:
            idx = np.argpartition(xs, n - kk)[-kk:]
            a, b = self.fit_line(xs[idx], ys[idx])
            x1 = x_max
            y1 = int(np.clip(np.rint(a * x1 + b), 0, H - 1))
            y_edge = int(np.clip(np.rint(a * (W - 1) + b), 0, H - 1))
            rr, cc = sk_line(y1, x1, y_edge, W - 1)
            m[rr, cc] = True

        return m

    def _largest_n_components(self, mask: np.ndarray, n_keep: int = 3) -> np.ndarray:
        lab = label(mask)
        if lab.max() == 0:
            return mask.astype(bool)
        areas = np.bincount(lab.ravel())
        areas[0] = 0
        keep = np.argsort(areas)[-n_keep:]
        return np.isin(lab, keep)

    def extract_boundary(self, img: np.ndarray, thr_func, min_size: int = 100, hole_area: int = 500, opening_size: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Threshold + clean + keep largest region + return interface boundary as a 1px mask."""
        bw = img > thr_func(img)
        if min_size > 0:
            bw = remove_small_objects(bw, min_size=min_size)
        if hole_area > 0:
            bw = remove_small_holes(bw, area_threshold=hole_area)
        if opening_size > 0:
            bw = binary_opening(bw, disk(opening_size))

        bw = self._largest_n_components(bw, n_keep=1)

        # boundary of region
        path = bw ^ binary_erosion(bw, disk(1))
        path = self._largest_n_components(path, n_keep=1)
        return path.astype(bool), bw.astype(bool)

    def find_best_left_right_route(self, mask: np.ndarray, edge_margin: int = 30, step_cost: float = 100.0) -> Optional[np.ndarray]:
        """
        Multi-source Dijkstra from any left-edge point to any right-edge point.
        Prefers paths on mask; allows small off-mask moves via penalty.
        """
        H, W = mask.shape
        left = np.argwhere(mask[:, :edge_margin])
        right = np.argwhere(mask[:, W - edge_margin :])

        if left.size == 0 or right.size == 0:
            m2 = binary_dilation(mask, disk(2))
            left = np.argwhere(m2[:, :edge_margin])
            right = np.argwhere(m2[:, W - edge_margin :])
            if left.size == 0 or right.size == 0:
                return None
            mask = m2

        right_set = {(int(y), int(W - edge_margin + x)) for y, x in right}

        off_penalty = step_cost * 0.2
        cost = np.where(mask, step_cost, step_cost + off_penalty).astype(np.float32, copy=False)

        visited = np.zeros((H, W), dtype=bool)
        prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
        pq: List[Tuple[float, Tuple[int, int]]] = []

        for y, x in left:
            p = (int(y), int(x))
            heappush(pq, (0.0, p))
            prev[p] = None

        end: Optional[Tuple[int, int]] = None

        while pq:
            c, (y, x) = heappop(pq)
            if visited[y, x]:
                continue
            visited[y, x] = True
            if (y, x) in right_set:
                end = (y, x)
                break

            for dy, dx in self._NEIGH8:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx]:
                    nc = c + float(cost[ny, nx])
                    heappush(pq, (nc, (ny, nx)))
                    prev.setdefault((ny, nx), (y, x))

        if end is None:
            return None

        out = np.zeros((H, W), dtype=bool)
        cur: Optional[Tuple[int, int]] = end
        while cur is not None:
            out[cur] = True
            cur = prev[cur]
        return out

    def extract_edge_mode_boundary(self, img_med: np.ndarray, img_shape: Tuple[int, int], params: Optional[RidgeParams] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Edge-mode boundary extraction with robustness fallbacks."""
        params = params or self.params
        path, bw = self.extract_boundary(img_med, threshold_li)

        H, W = img_shape
        if path.sum() > 1.5 * W:
            path, bw = self.extract_boundary(img_med, threshold_minimum, min_size=0, hole_area=500, opening_size=0)
            if path.sum() > 1.5 * W:
                holes = binary_fill_holes(bw) & ~bw
                holes_grown = binary_dilation(holes, disk(20))
                bw_big_holes = bw & ~holes_grown

                path = bw_big_holes ^ binary_erosion(bw_big_holes, disk(20))
                path = self._largest_n_components(path, n_keep=1).astype(bool)
                if path.sum() > 1.5 * W:
                    path, bw = self.extract_ridge_mode_boundary(img_med, params=params)

        elif path[0, :].any() or path[-1, :].any():
            g = exposure.adjust_gamma(img_med, gamma=0.2)
            path, bw = self.extract_boundary(g, threshold_triangle, min_size=0, hole_area=0, opening_size=10)

        path = self.extend_path_to_edges(path, k=5, max_gap=params.edge_extend_gap)
        return path, bw

    def extract_ridge_mode_boundary(self, img_med: np.ndarray, params: Optional[RidgeParams] = None) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Ridge-mode boundary extraction via Sato + skeleton routing."""
        params = params or self.params
        resp = sato(img_med, sigmas=params.sato_sigmas, black_ridges=True)
        resp = exposure.rescale_intensity(resp)
        bw = resp > threshold_otsu(resp)
        bw = binary_closing(bw, footprint=disk(params.ridge_close_disk))
        bw = remove_small_objects(bw, min_size=params.ridge_min_size)

        skel = skeletonize(bw)
        skel_main = self._largest_n_components(skel, n_keep=params.ridge_keep_components)

        path = self.find_best_left_right_route(skel_main, edge_margin=params.edge_margin, step_cost=params.step_cost)
        if path is None:
            path = self.find_best_left_right_route(skel, edge_margin=params.edge_margin, step_cost=params.step_cost)

        if path is not None:
            path = self.extend_path_to_edges(path, k=5, max_gap=params.ridge_extend_gap)

        return path, bw

    def find_ridge_through_plane(self, img: np.ndarray, params: Optional[RidgeParams] = None, z: Optional[int] = None) -> Tuple[Optional[np.ndarray], str, float]:
        """
        Returns (path_mask, mode, brightness_ratio).
        mode in {"no_image","edge","ridge","fail"}.
        """
        params = params or self.params
        vr = self.vertical_brightness_ratio(img)
        if vr == 0.0:
            return None, "no_image", vr

        try:
            img_med = median(img, disk(7))
            mode = "edge" if vr > params.ratio_cutoff else "ridge"

            if mode == "edge":
                path, bw = self.extract_edge_mode_boundary(img_med, img.shape, params=params)
            else:
                path, bw = self.extract_ridge_mode_boundary(img_med, params=params)

            return path, mode, vr

        except Exception:
            return None, "fail", vr

    def pathmask_to_profile_y(self, pm: np.ndarray) -> Optional[np.ndarray]:
        """Convert a sparse 1px-ish path mask into a dense y(x) profile of length W."""
        H, W = pm.shape
        y, x = np.nonzero(pm)
        if y.size < 2:
            return None
        order = np.argsort(x)
        x = x[order].astype(np.float32)
        y = y[order].astype(np.float32)
        x_u, idx = np.unique(x, return_index=True)
        y_u = y[idx]
        if x_u.size < 2:
            return None
        return np.interp(np.arange(W, dtype=np.float32), x_u, y_u, left=y_u[0], right=y_u[-1]).astype(np.float32)

    def collect_lines_over_z(self, vol_zyx: np.ndarray, params: Optional[RidgeParams] = None) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], List[int]]:
        """
        Returns:
          lines[z]     -> bool (H,W), this is a per-slice path mask (boolean HxW image)
          profiles[z]  -> float32 (W,), this is the corresponding dense y(x) curve (length W float array) extracted from the mask
          found_zs     -> sorted list of z where a line exists
        """
        params = params or self.params
        Z, H, W = vol_zyx.shape
        lines: Dict[int, np.ndarray] = {}
        profiles: Dict[int, np.ndarray] = {}
        found: List[int] = []

        for z in range(Z):
            pm, mode, vr = self.find_ridge_through_plane(vol_zyx[z], params=params, z=z)
            if pm is None or pm.sum() == 0:
                continue
            pm = pm.astype(bool)
            yx = self.pathmask_to_profile_y(pm)
            if yx is None:
                continue
            lines[z] = pm
            profiles[z] = yx
            found.append(z)

        return lines, profiles, sorted(found)

    def keep_similar_high_z(self, lines: Dict[int, np.ndarray], profiles: Dict[int, np.ndarray], found_zs: List[int], *, missing_run: int = 3, diff_thresh_px: Optional[float] = None, core_frac: float = 0.6, core_min: int = 5, min_keep: int = 3, consec_bad: int = 2) -> Tuple[List[int], Dict[int, np.ndarray]]:
        """
        Keep the main contiguous z-band (trims weird ends), with:
          - low-z pre-cut before first missing-run
          - optional exclusion of paths touching top/bottom rows
        """
        zs = sorted([z for z in found_zs if z in lines and z in profiles])
        if not zs:
            return [], {}
        # exclude lines touching top and bottom
        zs = [z for z in zs if not (lines[z][0, :].any() or lines[z][-1, :].any())]
        if not zs:
            return [], {}

        if len(zs) == 1:
            z0 = zs[0]
            return [z0], {z0: lines[z0]}

        # pre-cut low-z before first missing run (based on remaining zs)
        cut_z = None
        for a, b in zip(zs[:-1], zs[1:]):
            if (b - a - 1) >= missing_run:
                cut_z = a + missing_run
                break
        if cut_z is not None:
            zs = [z for z in zs if z > cut_z]
        if not zs:
            return [], {}

        n = len(zs)
        if n == 1:
            z0 = zs[0]
            return [z0], {z0: lines[z0]}

        core_n = min(n, max(core_min, int(np.ceil(core_frac * n))))
        start = (n - core_n) // 2
        core = zs[start : start + core_n]
        consensus = np.median(np.stack([profiles[z] for z in core], axis=0), axis=0)

        diffs = np.array([np.median(np.abs(profiles[z] - consensus)) for z in zs], dtype=np.float32)

        # if diff_thresh_px is None:
        #     core_diffs = np.array([np.median(np.abs(profiles[z] - consensus)) for z in core], dtype=np.float32)
        #     med = float(np.median(core_diffs))
        #     mad = float(np.median(np.abs(core_diffs - med))) + 1e-9
        #     thr = med + 3.0 * 1.4826 * mad
        # else:
        thr = float(diff_thresh_px)

        good = diffs <= thr

        best = None  # (score, i0, i1)
        i = 0
        while i < n:
            if not good[i]:
                i += 1
                continue

            i0, bad_run, j = i, 0, i
            while j < n:
                if good[j]:
                    bad_run = 0
                else:
                    bad_run += 1
                    if bad_run >= consec_bad:
                        break
                j += 1

            i1 = j
            score = (int(good[i0:i1].sum()), i1 - i0)
            if best is None or score > best[0]:
                best = (score, i0, i1)
            i = i1

        if best is None:
            return [], {}

        _, i0, i1 = best
        kept = [z for z, g in zip(zs[i0:i1], good[i0:i1]) if bool(g)]

        if len(kept) < min_keep:
            kept = [zs[int(np.argmin(diffs))]]

        return kept, {z: lines[z] for z in kept}

    def smooth_surface_yzx(self, profiles: Dict[int, np.ndarray], kept_zs: List[int], Z: int, W: int, *, sigma_z: float = 1.0, sigma_x: float = 2.0) -> np.ndarray:
        """Build y[z,x] from kept_zs, interpolate along z per x, then smooth in (z,x)."""
        ysurf = np.full((Z, W), np.nan, dtype=np.float32)
        for z in kept_zs:
            ysurf[z] = profiles[z].astype(np.float32, copy=False)

        zz = np.arange(Z, dtype=np.float32)
        for x in range(W):
            col = ysurf[:, x]
            ok = ~np.isnan(col)
            if ok.sum() >= 2:
                ysurf[:, x] = np.interp(zz, zz[ok], col[ok]).astype(np.float32)
            elif ok.sum() == 1:
                ysurf[:, x] = float(col[ok][0])

        return gaussian_filter(ysurf, sigma=(sigma_z, sigma_x), mode="nearest")

    def fill_missing_profiles(self, profiles: Dict[int, np.ndarray], Z: int) -> np.ndarray:
        """
        profiles[z] is (W,) y(x) for that z.
        Returns y_full with shape (Z, W), filled for all z by linear interpolation in z.
        """
        zs = np.array(sorted(profiles.keys()), dtype=int)
        if zs.size == 0:
            raise ValueError("No profiles provided")

        W = int(profiles[zs[0]].shape[0])
        y_full = np.full((Z, W), np.nan, dtype=np.float32)

        for z in zs:
            y_full[z] = profiles[z].astype(np.float32, copy=False)

        zz = np.arange(Z, dtype=np.float32)
        for x in range(W):
            col = y_full[:, x]
            ok = ~np.isnan(col)
            if ok.sum() >= 2:
                col[~ok] = np.interp(zz[~ok], zz[ok], col[ok])
                y_full[:, x] = col
            elif ok.sum() == 1:
                y_full[:, x] = col[ok][0]  # constant fill if only one slice exists

        return y_full

    def profiles_to_path_masks(self, y_full: np.ndarray, H: int) -> np.ndarray:
        """
        y_full is (Z,W) of y(x) values.
        Returns path_vol (Z,H,W) boolean with 1-pixel line in each z.
        """
        Z, W = y_full.shape
        path_vol = np.zeros((Z, H, W), dtype=bool)
        xs = np.arange(W)
        for z in range(Z):
            yy = np.clip(np.rint(y_full[z]).astype(int), 0, H - 1)
            path_vol[z, yy, xs] = True
        return path_vol

    def segment_from_smoothed_surface(self, vol_shape_zyx: Tuple[int, int, int], kept_zs: List[int], profiles: Dict[int, np.ndarray], *, side: str = "below", sigma_z: float = 1.0, sigma_x: float = 2.0, fill_within_kept_band: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], int, int, np.ndarray, List[int]]:
        """
        Segment volume using a smoothed boundary surface y(z,x).

        If fill_within_kept_band=True:
          - build a contiguous band from min(kept_zs) to max(kept_zs)
          - segment EVERY z in that band (including planes missing from profiles/lines)

        Returns:
          (region, other, area_by_z, vol_region, vol_other, y_smooth_zx, used_zs)
        """
        Z, H, W = vol_shape_zyx
        if not kept_zs:
            raise ValueError("kept_zs is empty")

        y_smooth = self.smooth_surface_yzx(profiles, kept_zs, Z, W, sigma_z=sigma_z, sigma_x=sigma_x)

        if fill_within_kept_band:
            z0, z1 = int(min(kept_zs)), int(max(kept_zs))
            used_zs = list(range(z0, z1 + 1))
        else:
            used_zs = list(map(int, kept_zs))

        region = np.zeros((Z, H, W), dtype=bool)
        other = np.zeros((Z, H, W), dtype=bool)
        area_by_z: Dict[int, int] = {}

        Y = np.arange(H)[:, None]
        for z in used_zs:
            yy = np.clip(np.rint(y_smooth[z]).astype(np.int32), 0, H - 1)[None, :]
            below = Y >= yy

            if side == "below":
                region[z] = below
                other[z] = ~below
                area_by_z[z] = int(below.sum())
            elif side == "above":
                region[z] = ~below
                other[z] = below
                area_by_z[z] = int((~below).sum())
            else:
                raise ValueError("side must be 'below' or 'above'")

        return region, other, area_by_z, int(region.sum()), int(other.sum()), y_smooth, used_zs

    def step(self, axis, xa):
        if axis not in xa.coords or xa.coords[axis].size < 2:
            return None
        return float(xa.coords[axis][1] - xa.coords[axis][0])

    def segmentation_geometry_failure(self, profiles: Dict[int, np.ndarray], kept_zs: List[int], y_max: int, *, frac_bad_thresh: float = 0.5, span_frac_thresh: float = 1 / 3):
        """
        Check for segmentation failure based on vertical span of interface.

        For each z in kept_zs:
          Δy = |y(x=W-1) - y(x=0)|

        Flags failure if:
          fraction of slices with Δy > span_frac_thresh * y_max
          exceeds frac_bad_thresh.

        Returns
        -------
        failed : bool
        diagnostics : dict
        """
        if not kept_zs:
            return True, {"reason": "no_kept_slices"}

        bad = []
        spans = []

        for z in kept_zs:
            yx = profiles[z]
            y0 = float(yx[0])
            y1 = float(yx[-1])
            dy = abs(y1 - y0)
            spans.append(dy)
            bad.append(dy > span_frac_thresh * y_max)

        bad = np.asarray(bad, dtype=bool)
        spans = np.asarray(spans, dtype=float)

        frac_bad = bad.mean()

        diagnostics = {
            "n_kept": len(kept_zs),
            "n_bad": int(bad.sum()),
            "frac_bad": float(frac_bad),
            "span_thresh_px": span_frac_thresh * y_max,
            "median_span_px": float(np.median(spans)),
            "max_span_px": float(np.max(spans)),
            "spans_px": spans,
        }

        failed = frac_bad > frac_bad_thresh
        return failed, diagnostics

    def bulge_failure_over_slices(self, profiles: Dict[int, np.ndarray], zs: List[int], *, frac_thresh: float = 0.5, **kwargs):
        zs_eval = [int(z) for z in zs if int(z) in profiles]
        if not zs_eval:
            return True, {"reason": "no_slices"}

        per_z = {}
        flags = []
        for z in zs_eval:
            has_bulge, d = self.profile_has_bulge_chord(
                profiles[z],
                height_thr_px=80.0,
                area_thr_px=1500.0,
                width_frac_thr=0.12,
            )

            per_z[z] = d
            flags.append(bool(has_bulge))

        flags = np.asarray(flags, dtype=bool)
        frac = float(flags.mean())
        return (frac > frac_thresh), {"n_eval": len(zs_eval), "n_bulge": int(flags.sum()), "frac_bulge": frac, "per_z": per_z}

    def profile_has_bulge_chord(self, yx: np.ndarray, *, height_thr_px: float = 20.0, area_thr_px: float = 4000.0, width_frac_thr: float = 0.12, smooth_win: int = 21) -> Tuple[bool, Dict]:
        """
        Detect a bulge by comparing y(x) to the straight chord between endpoints.
        Works even when the residual MAD is large.
        """
        y = yx.astype(np.float32, copy=False)
        W = int(y.shape[0])

        # light smoothing
        if smooth_win >= 3:
            k = np.ones(int(smooth_win), dtype=np.float32) / float(smooth_win)
            ypad = np.pad(y, (smooth_win // 2, smooth_win // 2), mode="edge")
            y = np.convolve(ypad, k, mode="valid").astype(np.float32, copy=False)

        # chord baseline between endpoints
        x = np.arange(W, dtype=np.float32)
        y0, y1 = float(y[0]), float(y[-1])
        chord = y0 + (y1 - y0) * (x / max(1.0, (W - 1)))

        # positive deviation from chord = bulge (flip sign if you expect downward bulges)
        d = y - chord
        dpos = np.maximum(np.abs(d), 0.0)

        max_dev = float(dpos.max())

        # width at half max (on dpos)
        i0 = int(np.argmax(dpos))
        half = 0.5 * float(dpos[i0])
        L = i0
        while L > 0 and float(dpos[L]) > half:
            L -= 1
        R = i0
        while R < W - 1 and float(dpos[R]) > half:
            R += 1
        width_frac = float((R - L) / max(1, W - 1))

        # "area" of bulge above chord (in pixel-units)
        area = float(dpos.sum())

        has_bulge = (max_dev >= height_thr_px) and (width_frac >= width_frac_thr) and (area >= area_thr_px)

        diag = {
            "has_bulge": bool(has_bulge),
            "max_dev_px": max_dev,
            "width_frac": width_frac,
            "area_px": area,
            "y0": y0,
            "y1": y1,
        }
        return has_bulge, diag

    def measure_volumes_and_interface_um(self, maternal: np.ndarray, fetal: np.ndarray, x_um: float, z_um: float):
        """
        maternal, fetal: bool (Z,H,W) masks (ideally complements on kept_zs)
        x_um: pixel size in X and Y (um)
        z_um: spacing in Z (um)

        Returns: (maternal_um3, fetal_um3, interface_um2)
        """
        vx = float(x_um)
        vy = float(x_um)
        vz = float(z_um)

        voxel_vol = vx * vy * vz

        maternal_um3 = float(maternal.sum()) * voxel_vol
        fetal_um3 = float(fetal.sum()) * voxel_vol

        # Count maternal<->fetal adjacencies across the 3 axes (6-neighbour interface)
        # x-faces have area vy*vz, y-faces vx*vz, z-faces vx*vy
        adj_x = (maternal[:, :, :-1] & fetal[:, :, 1:]) | (fetal[:, :, :-1] & maternal[:, :, 1:])
        adj_y = (maternal[:, :-1, :] & fetal[:, 1:, :]) | (fetal[:, :-1, :] & maternal[:, 1:, :])
        adj_z = (maternal[:-1, :, :] & fetal[1:, :, :]) | (fetal[:-1, :, :] & maternal[1:, :, :])

        interface_um2 = (float(adj_x.sum()) * (vy * vz) + float(adj_y.sum()) * (vx * vz) + float(adj_z.sum()) * (vx * vy))

        return maternal_um3, fetal_um3, interface_um2

    def permeability_calculation(self, final_fetal_intensity, initial_fetal_intensity, initial_maternal_intensity, time_separation_s, bleaching_coefficient, fetal_um3, interface_um2):
        p_um_per_s = (1 / time_separation_s) * (fetal_um3 / interface_um2) * (
            ((bleaching_coefficient * final_fetal_intensity) - initial_fetal_intensity)
            / ((initial_maternal_intensity) - (initial_fetal_intensity))
        )
        p_cm_per_s = p_um_per_s * 0.0001
        return p_cm_per_s


class PlacentaPerfusionApp:
    def __init__(self):
        self.viewer = napari.Viewer()

        # In a terminal script we want the Python process to exit when the napari window closes.
        # In a notebook we typically *don't* want closing napari to tear down the kernel.
        qt_app = QApplication.instance()
        if qt_app is not None:
            qt_app.setQuitOnLastWindowClosed(not _running_in_notebook())

        self.processor = PlacentaPerfusionProcessor(params=RidgeParams(ratio_cutoff=0.9))

        self._selected_lif_path: Optional[Path] = None
        self._image_choice_map: Dict[str, int] = {}
        self._image_meta_map: Dict[str, Dict[str, int]] = {}

        self._results: List[dict] = []
        self._results_df: Optional[pd.DataFrame] = None
        self._last_segmentation: Optional[dict] = None

        self.images_output = TextEdit(value="")
        try:
            self.images_output.native.setReadOnly(True)
        except Exception:
            pass
        self.images_output.min_height = 120
        self.images_output.max_height = 300

        self.list_images = magicgui(
            self._list_images,
            lif_path={"label": "Select .lif", "mode": "r", "filter": "*.lif"},
            call_button="Load images",
        )
        self.segment_and_view = magicgui(
            self._segment_and_view,
            image_choice={"label": "Image", "choices": ["(load images)"], "widget_type": "ComboBox"},
            dextran_channel={"label": "Dextran channel", "choices": ["0"], "widget_type": "ComboBox"},
            frame_interval_s={"label": "Time separation (s)", "min": 1, "max": 1000000, "step": 1, "value": 180, "widget_type": "SpinBox"},
            clear_layers={"label": "Clear viewer first"},
            call_button="Segment + View",
        )
        self.override_z_range = magicgui(
            self._rerun_with_z_range,
            min_z={"label": "Min z", "min": 0, "max": 0, "value": 0},
            max_z={"label": "Max z", "min": 0, "max": 0, "value": 0},
            call_button="Run again with chosen z range",
        )
        self.override_z_range.call_button.enabled = False
        self.save_results = magicgui(self._save_results, call_button="Save results")
        self.run_all_images = magicgui(self._segment_all_images, call_button="Run all images to CSV")

        self.segment_and_view.image_choice.changed.connect(self._update_channel_choices)
        self.segment_and_view.image_choice.changed.connect(self._update_segment_button)
        self.segment_and_view.dextran_channel.changed.connect(self._update_segment_button)
        self.segment_and_view.call_button.enabled = False
        self._update_channel_choices()
        self._update_segment_button()

        list_panel = Container(widgets=[self.list_images, self.images_output])
        self.viewer.window.add_dock_widget(list_panel, area="right")
        self.viewer.window.add_dock_widget(self.segment_and_view, area="right")
        self.viewer.window.add_dock_widget(self.override_z_range, area="right")
        self.viewer.window.add_dock_widget(self.save_results, area="right")
        self.viewer.window.add_dock_widget(self.run_all_images, area="right")

    def _append_log(self, message: str):
        self.images_output.value = (
            self.images_output.value.rstrip() + "\n" + message if self.images_output.value else message
        )

    def _show_paths_layer(self, lines: Dict[int, np.ndarray], vol_shape: Tuple[int, int, int]):
        if not lines:
            return
        Z, H, W = vol_shape
        path_vol = np.zeros((Z, H, W), dtype=bool)
        for z, pm in lines.items():
            zi = int(z)
            if 0 <= zi < Z:
                path_vol[zi] = pm.astype(bool)
        if "paths_all" in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers["paths_all"])
        self.viewer.add_labels(
            path_vol,
            name="paths_all",
            colormap=self.processor.label_colormap("gold", alpha=0.6),
        )

    def _update_override_controls(self, found_zs: List[int]):
        if not found_zs:
            self.override_z_range.call_button.enabled = False
            return
        z_min = int(min(found_zs))
        z_max = int(max(found_zs))
        self.override_z_range.min_z.min = z_min
        self.override_z_range.min_z.max = z_max
        self.override_z_range.max_z.min = z_min
        self.override_z_range.max_z.max = z_max
        if self.override_z_range.min_z.value < z_min or self.override_z_range.min_z.value > z_max:
            self.override_z_range.min_z.value = z_min
        if self.override_z_range.max_z.value < z_min or self.override_z_range.max_z.value > z_max:
            self.override_z_range.max_z.value = z_max
        self.override_z_range.call_button.enabled = True

    def _calculate_from_kept(
        self,
        *,
        kept_zs: List[int],
        profiles: Dict[int, np.ndarray],
        dextran_t0: np.ndarray,
        dextran_t1_2: np.ndarray,
        dextran_tfinal: np.ndarray,
        number_timepoints: int,
        x_um: float,
        z_um: float,
        lif_name: str,
        lif_path: str,
        image_index: int,
        image_name_full: str,
        found_zs: List[int],
        flag: str,
        frame_interval_s: int = 180,
        override_range: Optional[Tuple[int, int]] = None,
    ):
        failed_hump, diag = self.processor.bulge_failure_over_slices(
            profiles, kept_zs, frac_thresh=0.1
        )
        hump_zs = sorted(
            int(z)
            for z, d in diag.get("per_z", {}).items()
            if d.get("has_bulge")
        )
        failed_geom, _ = self.processor.segmentation_geometry_failure(
            profiles, kept_zs, y_max=dextran_t0.shape[1]
        )
        (
            maternal_region,
            fetal_region,
            area_by_z,
            maternal_region_volume,
            fetal_region_volume,
            y_smooth_zx,
            used_zs,
        ) = self.processor.segment_from_smoothed_surface(
            dextran_t0.shape,
            kept_zs,
            profiles,
            side="below",
            sigma_z=1.0,
            sigma_x=3.0,
            fill_within_kept_band=True,
        )

        maternal_um3, fetal_um3, interface_um2 = self.processor.measure_volumes_and_interface_um(
            maternal_region, fetal_region, x_um=x_um, z_um=z_um
        )

        t0_maternal_intensity = np.float64(np.sum(dextran_t0[maternal_region == 1]))
        t0_fetal_intensity = np.float64(np.sum(dextran_t0[fetal_region == 1]))
        t1_2_maternal_intensity = np.float64(np.sum(dextran_t1_2[maternal_region == 1]))
        t1_2_fetal_intensity = np.float64(np.sum(dextran_t1_2[fetal_region == 1]))
        tfinal_maternal_intensity = np.float64(np.sum(dextran_tfinal[maternal_region == 1]))
        tfinal_fetal_intensity = np.float64(np.sum(dextran_tfinal[fetal_region == 1]))

        tfinal_index = int(number_timepoints - 1)
        t1_2_index = int(np.ceil(number_timepoints / 2)) if number_timepoints > 3 else None
        frame_interval_s = int(frame_interval_s)
        if frame_interval_s < 1:
            raise ValueError('frame_interval_s must be >= 1')
        t0_tfinal_time_s = frame_interval_s * tfinal_index
        t0_t1_2_time_s = frame_interval_s * t1_2_index if t1_2_index is not None else None
        t1_2_tfinal_time_s = (
            frame_interval_s * (tfinal_index - t1_2_index) if t1_2_index is not None else None
        )
        t0_t1_2_label = str(t0_t1_2_time_s) if t0_t1_2_time_s is not None else "n/a"
        t1_2_tfinal_label = (
            str(t1_2_tfinal_time_s) if t1_2_tfinal_time_s is not None else "n/a"
        )
        time_info = (
            f"[INFO] frame_interval_s={frame_interval_s}; time_separation_s: t0->tfinal={t0_tfinal_time_s}, "
            f"t0->t1/2={t0_t1_2_label}, t1/2->tfinal={t1_2_tfinal_label}."
        )
        self._append_log(time_info)

        bleaching_coefficient_t0_tfinal = t0_maternal_intensity / tfinal_maternal_intensity
        t0_tfinal_p = self.processor.permeability_calculation(
            tfinal_fetal_intensity,
            t0_fetal_intensity,
            t0_maternal_intensity,
            t0_tfinal_time_s,
            bleaching_coefficient_t0_tfinal,
            fetal_um3,
            interface_um2,
        )

        if number_timepoints > 3:
            bleaching_coefficient_t0_t1_2 = t0_maternal_intensity / t1_2_maternal_intensity
            bleaching_coefficient_t1_2_tfinal = t1_2_maternal_intensity / tfinal_maternal_intensity

            t0_t1_2_p = self.processor.permeability_calculation(
                t1_2_fetal_intensity,
                t0_fetal_intensity,
                t0_maternal_intensity,
                t0_t1_2_time_s,
                bleaching_coefficient_t0_t1_2,
                fetal_um3,
                interface_um2,
            )

            t1_2_tfinal_p = self.processor.permeability_calculation(
                tfinal_fetal_intensity,
                t1_2_fetal_intensity,
                t1_2_maternal_intensity,
                t1_2_tfinal_time_s,
                bleaching_coefficient_t1_2_tfinal,
                fetal_um3,
                interface_um2,
            )
        else:
            bleaching_coefficient_t0_t1_2 = np.nan
            bleaching_coefficient_t1_2_tfinal = np.nan
            t0_t1_2_p = np.nan
            t1_2_tfinal_p = np.nan

        debug_info = (
            f"[DEBUG] t0_tfinal_p={t0_tfinal_p}, "
            f"t0_t1_2_p={t0_t1_2_p}, kept_z_min={int(kept_zs[0])}, "
            f"maternal_region_volume_px={int(maternal_region_volume)}"
        )
        self._append_log(debug_info)

        result = {
            "lif_name": lif_name,
            "lif_path": lif_path,
            "image_index": image_index,
            "image_name": image_name_full,
            "maternal_initial_intensity": float(t0_maternal_intensity),
            "fetal_initial_intensity": float(t0_fetal_intensity),
            "maternal_final_intensity": float(tfinal_maternal_intensity),
            "fetal_final_intensity": float(tfinal_fetal_intensity),
            "t0_tfinal_p": float(t0_tfinal_p),
            "t0_t1_2_p": float(t0_t1_2_p),
            "t1_2_tfinal_p": float(t1_2_tfinal_p),
            "t0_t1_2_pct_change": float(100 * (t0_t1_2_p - t0_tfinal_p) / t0_tfinal_p)
            if t0_tfinal_p
            else np.nan,
            "t1_2_tfinal_pct_change": float(100 * (t1_2_tfinal_p - t0_tfinal_p) / t0_tfinal_p)
            if t0_tfinal_p
            else np.nan,
            "frame_interval_s": int(frame_interval_s),
            "bleaching_coefficient_t0_tfinal": float(bleaching_coefficient_t0_tfinal),
            "bleaching_coefficient_t0_t1_2": float(bleaching_coefficient_t0_t1_2),
            "bleaching_coefficient_t1_2_tfinal": float(bleaching_coefficient_t1_2_tfinal),
            "image_shape": dextran_t0.shape,
            "n_planes_found": len(found_zs),
            "n_planes_kept": len(kept_zs),
            "kept_z_min": int(kept_zs[0]),
            "kept_z_max": int(kept_zs[-1]),
            "maternal_region_volume_px": int(maternal_region_volume),
            "fetal_region_volume_px": int(fetal_region_volume),
            "x_um": x_um,
            "z_um": z_um,
            "maternal_um3": float(maternal_um3),
            "fetal_um3": float(fetal_um3),
            "interface_um2": float(interface_um2),
            "failed_geometry": bool(failed_geom),
            "failed_hump": bool(failed_hump),
            "hump_zs": hump_zs,
            "flag": flag,
        }
        if override_range is not None:
            result["override_z_min"] = int(override_range[0])
            result["override_z_max"] = int(override_range[1])
        return maternal_region, fetal_region, result

    def _rerun_with_z_range(self, min_z: int = 0, max_z: int = 0):
        if self._last_segmentation is None:
            self._append_log("[WARN] No segmentation available to override yet.")
            show_warning("No segmentation available to override yet.")
            return None
        if min_z > max_z:
            min_z, max_z = max_z, min_z
        data = self._last_segmentation
        found_zs = data["found_zs"]
        profiles = data["profiles"]
        kept_zs = [z for z in found_zs if min_z <= z <= max_z and z in profiles]
        if not kept_zs:
            self._append_log("[WARN] Selected z range contains no valid paths.")
            show_warning("Selected z range contains no valid paths.")
            return None
        kept_zs = sorted(set(int(z) for z in kept_zs))
        self._append_log(f"[INFO] Recomputing with z range {min_z}..{max_z}.")
        maternal_region, fetal_region, result = self._calculate_from_kept(
            kept_zs=kept_zs,
            profiles=profiles,
            dextran_t0=data["dextran_t0"],
            dextran_t1_2=data["dextran_t1_2"],
            dextran_tfinal=data["dextran_tfinal"],
            number_timepoints=data["number_timepoints"],
            x_um=data["x_um"],
            z_um=data["z_um"],
            lif_name=data["lif_name"],
            lif_path=data["lif_path"],
            image_index=data["image_index"],
            image_name_full=data["image_name_full"],
            found_zs=found_zs,
            flag=data["flag"],
            frame_interval_s=int(data.get("frame_interval_s", 180)),
            override_range=(min_z, max_z),
        )

        if "maternal" in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers["maternal"])
        if "fetal" in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers["fetal"])
        self.viewer.add_labels(
            maternal_region,
            name="maternal",
            colormap=self.processor.label_colormap("dodgerblue", alpha=0.2),
        )
        self.viewer.add_labels(
            fetal_region,
            name="fetal",
            colormap=self.processor.label_colormap("red", alpha=0.2),
        )

        self._results.append(result)
        self._results_df = pd.DataFrame(self._results)
        display(self._results_df)
        return result

    @staticmethod
    def _infer_tc(shape) -> Tuple[int, int]:
        if not shape:
            return 1, 1
        if len(shape) >= 5:
            return int(shape[0]), int(shape[1])
        if len(shape) == 4:
            return 1, int(shape[0])
        if len(shape) == 3:
            return 1, 1
        return 1, 1

    @staticmethod
    def _read_dextran_t0(lif_img, dextran_channel: int) -> np.ndarray:
        img = lif_img.asarray()
        if img.ndim == 5:
            number_timepoints, c_dim, _, _, _ = img.shape
            if dextran_channel < 0 or dextran_channel >= c_dim:
                raise ValueError(f"dextran_channel out of range (0..{c_dim - 1})")
            if number_timepoints < 1:
                raise ValueError("no timepoints available")
            return img[0, dextran_channel, :, :, :]
        if img.ndim == 4:
            c_dim = img.shape[0]
            if dextran_channel < 0 or dextran_channel >= c_dim:
                raise ValueError(f"dextran_channel out of range (0..{c_dim - 1})")
            return img[dextran_channel, :, :, :]
        if img.ndim == 3:
            if dextran_channel != 0:
                raise ValueError("only one channel available")
            return img
        raise ValueError(f"unsupported image shape: {img.shape}")

    def _segment_single_image_from_lif(
        self, lif, lif_path: Path, image_index: int, dextran_channel: int, frame_interval_s: int
    ):
        if image_index < 0 or image_index >= len(lif.images):
            raise ValueError("image index out of range")
        lif_img = lif.images[image_index]
        xa = lif_img.asxarray()
        x_um = self.processor.step("X", xa) * 1e6 if self.processor.step("X", xa) is not None else None
        z_um = self.processor.step("Z", xa) * 1e6 if self.processor.step("Z", xa) is not None else None
        image_name_full = "".join(getattr(lif_img, "path", ())) or f"Image {image_index}"

        img = lif_img.asarray()
        flag = "None"
        if img.ndim == 5:
            number_timepoints, c_dim, _, _, _ = img.shape
            if dextran_channel < 0 or dextran_channel >= c_dim:
                raise ValueError(f"dextran_channel out of range (0..{c_dim - 1})")
            t0_index = 0
            t1_2_index = int(np.ceil(number_timepoints / 2)) if number_timepoints > 3 else None
            tfinal_index = number_timepoints - 1
            dextran_t0 = img[t0_index, dextran_channel, :, :, :]
            dextran_t1_2 = (
                img[t1_2_index, dextran_channel, :, :, :]
                if t1_2_index is not None
                else np.zeros_like(dextran_t0)
            )
            dextran_tfinal = (
                img[tfinal_index, dextran_channel, :, :, :]
                if number_timepoints > 2
                else np.zeros_like(dextran_t0)
            )
            t1_2_label = str(t1_2_index) if t1_2_index is not None else "n/a"
            info = f"[INFO] timepoints={number_timepoints}, t0={t0_index}, tfinal={tfinal_index}, t1/2={t1_2_label}."
            self._append_log(info)
        else:
            raise ValueError("Expected 5D (t,c,z,y,x) image data for timepoint selection")

        lines, profiles, found_zs = self.processor.collect_lines_over_z(dextran_t0)
        kept_zs, _ = self.processor.keep_similar_high_z(
            lines, profiles, found_zs, missing_run=2, consec_bad=2, core_frac=0.6, diff_thresh_px=20
        )
        if not kept_zs and found_zs:
            kept_zs = sorted(set(int(z) for z in found_zs))
            self._append_log(
                f"[WARN] No kept slices after filtering; using all found slices (n={len(kept_zs)})."
            )

        self._last_segmentation = {
            "lines": lines,
            "profiles": profiles,
            "found_zs": found_zs,
            "dextran_t0": dextran_t0,
            "dextran_t1_2": dextran_t1_2,
            "dextran_tfinal": dextran_tfinal,
            "number_timepoints": number_timepoints,
            "frame_interval_s": int(frame_interval_s),
            "x_um": x_um,
            "z_um": z_um,
            "lif_name": lif.name,
            "lif_path": str(lif_path),
            "image_index": image_index,
            "image_name_full": image_name_full,
            "flag": flag,
        }
        if not kept_zs:
            result = {
                "lif_name": lif.name,
                "lif_path": str(lif_path),
                "image_index": image_index,
                "image_name": image_name_full,
                "image_shape": getattr(lif_img, "shape", None),
                "error": "no_kept_slices",
                "hump_zs": [],
                "flag": flag,
            }
            return dextran_t0, None, None, result

        maternal_region, fetal_region, result = self._calculate_from_kept(
            kept_zs=kept_zs,
            profiles=profiles,
            dextran_t0=dextran_t0,
            dextran_t1_2=dextran_t1_2,
            dextran_tfinal=dextran_tfinal,
            number_timepoints=number_timepoints,
            x_um=x_um,
            z_um=z_um,
            lif_name=lif.name,
            lif_path=str(lif_path),
            image_index=image_index,
            image_name_full=image_name_full,
            found_zs=found_zs,
            flag=flag,
            frame_interval_s=int(frame_interval_s),
        )
        return dextran_t0, maternal_region, fetal_region, result

    def _segment_single_image(
        self, lif_path: Path, image_index: int, dextran_channel: int, frame_interval_s: int
    ):
        with LifFile(lif_path) as lif:
            return self._segment_single_image_from_lif(
                lif, lif_path, image_index, dextran_channel, frame_interval_s
            )

    def _list_images(self, lif_path: Path = Path()):
        if not lif_path or not lif_path.exists():
            self.images_output.value = "[WARN] Please select a .lif file."
            return

        self._selected_lif_path = lif_path

        choices = []
        choice_map: Dict[str, int] = {}
        meta_map: Dict[str, Dict[str, int]] = {}
        try:
            with LifFile(lif_path) as lif:
                for i, img in enumerate(lif.images):
                    name = "".join(getattr(img, "path", ())) or f"Image {i}"
                    label = f"{i}: {name}"
                    shape = getattr(img, "shape", None)
                    t_count, c_count = self._infer_tc(shape)
                    choices.append(label)
                    choice_map[label] = i
                    meta_map[label] = {"index": i, "channels": c_count, "times": t_count}
        except Exception as e:
            self.images_output.value = (
                self.images_output.value + f"\n[ERROR] Failed to read .lif: {type(e).__name__}: {e}"
            )
            return

        if not choices:
            self.images_output.value = self.images_output.value + "\n[WARN] No images found."
            self._image_choice_map = {}
            self._image_meta_map = {}
            self.segment_and_view.image_choice.choices = ["(load images)"]
            self.segment_and_view.image_choice.value = "(load images)"
            self._update_channel_choices()
            self._update_segment_button()
            return

        self._image_choice_map = choice_map
        self._image_meta_map = meta_map
        self.segment_and_view.image_choice.choices = choices
        self.segment_and_view.image_choice.value = choices[0]
        self._update_channel_choices()
        self.images_output.value = (
            self.images_output.value
            + f"\n[OK] Found {len(choices)} images. Select one from the dropdown."
        )
        self._update_segment_button()

    def _segment_and_view(
        self,
        image_choice: str = "(load images)",
        dextran_channel: str = "0",
        frame_interval_s: int = 180,
        clear_layers: bool = True,
    ):
        try:
            frame_interval_s = int(frame_interval_s)
            if frame_interval_s < 1:
                raise ValueError()
        except Exception:
            frame_interval_s = 180
            self._append_log("[WARN] Invalid time separation; using default 180s.")
            show_warning("Invalid time separation; using default 180s.")
        if self._selected_lif_path is None or not self._selected_lif_path.exists():
            self.images_output.value = (
                self.images_output.value.rstrip()
                + "\n[WARN] Select a .lif file and click 'Load images' first."
                if self.images_output.value
                else "[WARN] Select a .lif file and click 'Load images' first."
            )
            return None

        if not self._image_choice_map:
            self.images_output.value = (
                self.images_output.value.rstrip()
                + "\n[WARN] Click 'Load images' to populate the dropdown."
                if self.images_output.value
                else "[WARN] Click 'Load images' to populate the dropdown."
            )
            return None

        image_index = self._image_choice_map.get(image_choice)
        if image_index is None:
            self.images_output.value = (
                self.images_output.value.rstrip()
                + "\n[WARN] Please select an image from the dropdown."
                if self.images_output.value
                else "[WARN] Please select an image from the dropdown."
            )
            return None

        try:
            dextran_channel_index = int(dextran_channel)
        except Exception:
            self.images_output.value = (
                self.images_output.value.rstrip() + "\n[WARN] Invalid dextran channel selection."
                if self.images_output.value
                else "[WARN] Invalid dextran channel selection."
            )
            show_warning("Select a valid dextran channel.")
            return None

        self.images_output.value = (
            self.images_output.value.rstrip()
            + f"\n[INFO] Segmenting image_index={image_index} with dextran_channel={dextran_channel_index}...\n"
            if self.images_output.value
            else f"[INFO] Segmenting image_index={image_index} with dextran_channel={dextran_channel_index}...\n"
        )

        try:
            dextran_t0, maternal_region, fetal_region, result = self._segment_single_image(
                self._selected_lif_path,
                image_index,
                dextran_channel_index,
                frame_interval_s,
            )
        except Exception as e:
            self.images_output.value = self.images_output.value.rstrip() + f"\nSegmentation failed: {type(e).__name__}: {e}"
            show_error(f"Segmentation failed: {type(e).__name__}: {e}")
            return None

        if clear_layers:
            self.viewer.layers.clear()

        self.viewer.add_image(dextran_t0, name="dextran_t0")
        if maternal_region is not None and fetal_region is not None:
            self.viewer.add_labels(
                maternal_region,
                name="maternal",
                colormap=self.processor.label_colormap("dodgerblue", alpha=0.2),
            )
            self.viewer.add_labels(
                fetal_region,
                name="fetal",
                colormap=self.processor.label_colormap("red", alpha=0.2),
            )

        if self._last_segmentation is not None:
            self._show_paths_layer(
                self._last_segmentation["lines"],
                self._last_segmentation["dextran_t0"].shape,
            )
            self._update_override_controls(self._last_segmentation["found_zs"])

        self.images_output.value = self.images_output.value.rstrip() + "\n Segmentation complete :)"
        self._results.append(result)
        self._results_df = pd.DataFrame(self._results)
        display(self._results_df)
        return result

    def _segment_all_images(self):
        if self._selected_lif_path is None or not self._selected_lif_path.exists():
            self.images_output.value = (
                self.images_output.value.rstrip()
                + "\n[WARN] Select a .lif file and click 'Load images' first."
                if self.images_output.value
                else "[WARN] Select a .lif file and click 'Load images' first."
            )
            return None

        if not self._image_choice_map:
            self.images_output.value = (
                self.images_output.value.rstrip()
                + "\n[WARN] Click 'Load images' to populate the dropdown."
                if self.images_output.value
                else "[WARN] Click 'Load images' to populate the dropdown."
            )
            return None

        try:
            dextran_channel_index = int(self.segment_and_view.dextran_channel.value)
        except Exception:
            self.images_output.value = (
                self.images_output.value.rstrip() + "\n[WARN] Invalid dextran channel selection."
                if self.images_output.value
                else "[WARN] Invalid dextran channel selection."
            )
            show_warning("Select a valid dextran channel.")
            return None
        try:
            frame_interval_s = int(self.segment_and_view.frame_interval_s.value)
            if frame_interval_s < 1:
                raise ValueError()
        except Exception:
            frame_interval_s = 180
            self._append_log("[WARN] Invalid time separation; using default 180s.")

        default_dir = (
            str(self._selected_lif_path.parent) if self._selected_lif_path else str(Path.cwd())
        )
        filename, _ = QFileDialog.getSaveFileName(
            None,
            "Save results CSV",
            default_dir,
            "CSV files (*.csv)",
        )
        if not filename:
            self.images_output.value = self.images_output.value.rstrip() + "\n[INFO] Save canceled."
            return None

        if not filename.lower().endswith(".csv"):
            filename = f"{filename}.csv"

        image_indices = sorted(set(self._image_choice_map.values()))
        self._append_log(f"[INFO] Running all images (n={len(image_indices)}) to CSV: {filename}")
        batch_results: List[dict] = []

        for image_index in image_indices:
            image_name = f"Image {image_index}"
            try:
                _, _, _, result = self._segment_single_image(
                    self._selected_lif_path,
                    image_index,
                    dextran_channel_index,
                    frame_interval_s,
                )
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                self._append_log(f"[ERROR] image_index={image_index} failed: {msg}")
                result = {
                    "lif_name": self._selected_lif_path.name,
                    "lif_path": str(self._selected_lif_path),
                    "image_index": image_index,
                    "image_name": image_name,
                    "error": msg,
                }
            batch_results.append(result)

        if not batch_results:
            self._append_log("[WARN] No results collected.")
            return None

        self._results = batch_results
        self._results_df = pd.DataFrame(batch_results)
        self._results_df.to_csv(filename, index=False)
        self._append_log(f"[OK] Saved results to: {filename}")
        display(self._results_df)
        return filename

    def _save_results(self):
        if self._results_df is None or self._results_df.empty:
            self.images_output.value = self.images_output.value.rstrip() + "\n[WARN] No results to save yet."
            show_warning("No results to save yet.")
            return None

        default_dir = (
            str(self._selected_lif_path.parent) if self._selected_lif_path else str(Path.cwd())
        )
        filename, _ = QFileDialog.getSaveFileName(
            None,
            "Save results CSV",
            default_dir,
            "CSV files (*.csv)",
        )
        if not filename:
            self.images_output.value = self.images_output.value.rstrip() + "\n[INFO] Save canceled."
            return None

        if not filename.lower().endswith(".csv"):
            filename = f"{filename}.csv"

        self._results_df.to_csv(filename, index=False)
        self.images_output.value = self.images_output.value.rstrip() + f"\n[OK] Saved results to: {filename}"
        return filename

    def _update_channel_choices(self, *_):
        label = self.segment_and_view.image_choice.value
        meta = self._image_meta_map.get(label)
        if not meta:
            self.segment_and_view.dextran_channel.choices = ["0"]
            self.segment_and_view.dextran_channel.value = "0"
            return
        c_count = max(1, int(meta.get("channels", 1)))
        channel_choices = [str(i) for i in range(c_count)]
        self.segment_and_view.dextran_channel.choices = channel_choices
        if self.segment_and_view.dextran_channel.value not in channel_choices:
            self.segment_and_view.dextran_channel.value = channel_choices[0]

    def _update_segment_button(self, *_):
        image_value = self.segment_and_view.image_choice.value
        image_ok = bool(self._image_choice_map) and image_value in self._image_choice_map
        channel_value = self.segment_and_view.dextran_channel.value
        channel_ok = channel_value in getattr(
            self.segment_and_view.dextran_channel, "choices", [channel_value]
        )
        self.segment_and_view.call_button.enabled = image_ok and channel_ok


app = PlacentaPerfusionApp()




def main():
    """Launch the napari GUI."""
    _app = PlacentaPerfusionApp()
    import napari as _napari
    _napari.run()
    raise SystemExit(0)


if __name__ == '__main__':
    main()
