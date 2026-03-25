# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import re
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from earth2studio.statistics import mae, mean, rmse

LOGGER = logging.getLogger(__name__)

G0 = 9.81
TARGET_AGL_M = 80.0
LOG_FACTOR_80_FROM_10 = np.log(80.0) / np.log(10.0)


def parse_level(name: str, prefix: str) -> int | None:
    """Parse '<prefix><N>hl' variable names and return integer level N."""
    m = re.fullmatch(rf"{re.escape(prefix)}(\d+)hl", name)
    return int(m.group(1)) if m else None


def collect_hl_names(ds: xr.Dataset, prefix: str) -> tuple[list[str], list[int]]:
    """Collect and sort hybrid-level variable names for a given prefix."""
    pairs: list[tuple[int, str]] = []
    for name in ds.data_vars:
        level = parse_level(str(name), prefix)
        if level is not None:
            pairs.append((level, str(name)))
    pairs.sort(key=lambda x: x[0])
    return [n for _, n in pairs], [k for k, _ in pairs]


def interp_uv_to_height(
    u: np.ndarray,
    v: np.ndarray,
    h: np.ndarray,
    target: float,
    lev_axis: int,
    return_level_indices: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearly interpolate U/V winds from hybrid levels to a target AGL height."""
    if not (u.shape == v.shape == h.shape):
        raise ValueError(
            f"u/v/h shapes must match; got {u.shape}, {v.shape}, {h.shape}"
        )
    nlev = h.shape[lev_axis]
    if nlev < 2:
        raise ValueError("Need at least two hybrid levels for interpolation.")

    k = np.sum(h <= target, axis=lev_axis) - 1
    k = np.clip(k, 0, nlev - 2).astype(np.int64)
    k_exp = np.expand_dims(k, axis=lev_axis)

    h0 = np.take_along_axis(h, k_exp, axis=lev_axis).squeeze(axis=lev_axis)
    h1 = np.take_along_axis(h, k_exp + 1, axis=lev_axis).squeeze(axis=lev_axis)
    u0 = np.take_along_axis(u, k_exp, axis=lev_axis).squeeze(axis=lev_axis)
    u1 = np.take_along_axis(u, k_exp + 1, axis=lev_axis).squeeze(axis=lev_axis)
    v0 = np.take_along_axis(v, k_exp, axis=lev_axis).squeeze(axis=lev_axis)
    v1 = np.take_along_axis(v, k_exp + 1, axis=lev_axis).squeeze(axis=lev_axis)

    den = np.where(np.abs(h1 - h0) > 1e-12, h1 - h0, 1e-12)
    w = np.clip((target - h0) / den, 0.0, 1.0)
    u_t = u0 + w * (u1 - u0)
    v_t = v0 + w * (v1 - v0)
    if return_level_indices:
        return u_t, v_t, k
    return u_t, v_t


def metric_dict(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Compute bias, MAE, RMSE, and correlation on valid paired points."""
    valid = np.isfinite(pred) & np.isfinite(truth)
    n = int(np.sum(valid))
    if n == 0:
        return {"n": 0, "bias": np.nan, "mae": np.nan, "rmse": np.nan, "corr": np.nan}
    p = torch.as_tensor(pred[valid], dtype=torch.float32)
    t = torch.as_tensor(truth[valid], dtype=torch.float32)
    coords = OrderedDict({"sample": np.arange(n, dtype=np.int64)})
    bias_t, _ = mean(["sample"])(p - t, coords)
    mae_t, _ = mae(["sample"])(p, coords, t, coords)
    rmse_t, _ = rmse(["sample"])(p, coords, t, coords)
    corr = np.corrcoef(p.cpu().numpy(), t.cpu().numpy())[0, 1] if n > 1 else np.nan
    return {
        "n": n,
        "bias": float(bias_t.item()),
        "mae": float(mae_t.item()),
        "rmse": float(rmse_t.item()),
        "corr": float(corr),
    }


def fit_affine_per_pixel(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-pixel affine mapping y ~= a*x + b over sample axis."""
    valid = np.isfinite(x) & np.isfinite(y)
    n = np.sum(valid, axis=0).astype(np.float64)
    sx = np.sum(np.where(valid, x, 0.0), axis=0)
    sy = np.sum(np.where(valid, y, 0.0), axis=0)
    sxx = np.sum(np.where(valid, x * x, 0.0), axis=0)
    sxy = np.sum(np.where(valid, x * y, 0.0), axis=0)
    denom = sxx - (sx * sx) / np.maximum(n, 1.0)
    num = sxy - (sx * sy) / np.maximum(n, 1.0)
    a = np.where((n >= 2) & (np.abs(denom) > 1e-12), num / denom, 1.0)
    b = np.where(n > 0, sy / np.maximum(n, 1.0) - a * sx / np.maximum(n, 1.0), 0.0)
    return a.astype(np.float32), b.astype(np.float32)


def build_wind_feature_tensor(ds: xr.Dataset, out_dims: tuple[str, ...]) -> np.ndarray:
    """
    Build predictor tensor from all available wind fields.

    Features include:
    - bias term (constant 1)
    - u10m, v10m
    - all u*hl levels
    - all v*hl levels

    Returns array with shape [sample, feature, y, x], where sample is the flattened
    product of non-spatial dims in out_dims.
    """
    u_names, _ = collect_hl_names(ds, "u")
    v_names, _ = collect_hl_names(ds, "v")
    lead_dims = [d for d in out_dims if d not in ("hrrr_y", "hrrr_x")]
    feature_dims = tuple(lead_dims + ["hrrr_y", "hrrr_x"])

    feats: list[np.ndarray] = []
    u10 = ds["u10m"].transpose(*feature_dims).values
    v10 = ds["v10m"].transpose(*feature_dims).values
    ones = np.ones_like(u10)
    feats.extend([ones, u10, v10])

    feats.extend([ds[name].transpose(*feature_dims).values for name in u_names])
    feats.extend([ds[name].transpose(*feature_dims).values for name in v_names])

    # [feature, *lead_dims, y, x] -> [sample, feature, y, x]
    X = np.stack(feats, axis=0)
    nfeat = X.shape[0]
    ny, nx = X.shape[-2], X.shape[-1]
    X = np.moveaxis(X, 0, -3)  # [*lead_dims, feature, y, x]
    X = X.reshape(-1, nfeat, ny, nx)
    return X.astype(np.float32)


def fit_lr_global(X: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """
    Fit one global linear regression y = sum_f beta_f * X_f.

    X: [sample, feature, y, x]
    y: [sample, y, x]
    Returns beta: [feature]
    """
    if X.ndim != 4 or y.ndim != 3:
        raise ValueError(f"Expected X[s,f,y,x], y[s,y,x], got {X.shape}, {y.shape}")
    if X.shape[0] != y.shape[0] or X.shape[-2:] != y.shape[-2:]:
        raise ValueError("X/y shape mismatch for LR fit.")

    _, f, _, _ = X.shape
    X2 = np.moveaxis(X, 1, -1).reshape(-1, f)  # [n, f]
    y2 = y.reshape(-1)  # [n]
    valid = np.isfinite(y2)
    valid &= np.all(np.isfinite(X2), axis=1)
    if not np.any(valid):
        return np.zeros((f,), dtype=np.float32)

    Xv = X2[valid].astype(np.float32, copy=False)
    yv = y2[valid].astype(np.float32, copy=False)

    # Prefer GPU when available; fallback to NumPy on failure.
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        try:
            x_t = torch.as_tensor(Xv, dtype=torch.float32, device="cuda")
            y_t = torch.as_tensor(yv, dtype=torch.float32, device="cuda")
            xtx = x_t.transpose(0, 1) @ x_t
            xty = x_t.transpose(0, 1) @ y_t
            xtx = xtx + ridge * torch.eye(f, dtype=torch.float32, device="cuda")
            beta = torch.linalg.solve(xtx, xty)
            return beta.detach().cpu().numpy().astype(np.float32)
        except Exception:
            # Fall back to CPU path if GPU solve fails.
            LOGGER.debug("GPU global LR fit failed; using CPU path", exc_info=True)

    # CPU NumPy path.
    xtx = Xv.T @ Xv
    xty = Xv.T @ yv
    xtx = xtx + ridge * np.eye(f, dtype=np.float32)
    beta = np.linalg.solve(xtx, xty)
    return beta.astype(np.float32)


def apply_lr_global(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Apply global linear regression weights.

    X: [sample, feature, y, x]
    beta: [feature]
    Returns yhat: [sample, y, x]
    """
    if beta.ndim != 1 or X.shape[1] != beta.shape[0]:
        raise ValueError(f"Shape mismatch X={X.shape}, beta={beta.shape}")
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        try:
            x_t = torch.as_tensor(X, dtype=torch.float32, device="cuda")
            b_t = torch.as_tensor(beta, dtype=torch.float32, device="cuda")
            y_t = torch.einsum("sfyx,f->syx", x_t, b_t)
            return y_t.cpu().numpy().astype(np.float32)
        except Exception:
            LOGGER.debug("GPU LR apply failed; using CPU path", exc_info=True)
    yhat = np.einsum("sfyx,f->syx", X, beta, optimize=True)
    return yhat.astype(np.float32)


def fit_scale_uv(
    u10: np.ndarray, v10: np.ndarray, u80_truth: np.ndarray, v80_truth: np.ndarray
) -> np.ndarray:
    """
    Fit one per-pixel scale r so that:
      u80 ~= r*u10 and v80 ~= r*v10
    using least squares over sample axis 0.

    Inputs are [sample, y, x]. Returns r [y, x].
    """
    valid = (
        np.isfinite(u10)
        & np.isfinite(v10)
        & np.isfinite(u80_truth)
        & np.isfinite(v80_truth)
    )
    num = np.sum(np.where(valid, u80_truth * u10 + v80_truth * v10, 0.0), axis=0)
    den = np.sum(np.where(valid, u10 * u10 + v10 * v10, 0.0), axis=0)
    r = np.where(den > 1e-12, num / den, LOG_FACTOR_80_FROM_10)
    r = np.clip(r, 0.3, 5.0)
    r = np.where(np.isfinite(r), r, LOG_FACTOR_80_FROM_10)
    return r.astype(np.float32)


def fit_loglaw_z0(
    u10: np.ndarray,
    v10: np.ndarray,
    u80_truth: np.ndarray,
    v80_truth: np.ndarray,
    z_ref: float = 10.0,
    z_tgt: float = 80.0,
) -> np.ndarray:
    """
    Fit effective per-pixel roughness length z0 (assuming d=0) from calibration ratio.

    Uses speed ratio:
      R = |V80| / |V10|
      R = ln(z_tgt/z0) / ln(z_ref/z0)
      z0 = (z_tgt / z_ref^R)^(1/(1-R))

    Inputs are [sample, y, x]. Returns z0 [y, x].
    """
    sp10 = np.hypot(u10, v10)
    sp80 = np.hypot(u80_truth, v80_truth)
    valid = np.isfinite(sp10) & np.isfinite(sp80) & (sp10 > 1e-3)
    ratio = np.where(valid, sp80 / sp10, np.nan)

    # Robust center ratio per pixel over calibration samples.
    R = np.nanmedian(ratio, axis=0)
    R = np.where(np.isfinite(R), R, LOG_FACTOR_80_FROM_10)
    R = np.clip(R, 1.0 + 1e-6, 20.0)

    exp = 1.0 / (1.0 - R)
    z0 = (z_tgt / (z_ref**R)) ** exp
    z0 = np.clip(z0, 1e-4, 9.0)
    z0 = np.where(np.isfinite(z0), z0, 0.03)
    return z0.astype(np.float32)


def loglaw_factor_from_z0(
    z0: np.ndarray, z_ref: float = 10.0, z_tgt: float = 80.0
) -> np.ndarray:
    """Compute log-law scale factor from z0 with d=0."""
    z0_safe = np.clip(z0, 1e-4, 9.0)
    num = np.log(z_tgt / z0_safe)
    den = np.log(z_ref / z0_safe)
    factor = np.where(np.abs(den) > 1e-12, num / den, LOG_FACTOR_80_FROM_10)
    factor = np.where(np.isfinite(factor), factor, LOG_FACTOR_80_FROM_10)
    return factor.astype(np.float32)


def fit_ic_affine_correction(
    pred_u: np.ndarray,
    pred_v: np.ndarray,
    truth_u: np.ndarray,
    truth_v: np.ndarray,
    out_dims: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit affine correction from lead-0 predictions to lead-0 truth per pixel."""
    lead_axis = out_dims.index("lead_time")
    x_u0 = np.take(pred_u, 0, axis=lead_axis).reshape(
        -1, pred_u.shape[-2], pred_u.shape[-1]
    )
    x_v0 = np.take(pred_v, 0, axis=lead_axis).reshape(
        -1, pred_v.shape[-2], pred_v.shape[-1]
    )
    y_u0 = np.take(truth_u, 0, axis=lead_axis).reshape(
        -1, truth_u.shape[-2], truth_u.shape[-1]
    )
    y_v0 = np.take(truth_v, 0, axis=lead_axis).reshape(
        -1, truth_v.shape[-2], truth_v.shape[-1]
    )
    a_u, b_u = fit_affine_per_pixel(x_u0, y_u0)
    a_v, b_v = fit_affine_per_pixel(x_v0, y_v0)
    return a_u, b_u, a_v, b_v


def rmse_by_lead(
    pred: np.ndarray, truth: np.ndarray, out_dims: tuple[str, ...]
) -> np.ndarray:
    """Compute RMSE independently for each lead_time index."""
    lead_axis = out_dims.index("lead_time")
    nlead = pred.shape[lead_axis]
    vals = np.full(nlead, np.nan, dtype=np.float64)
    for i in range(nlead):
        p = np.take(pred, i, axis=lead_axis)
        t = np.take(truth, i, axis=lead_axis)
        valid = np.isfinite(p) & np.isfinite(t)
        if np.any(valid):
            vals[i] = float(np.sqrt(np.mean((p[valid] - t[valid]) ** 2)))
    return vals


def print_level_usage(
    k: np.ndarray, level_values: list[int], lead_dim_index: int | None = None
) -> None:
    """Print hybrid-level bracket usage overall and optionally per lead."""
    flat = k.reshape(-1)
    unique, counts = np.unique(flat, return_counts=True)
    total = flat.size
    print("\nHybrid-level usage (overall):")
    for idx, cnt in zip(unique, counts):
        lo = level_values[int(idx)]
        hi = level_values[int(idx) + 1]
        print(f"  k={int(idx):2d} -> [{lo}hl, {hi}hl]: {cnt} ({100.0*cnt/total:.2f}%)")
    if lead_dim_index is not None:
        print("\nHybrid-level usage by lead:")
        for li in range(k.shape[lead_dim_index]):
            kk = np.take(k, li, axis=lead_dim_index).reshape(-1)
            u, c = np.unique(kk, return_counts=True)
            denom = max(1, kk.size)
            text = ", ".join(
                f"[{level_values[int(i)]},{level_values[int(i)+1]}]={100.0*cnt/denom:.1f}%"
                for i, cnt in zip(u, c)
            )
            print(f"  lead[{li}]: {text}")


def plot_three_panel_all_leads(
    pred: xr.DataArray, truth: xr.DataArray, var_name: str, out_path: str
) -> None:
    """Plot prediction, truth, and error maps for every lead and save figure."""
    sel = {}
    if "ensemble" in pred.dims:
        sel["ensemble"] = 0
    if "time" in pred.dims:
        sel["time"] = 0
    p = pred.isel(**sel)
    t = truth.isel(**sel)
    e = p - t
    if "lead_time" not in p.dims:
        p = p.expand_dims({"lead_time": np.array(["single"])})
        t = t.expand_dims({"lead_time": np.array(["single"])})
        e = e.expand_dims({"lead_time": np.array(["single"])})
    lead_values = p["lead_time"].values
    nlead = p.sizes["lead_time"]

    tv, ev = t.values, e.values
    m_truth = np.isfinite(tv)
    if np.any(m_truth):
        vmin = float(np.nanmin(tv[m_truth]))
        vmax = float(np.nanmax(tv[m_truth]))
    else:
        vmin, vmax = -1.0, 1.0
    me = np.isfinite(ev)
    emax = float(np.nanmax(np.abs(ev[me]))) if np.any(me) else 1.0
    emax = max(emax, 1e-6)

    plt.close("all")
    fig, axes = plt.subplots(nlead, 3, figsize=(16, 4.5 * nlead))
    if nlead == 1:
        axes = np.array([axes])
    for i in range(nlead):
        p2, t2, e2 = (
            p.isel(lead_time=i).values,
            t.isel(lead_time=i).values,
            e.isel(lead_time=i).values,
        )
        lbl = str(lead_values[i])
        im0 = axes[i, 0].imshow(
            p2, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax
        )
        im1 = axes[i, 1].imshow(
            t2, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax
        )
        im2 = axes[i, 2].imshow(
            e2, origin="lower", cmap="RdBu_r", vmin=-emax, vmax=emax
        )
        axes[i, 0].set_title(f"{var_name} pred | lead={lbl}")
        axes[i, 1].set_title(f"{var_name} truth | lead={lbl}")
        axes[i, 2].set_title(f"{var_name} err | lead={lbl}")
        for j in range(3):
            axes[i, j].set_xlabel("hrrr_x")
            axes[i, j].set_ylabel("hrrr_y")
        fig.colorbar(im0, ax=axes[i, 0], shrink=0.8)
        fig.colorbar(im1, ax=axes[i, 1], shrink=0.8)
        fig.colorbar(im2, ax=axes[i, 2], shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
