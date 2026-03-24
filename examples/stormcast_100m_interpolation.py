# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%
"""
StormCast Ensemble 80 m Wind Interpolation
===========================================

Run a short StormCast ensemble forecast, then interpolate hybrid-level winds
to 80 m AGL using:

  H_agl = Zhl / 9.81

where ``Zhl`` is the hybrid-level geopotential-like field in m^2 s^-2.
"""
# /// script
# dependencies = [
#   "earth2studio[data,stormcast] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "xarray",
# ]
# ///

from __future__ import annotations

import os
import re
from collections import OrderedDict
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

import earth2studio.run as run
from earth2studio.data import HRRR
from earth2studio.io import ZarrBackend
from earth2studio.models.px import StormCast
from earth2studio.perturbation import Zero
from earth2studio.statistics import mae, mean, rmse

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

os.makedirs("outputs", exist_ok=True)
load_dotenv()

G0 = 9.81
TARGET_AGL_M = 80.0
N_STEPS = 2
ENSEMBLE_SIZE = 2
BATCH_SIZE = 2


def _parse_level(name: str, prefix: str) -> int | None:
    match = re.fullmatch(rf"{re.escape(prefix)}(\d+)hl", name)
    if match is None:
        return None
    return int(match.group(1))


def _collect_hl_names(ds: xr.Dataset, prefix: str) -> tuple[list[str], list[int]]:
    pairs: list[tuple[int, str]] = []
    for name in ds.data_vars:
        level = _parse_level(str(name), prefix)
        if level is not None:
            pairs.append((level, str(name)))
    pairs.sort(key=lambda p: p[0])
    levels = [p[0] for p in pairs]
    names = [p[1] for p in pairs]
    return names, levels


def _interp_uv_to_height(
    u: np.ndarray,
    v: np.ndarray,
    h: np.ndarray,
    target: float,
    lev_axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate u/v from hybrid levels to target height.

    Inputs are shaped [..., nlev, hrrr_y, hrrr_x]. This supports deterministic
    and ensemble outputs.
    """
    if not (u.shape == v.shape == h.shape):
        raise ValueError(
            f"u/v/h shapes must match; got {u.shape}, {v.shape}, {h.shape}"
        )

    nlev = h.shape[lev_axis]
    if nlev < 2:
        raise ValueError("Need at least two hybrid levels for interpolation.")

    # Bracketing level index per pixel/member/time: k <= target < k+1.
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
    return u_t, v_t


def _metric_dict(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Return aggregate metrics using earth2studio statistics classes."""
    valid = np.isfinite(pred) & np.isfinite(truth)
    n = int(np.sum(valid))
    if n == 0:
        return {"n": 0, "bias": np.nan, "mae": np.nan, "rmse": np.nan, "corr": np.nan}

    # Reduce over a single axis so we can safely ignore invalid points.
    p = torch.as_tensor(pred[valid], dtype=torch.float32)
    t = torch.as_tensor(truth[valid], dtype=torch.float32)
    coords = OrderedDict({"sample": np.arange(n, dtype=np.int64)})

    bias_t, _ = mean(["sample"])(p - t, coords)
    mae_t, _ = mae(["sample"])(p, coords, t, coords)
    rmse_t, _ = rmse(["sample"])(p, coords, t, coords)

    # Pearson correlation is not part of the standard e2studio metric classes.
    corr = np.corrcoef(p.cpu().numpy(), t.cpu().numpy())[0, 1] if n > 1 else np.nan
    return {
        "n": n,
        "bias": float(bias_t.item()),
        "mae": float(mae_t.item()),
        "rmse": float(rmse_t.item()),
        "corr": float(corr),
    }


def _plot_error_and_scatter(
    pred: xr.DataArray,
    truth: xr.DataArray,
    var_name: str,
    out_prefix: str,
) -> None:
    """Save prediction, truth, and error maps side-by-side."""
    # Pick a representative slice (first ensemble member if present, first time/lead).
    sel_indexers = {}
    if "ensemble" in pred.dims:
        sel_indexers["ensemble"] = 0
    if "time" in pred.dims:
        sel_indexers["time"] = 0
    if "lead_time" in pred.dims:
        sel_indexers["lead_time"] = 0

    pred2d = pred.isel(**sel_indexers)
    truth2d = truth.isel(**sel_indexers)
    err2d = pred2d - truth2d

    # Use shared range for pred/truth to make comparison fair.
    pred_vals = pred2d.values
    truth_vals = truth2d.values
    finite_pt = np.isfinite(pred_vals) & np.isfinite(truth_vals)
    if np.any(finite_pt):
        vmin = float(
            np.nanmin(np.concatenate([pred_vals[finite_pt], truth_vals[finite_pt]]))
        )
        vmax = float(
            np.nanmax(np.concatenate([pred_vals[finite_pt], truth_vals[finite_pt]]))
        )
    else:
        vmin, vmax = -1.0, 1.0

    # Symmetric color scale for error.
    err_vals = err2d.values
    finite_e = np.isfinite(err_vals)
    if np.any(finite_e):
        emax = float(np.nanmax(np.abs(err_vals[finite_e])))
    else:
        emax = 1.0
    emax = max(emax, 1e-6)

    plt.close("all")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    im0 = axes[0].imshow(
        pred_vals, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax
    )
    axes[0].set_title(f"{var_name} prediction")
    axes[0].set_xlabel("hrrr_x")
    axes[0].set_ylabel("hrrr_y")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(
        truth_vals, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax
    )
    axes[1].set_title(f"{var_name} truth")
    axes[1].set_xlabel("hrrr_x")
    axes[1].set_ylabel("hrrr_y")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(err_vals, origin="lower", cmap="RdBu_r", vmin=-emax, vmax=emax)
    axes[2].set_title(f"{var_name} error (pred - truth)")
    axes[2].set_xlabel("hrrr_x")
    axes[2].set_ylabel("hrrr_y")
    fig.colorbar(im2, ax=axes[2], shrink=0.8)

    fig.tight_layout()
    out_path = f"outputs/{out_prefix}_{var_name}_diagnostics.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out_path}")


# %%
# Run a short StormCast ensemble forecast and save to Zarr.
package = StormCast.load_default_package()
model = StormCast.load_model(package)
data = HRRR()
perturb = Zero()
io = ZarrBackend(
    file_name="outputs/stormcast_raw_ensemble.zarr", backend_kwargs={"overwrite": True}
)

today = datetime.today() - timedelta(days=1)
date = today.isoformat().split("T")[0]
io = run.ensemble(
    [date],
    N_STEPS,
    ENSEMBLE_SIZE,
    model,
    data,
    io,
    perturb,
    batch_size=BATCH_SIZE,
)

print("Raw StormCast output saved:", "outputs/stormcast_raw_ensemble.zarr")
print(f"Ensemble settings: ensemble_size={ENSEMBLE_SIZE}, batch_size={BATCH_SIZE}")

# %%
# Load output and interpolate hybrid-level winds to 80 m AGL.
ds = xr.open_zarr("outputs/stormcast_raw_ensemble.zarr")

z_names, z_levels = _collect_hl_names(ds, "Z")
u_names, u_levels = _collect_hl_names(ds, "u")
v_names, v_levels = _collect_hl_names(ds, "v")

if not z_names or not u_names or not v_names:
    raise RuntimeError("Could not find Z*hl, u*hl, v*hl variables in StormCast output.")
if not (z_levels == u_levels == v_levels):
    raise RuntimeError(
        f"Level mismatch across Z/u/v hybrid fields: {z_levels}, {u_levels}, {v_levels}"
    )

z_da = (
    xr.concat([ds[name] for name in z_names], dim=xr.DataArray(z_levels, dims="level"))
    / G0
)
u_da = xr.concat(
    [ds[name] for name in u_names], dim=xr.DataArray(u_levels, dims="level")
)
v_da = xr.concat(
    [ds[name] for name in v_names], dim=xr.DataArray(v_levels, dims="level")
)

spatial_dims = ("hrrr_y", "hrrr_x")
for dim in spatial_dims:
    if dim not in z_da.dims:
        raise RuntimeError(f"Expected '{dim}' in StormCast output dims={z_da.dims}")

lead_dims = [d for d in z_da.dims if d not in ("level", *spatial_dims)]
ordered_dims = tuple(lead_dims + ["level", *spatial_dims])
z_da = z_da.transpose(*ordered_dims)
u_da = u_da.transpose(*ordered_dims)
v_da = v_da.transpose(*ordered_dims)

lev_axis = len(lead_dims)
u80, v80 = _interp_uv_to_height(
    u_da.values,
    v_da.values,
    z_da.values,
    TARGET_AGL_M,
    lev_axis=lev_axis,
)

out_dims = tuple(lead_dims + list(spatial_dims))
coords = {d: z_da.coords[d].values for d in out_dims}

out = xr.Dataset(
    data_vars={
        "u80m": xr.DataArray(u80, dims=out_dims, coords=coords),
        "v80m": xr.DataArray(v80, dims=out_dims, coords=coords),
    },
    attrs={
        "description": "80 m AGL wind interpolated from StormCast hybrid levels",
        "height_assumption": "H_agl = Zhl / 9.81",
        "target_height_m": TARGET_AGL_M,
        "ensemble_size": ENSEMBLE_SIZE,
        "batch_size": BATCH_SIZE,
        "levels_used": ",".join(str(level) for level in z_levels),
    },
)

out.to_zarr("outputs/stormcast_80m_wind_ensemble.zarr", mode="w")
print("80 m wind output saved:", "outputs/stormcast_80m_wind_ensemble.zarr")
print("Interpolated variables:", list(out.data_vars))

# %%
# Compute errors vs HRRR truth (u10m/v10m and u80m/v80m) at corresponding valid times.
if "time" not in out.coords or "lead_time" not in out.coords:
    raise RuntimeError(
        "Expected 'time' and 'lead_time' coordinates to compute truth errors."
    )

valid_times = out["time"].values[:, None].astype("datetime64[ns]") + out[
    "lead_time"
].values[None, :].astype("timedelta64[ns]")
unique_valid_times = np.unique(valid_times.reshape(-1))

hrrr_truth = HRRR()(unique_valid_times, ["u10m", "v10m", "u80m", "v80m"])
hrrr_truth = hrrr_truth.sel(
    hrrr_y=xr.DataArray(out["hrrr_y"].values, dims="hrrr_y"),
    hrrr_x=xr.DataArray(out["hrrr_x"].values, dims="hrrr_x"),
    method="nearest",
)

truth_times = hrrr_truth["time"].values.astype("datetime64[ns]")
time_to_idx = {int(t.astype("int64")): i for i, t in enumerate(truth_times)}
time_keys = valid_times.astype("datetime64[ns]").astype("int64")

nt = out.sizes["time"]
nl = out.sizes["lead_time"]
ny = out.sizes["hrrr_y"]
nx = out.sizes["hrrr_x"]

u10_idx = int(np.where(hrrr_truth["variable"].values == "u10m")[0][0])
v10_idx = int(np.where(hrrr_truth["variable"].values == "v10m")[0][0])
u80_idx = int(np.where(hrrr_truth["variable"].values == "u80m")[0][0])
v80_idx = int(np.where(hrrr_truth["variable"].values == "v80m")[0][0])
hrrr_vals = hrrr_truth.values  # [time, variable, y, x]

truth_u10 = np.full((nt, nl, ny, nx), np.nan, dtype=np.float32)
truth_v10 = np.full((nt, nl, ny, nx), np.nan, dtype=np.float32)
truth_u80 = np.full((nt, nl, ny, nx), np.nan, dtype=np.float32)
truth_v80 = np.full((nt, nl, ny, nx), np.nan, dtype=np.float32)
for i in range(nt):
    for j in range(nl):
        idx = time_to_idx.get(int(time_keys[i, j]))
        if idx is not None:
            truth_u10[i, j] = hrrr_vals[idx, u10_idx]
            truth_v10[i, j] = hrrr_vals[idx, v10_idx]
            truth_u80[i, j] = hrrr_vals[idx, u80_idx]
            truth_v80[i, j] = hrrr_vals[idx, v80_idx]

truth_coords = {
    "time": out["time"].values,
    "lead_time": out["lead_time"].values,
    "hrrr_y": out["hrrr_y"].values,
    "hrrr_x": out["hrrr_x"].values,
}
out_u10 = ds["u10m"].transpose(*out_dims)
out_v10 = ds["v10m"].transpose(*out_dims)

truth_u10_da = xr.DataArray(
    truth_u10,
    dims=("time", "lead_time", "hrrr_y", "hrrr_x"),
    coords=truth_coords,
).broadcast_like(out_u10)
truth_v10_da = xr.DataArray(
    truth_v10,
    dims=("time", "lead_time", "hrrr_y", "hrrr_x"),
    coords=truth_coords,
).broadcast_like(out_v10)
truth_u80_da = xr.DataArray(
    truth_u80,
    dims=("time", "lead_time", "hrrr_y", "hrrr_x"),
    coords=truth_coords,
).broadcast_like(out["u80m"])
truth_v80_da = xr.DataArray(
    truth_v80,
    dims=("time", "lead_time", "hrrr_y", "hrrr_x"),
    coords=truth_coords,
).broadcast_like(out["v80m"])

err = xr.Dataset(
    data_vars={
        "u10m_truth": truth_u10_da,
        "v10m_truth": truth_v10_da,
        "u10m_error": out_u10 - truth_u10_da,
        "v10m_error": out_v10 - truth_v10_da,
        "u80m_truth": truth_u80_da,
        "v80m_truth": truth_v80_da,
        "u80m_error": out["u80m"] - truth_u80_da,
        "v80m_error": out["v80m"] - truth_v80_da,
    },
    attrs={
        "description": "Errors for StormCast 80 m wind interpolation vs HRRR truth",
        "truth_source": "HRRR u10m/v10m and u80m/v80m",
    },
)
err.to_zarr("outputs/stormcast_80m_wind_errors_ensemble.zarr", mode="w")
print("Error output saved:", "outputs/stormcast_80m_wind_errors_ensemble.zarr")

u10_metrics = _metric_dict(out_u10.values, truth_u10_da.values)
v10_metrics = _metric_dict(out_v10.values, truth_v10_da.values)
u80_metrics = _metric_dict(out["u80m"].values, truth_u80_da.values)
v80_metrics = _metric_dict(out["v80m"].values, truth_v80_da.values)
print("\nOverall metrics vs HRRR truth:")
print("u10m:", u10_metrics)
print("v10m:", v10_metrics)
print("u80m:", u80_metrics)
print("v80m:", v80_metrics)

if "ensemble" in out.dims:
    ens_mean_u10 = out_u10.mean(dim="ensemble").values
    ens_mean_v10 = out_v10.mean(dim="ensemble").values
    ens_mean_u = out["u80m"].mean(dim="ensemble").values
    ens_mean_v = out["v80m"].mean(dim="ensemble").values
    print("\nEnsemble-mean metrics vs HRRR truth:")
    print("u10m:", _metric_dict(ens_mean_u10, truth_u10))
    print("v10m:", _metric_dict(ens_mean_v10, truth_v10))
    print("u80m:", _metric_dict(ens_mean_u, truth_u80))
    print("v80m:", _metric_dict(ens_mean_v, truth_v80))

# %%
# Quick plots for visual diagnostics.
_plot_error_and_scatter(out_u10, truth_u10_da, "u10m", "stormcast")
_plot_error_and_scatter(out_v10, truth_v10_da, "v10m", "stormcast")
_plot_error_and_scatter(out["u80m"], truth_u80_da, "u80m", "stormcast")
_plot_error_and_scatter(out["v80m"], truth_v80_da, "v80m", "stormcast")
