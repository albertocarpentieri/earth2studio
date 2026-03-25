# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
80m Wind Methods Benchmarks
===========================

This script provides two benchmarks:

1) HRRR-only: apply methods to HRRR fields and compare to HRRR u80m/v80m.
2) StormCast: apply methods on top of StormCast forecasts and compare to HRRR u80m/v80m.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

os.environ["EARTH2STUDIO_CACHE"] = "./cache"
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dotenv import load_dotenv

import earth2studio.run as run
from earth2studio.data import GFS_FX, HRRR
from earth2studio.io import ZarrBackend
from earth2studio.models.px import StormCast
from earth2studio.perturbation import Zero

LOGGER = logging.getLogger(__name__)

try:
    from .methods import (
        G0,
        LOG_FACTOR_80_FROM_10,
        TARGET_AGL_M,
        apply_lr_global,
        build_wind_feature_tensor,
        collect_hl_names,
        fit_loglaw_z0,
        fit_lr_global,
        fit_scale_uv,
        interp_uv_to_height,
        loglaw_factor_from_z0,
        metric_dict,
        plot_three_panel_all_leads,
        print_level_usage,
        rmse_by_lead,
    )
except ImportError:
    from methods import (
        G0,
        LOG_FACTOR_80_FROM_10,
        TARGET_AGL_M,
        apply_lr_global,
        build_wind_feature_tensor,
        collect_hl_names,
        fit_loglaw_z0,
        fit_lr_global,
        fit_scale_uv,
        interp_uv_to_height,
        loglaw_factor_from_z0,
        metric_dict,
        plot_three_panel_all_leads,
        print_level_usage,
        rmse_by_lead,
    )

load_dotenv()
os.makedirs("outputs", exist_ok=True)

N_STEPS = 2
ENSEMBLE_SIZE = 2
BATCH_SIZE = 2
CALIB_HOURS = 6
HRRR_FIT_TIME = datetime(2025, 1, 15, 0, 0, 0)
HRRR_TEST_TIME = datetime(2025, 1, 15, 1, 0, 0)


def _close_hrrr_sessions(*sources: object) -> None:
    """Best-effort cleanup of async S3 sessions to avoid warnings."""
    try:
        import s3fs
    except ImportError:
        return
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    for src in sources:
        if src is None:
            continue
        fs = getattr(src, "fs", None)
        if fs is None:
            continue
        # Common path for s3fs-backed datasources.
        if hasattr(fs, "s3"):
            try:
                s3fs.S3FileSystem.close_session(loop, fs.s3)
            except Exception:
                LOGGER.debug("Failed to close s3 session", exc_info=True)
        # Additional best-effort close hooks.
        try:
            fs.close()
        except Exception:
            LOGGER.debug("Failed to close fs handle", exc_info=True)
        try:
            session = getattr(fs, "session", None)
            if session is not None:
                loop.run_until_complete(session.close())
        except Exception:
            LOGGER.debug("Failed to close async session", exc_info=True)


def _build_truth_from_hrrr(
    hrrr_source: HRRR, ds_like: xr.Dataset, valid_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fetch HRRR u80m/v80m truth and align it to ds_like time/lead grid."""
    hrrr_truth = hrrr_source(valid_times, ["u80m", "v80m"])
    hrrr_truth = hrrr_truth.sel(
        hrrr_y=xr.DataArray(ds_like["hrrr_y"].values, dims="hrrr_y"),
        hrrr_x=xr.DataArray(ds_like["hrrr_x"].values, dims="hrrr_x"),
        method="nearest",
    )
    tvals = hrrr_truth["time"].values.astype("datetime64[ns]")
    idx_map = {int(t.astype("int64")): i for i, t in enumerate(tvals)}
    iu = int(np.where(hrrr_truth["variable"].values == "u80m")[0][0])
    iv = int(np.where(hrrr_truth["variable"].values == "v80m")[0][0])
    vals = hrrr_truth.values

    nt = ds_like.sizes["time"]
    nl = ds_like.sizes["lead_time"]
    ny = ds_like.sizes["hrrr_y"]
    nx = ds_like.sizes["hrrr_x"]
    truth_u = np.full((nt, nl, ny, nx), np.nan, dtype=np.float32)
    truth_v = np.full((nt, nl, ny, nx), np.nan, dtype=np.float32)
    keys = (
        (
            ds_like["time"].values[:, None].astype("datetime64[ns]")
            + ds_like["lead_time"].values[None, :].astype("timedelta64[ns]")
        )
        .astype("datetime64[ns]")
        .astype("int64")
    )
    for i in range(nt):
        for j in range(nl):
            k = idx_map.get(int(keys[i, j]))
            if k is not None:
                truth_u[i, j] = vals[k, iu]
                truth_v[i, j] = vals[k, iv]
    return truth_u, truth_v


def _derive_methods(
    ds: xr.Dataset,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], tuple[str, ...], list[int]]:
    """Build baseline 80m methods and return method outputs with target dims."""
    z_names, z_levels = collect_hl_names(ds, "Z")
    u_names, u_levels = collect_hl_names(ds, "u")
    v_names, v_levels = collect_hl_names(ds, "v")
    if (
        not z_names
        or not u_names
        or not v_names
        or not (z_levels == u_levels == v_levels)
    ):
        raise RuntimeError("Missing/mismatched hybrid-level fields.")

    z_da = (
        xr.concat([ds[n] for n in z_names], dim=xr.DataArray(z_levels, dims="level"))
        / G0
    )
    u_da = xr.concat([ds[n] for n in u_names], dim=xr.DataArray(u_levels, dims="level"))
    v_da = xr.concat([ds[n] for n in v_names], dim=xr.DataArray(v_levels, dims="level"))
    lead_dims = [d for d in z_da.dims if d not in ("level", "hrrr_y", "hrrr_x")]
    out_dims = tuple(lead_dims + ["hrrr_y", "hrrr_x"])
    z_da = z_da.transpose(*(lead_dims + ["level", "hrrr_y", "hrrr_x"]))
    u_da = u_da.transpose(*(lead_dims + ["level", "hrrr_y", "hrrr_x"]))
    v_da = v_da.transpose(*(lead_dims + ["level", "hrrr_y", "hrrr_x"]))
    lev_axis = len(lead_dims)

    u80_hl, v80_hl, k_idx = interp_uv_to_height(
        u_da.values,
        v_da.values,
        z_da.values,
        TARGET_AGL_M,
        lev_axis,
        return_level_indices=True,
    )
    lead_idx = out_dims.index("lead_time") if "lead_time" in out_dims else None
    print_level_usage(k_idx, z_levels, lead_dim_index=lead_idx)

    u10 = ds["u10m"].transpose(*out_dims).values
    v10 = ds["v10m"].transpose(*out_dims).values
    u80_log = u10 * LOG_FACTOR_80_FROM_10
    v80_log = v10 * LOG_FACTOR_80_FROM_10

    methods = {
        "hl_interp": (u80_hl, v80_hl),
        "loglaw_const": (u80_log, v80_log),
    }
    return methods, out_dims, z_levels


def _evaluate_and_plot(
    hrrr_source: HRRR,
    ds: xr.Dataset,
    methods: dict[str, tuple[np.ndarray, np.ndarray]],
    out_dims: tuple[str, ...],
    out_prefix: str,
    calib_truth_arrays: tuple[np.ndarray, np.ndarray] | None = None,
    lr_features: tuple[np.ndarray, np.ndarray] | None = None,  # (X_test, X_calib)
    calib_loglaw_arrays: (
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
    ) = None,
) -> None:
    """Evaluate all methods against HRRR truth and generate metrics/plots."""
    if "lead_time" not in ds.coords:
        ds = ds.expand_dims({"lead_time": np.array([np.timedelta64(0, "h")])})

    valid_times = np.unique(
        (
            ds["time"].values[:, None].astype("datetime64[ns]")
            + ds["lead_time"].values[None, :].astype("timedelta64[ns]")
        ).reshape(-1)
    )
    truth_u, truth_v = _build_truth_from_hrrr(hrrr_source, ds, valid_times)
    truth_coords = {
        "time": ds["time"].values,
        "lead_time": ds["lead_time"].values,
        "hrrr_y": ds["hrrr_y"].values,
        "hrrr_x": ds["hrrr_x"].values,
    }
    truth_u_da = xr.DataArray(
        truth_u, dims=("time", "lead_time", "hrrr_y", "hrrr_x"), coords=truth_coords
    )
    truth_v_da = xr.DataArray(
        truth_v, dims=("time", "lead_time", "hrrr_y", "hrrr_x"), coords=truth_coords
    )
    truth_u_b = truth_u_da.broadcast_like(ds["u10m"].transpose(*out_dims)).values
    truth_v_b = truth_v_da.broadcast_like(ds["v10m"].transpose(*out_dims)).values

    # Add direct LR on all wind fields (no dependence on hl_interp output).
    if lr_features is not None and calib_truth_arrays is not None:
        X_test, X_cal = lr_features
        cal_truth_u, cal_truth_v = calib_truth_arrays
        beta_u = fit_lr_global(X_cal, cal_truth_u)
        beta_v = fit_lr_global(X_cal, cal_truth_v)
        yhat_u = apply_lr_global(X_test, beta_u)
        yhat_v = apply_lr_global(X_test, beta_v)
        base_shape = methods["hl_interp"][0].shape
        methods["lr_all_wind_fields"] = (
            yhat_u.reshape(base_shape),
            yhat_v.reshape(base_shape),
        )

    # Option A: calibrated per-pixel scale from 10m->80m.
    # Option B: calibrated per-pixel effective log-law z0 from calibration window.
    if calib_loglaw_arrays is not None:
        cal_u10, cal_v10, cal_u80_truth, cal_v80_truth = calib_loglaw_arrays
        test_u10 = ds["u10m"].transpose(*out_dims).values
        test_v10 = ds["v10m"].transpose(*out_dims).values

        r = fit_scale_uv(cal_u10, cal_v10, cal_u80_truth, cal_v80_truth)
        methods["loglaw_scale_calib"] = (test_u10 * r, test_v10 * r)

        z0 = fit_loglaw_z0(cal_u10, cal_v10, cal_u80_truth, cal_v80_truth)
        fz0 = loglaw_factor_from_z0(z0)
        methods["loglaw_z0_calib"] = (test_u10 * fz0, test_v10 * fz0)

    for name, (up, vp) in methods.items():
        print(f"\nMethod: {name}")
        print("u80m:", metric_dict(up, truth_u_b))
        print("v80m:", metric_dict(vp, truth_v_b))

    lead_hours = ds["lead_time"].values.astype("timedelta64[h]").astype(int)
    plt.close("all")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, (up, vp) in methods.items():
        axes[0].plot(
            lead_hours, rmse_by_lead(up, truth_u_b, out_dims), marker="o", label=name
        )
        axes[1].plot(
            lead_hours, rmse_by_lead(vp, truth_v_b, out_dims), marker="o", label=name
        )
    axes[0].set_title("u80m RMSE by lead")
    axes[1].set_title("v80m RMSE by lead")
    for ax in axes:
        ax.set_xlabel("Lead hour")
        ax.set_ylabel("RMSE (m s-1)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(f"outputs/{out_prefix}_rmse_by_lead.png", dpi=150, bbox_inches="tight")

    coords = {d: ds[d].values for d in out_dims}
    for method_name, (u_pred, v_pred) in methods.items():
        u_da = xr.DataArray(u_pred, dims=out_dims, coords=coords)
        v_da = xr.DataArray(v_pred, dims=out_dims, coords=coords)
        plot_three_panel_all_leads(
            u_da,
            truth_u_da.broadcast_like(u_da),
            f"u80m ({method_name})",
            f"outputs/{out_prefix}_{method_name}_u80m_maps.png",
        )
        plot_three_panel_all_leads(
            v_da,
            truth_v_da.broadcast_like(v_da),
            f"v80m ({method_name})",
            f"outputs/{out_prefix}_{method_name}_v80m_maps.png",
        )


def run_hrrr_only() -> None:
    """Run HRRR-only benchmark with fixed fit/test timestamps 1 hour apart."""
    print("\n=== HRRR-only comparison ===")
    hrrr = HRRR()
    if HRRR_TEST_TIME - HRRR_FIT_TIME != timedelta(hours=1):
        raise ValueError(
            "HRRR_FIT_TIME and HRRR_TEST_TIME must be exactly 1 hour apart."
        )
    fit_time = np.datetime64(HRRR_FIT_TIME)
    test_time = np.datetime64(HRRR_TEST_TIME)
    variables = ["u10m", "v10m"] + [
        f"{p}{k}hl"
        for p in ["Z", "u", "v"]
        for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 20, 25, 30]
    ]
    try:
        # Fit dataset: one fixed HRRR field.
        fit_da = hrrr(np.array([fit_time]), variables)
        calib_da = fit_da
        calib_ds = calib_da.to_dataset(dim="variable")
        if "lead_time" not in calib_ds.coords:
            calib_ds = calib_ds.expand_dims(
                {"lead_time": np.array([np.timedelta64(0, "h")])}
            )
        _, calib_out_dims, _ = _derive_methods(calib_ds)
        calib_valid_times = np.unique(
            (
                calib_ds["time"].values[:, None].astype("datetime64[ns]")
                + calib_ds["lead_time"].values[None, :].astype("timedelta64[ns]")
            ).reshape(-1)
        )
        cal_truth_u, cal_truth_v = _build_truth_from_hrrr(
            hrrr, calib_ds, calib_valid_times
        )
        cal_truth_coords = {
            "time": calib_ds["time"].values,
            "lead_time": calib_ds["lead_time"].values,
            "hrrr_y": calib_ds["hrrr_y"].values,
            "hrrr_x": calib_ds["hrrr_x"].values,
        }
        cal_truth_u_da = xr.DataArray(
            cal_truth_u,
            dims=("time", "lead_time", "hrrr_y", "hrrr_x"),
            coords=cal_truth_coords,
        )
        cal_truth_v_da = xr.DataArray(
            cal_truth_v,
            dims=("time", "lead_time", "hrrr_y", "hrrr_x"),
            coords=cal_truth_coords,
        )
        cal_truth_u_b = cal_truth_u_da.broadcast_like(
            calib_ds["u10m"].transpose(*calib_out_dims)
        ).values
        cal_truth_v_b = cal_truth_v_da.broadcast_like(
            calib_ds["v10m"].transpose(*calib_out_dims)
        ).values
        cal_u10 = calib_ds["u10m"].transpose(*calib_out_dims).values
        cal_v10 = calib_ds["v10m"].transpose(*calib_out_dims).values

        # Test dataset: second fixed HRRR field exactly 1h later.
        da = hrrr(np.array([test_time]), variables)
        ds = da.to_dataset(dim="variable")
        if "lead_time" not in ds.coords:
            ds = ds.expand_dims({"lead_time": np.array([np.timedelta64(0, "h")])})
        methods, out_dims, _ = _derive_methods(ds)
        X_test = build_wind_feature_tensor(ds, out_dims)
        X_cal = build_wind_feature_tensor(calib_ds, calib_out_dims)

        _evaluate_and_plot(
            hrrr,
            ds,
            methods,
            out_dims,
            "hrrr_80m_methods",
            calib_truth_arrays=(
                cal_truth_u_b.reshape(
                    -1, cal_truth_u_b.shape[-2], cal_truth_u_b.shape[-1]
                ),
                cal_truth_v_b.reshape(
                    -1, cal_truth_v_b.shape[-2], cal_truth_v_b.shape[-1]
                ),
            ),
            lr_features=(X_test, X_cal),
            calib_loglaw_arrays=(
                cal_u10.reshape(-1, cal_u10.shape[-2], cal_u10.shape[-1]),
                cal_v10.reshape(-1, cal_v10.shape[-2], cal_v10.shape[-1]),
                cal_truth_u_b.reshape(
                    -1, cal_truth_u_b.shape[-2], cal_truth_u_b.shape[-1]
                ),
                cal_truth_v_b.reshape(
                    -1, cal_truth_v_b.shape[-2], cal_truth_v_b.shape[-1]
                ),
            ),
        )
    finally:
        _close_hrrr_sessions(hrrr)


def run_stormcast() -> None:
    """Run StormCast benchmark and compare derived 80m winds to HRRR truth."""
    print("\n=== StormCast vs HRRR comparison ===")
    init_test = (datetime.today() - timedelta(days=5)).replace(
        minute=0, second=0, microsecond=0
    )
    hrrr = HRRR()
    package = StormCast.load_default_package()
    model = StormCast.load_model(package, conditioning_data_source=GFS_FX)
    io = ZarrBackend(
        file_name="outputs/stormcast_raw_ensemble.zarr",
        backend_kwargs={"overwrite": True},
    )
    try:
        run.ensemble(
            [init_test.isoformat()],
            N_STEPS,
            ENSEMBLE_SIZE,
            model,
            hrrr,
            io,
            Zero(),
            batch_size=BATCH_SIZE,
        )
        ds = xr.open_zarr("outputs/stormcast_raw_ensemble.zarr")
        methods, out_dims, _ = _derive_methods(ds)

        # Calibration run from 6h before init_test.
        init_cal = init_test - timedelta(hours=CALIB_HOURS)
        io_cal = ZarrBackend(
            file_name="outputs/stormcast_raw_ensemble_calib.zarr",
            backend_kwargs={"overwrite": True},
        )
        run.ensemble(
            [init_cal.isoformat()],
            CALIB_HOURS,
            ENSEMBLE_SIZE,
            model,
            hrrr,
            io_cal,
            Zero(),
            batch_size=BATCH_SIZE,
        )
        calib_ds = xr.open_zarr("outputs/stormcast_raw_ensemble_calib.zarr")
        calib_valid = (
            calib_ds["time"].values[:, None].astype("datetime64[ns]")
            + calib_ds["lead_time"].values[None, :].astype("timedelta64[ns]")
        )[0]
        mask = (calib_valid >= np.datetime64(init_cal)) & (
            calib_valid < np.datetime64(init_test)
        )
        calib_ds = calib_ds.sel(lead_time=calib_ds["lead_time"].values[mask])
        _, calib_out_dims, _ = _derive_methods(calib_ds)
        calib_valid_times = np.unique(
            (
                calib_ds["time"].values[:, None].astype("datetime64[ns]")
                + calib_ds["lead_time"].values[None, :].astype("timedelta64[ns]")
            ).reshape(-1)
        )
        cal_truth_u, cal_truth_v = _build_truth_from_hrrr(
            hrrr, calib_ds, calib_valid_times
        )
        cal_truth_coords = {
            "time": calib_ds["time"].values,
            "lead_time": calib_ds["lead_time"].values,
            "hrrr_y": calib_ds["hrrr_y"].values,
            "hrrr_x": calib_ds["hrrr_x"].values,
        }
        cal_truth_u_da = xr.DataArray(
            cal_truth_u,
            dims=("time", "lead_time", "hrrr_y", "hrrr_x"),
            coords=cal_truth_coords,
        )
        cal_truth_v_da = xr.DataArray(
            cal_truth_v,
            dims=("time", "lead_time", "hrrr_y", "hrrr_x"),
            coords=cal_truth_coords,
        )
        cal_truth_u_b = cal_truth_u_da.broadcast_like(
            calib_ds["u10m"].transpose(*calib_out_dims)
        ).values
        cal_truth_v_b = cal_truth_v_da.broadcast_like(
            calib_ds["v10m"].transpose(*calib_out_dims)
        ).values
        cal_u10 = calib_ds["u10m"].transpose(*calib_out_dims).values
        cal_v10 = calib_ds["v10m"].transpose(*calib_out_dims).values
        X_test = build_wind_feature_tensor(ds, out_dims)
        X_cal = build_wind_feature_tensor(calib_ds, calib_out_dims)

        _evaluate_and_plot(
            hrrr,
            ds,
            methods,
            out_dims,
            "stormcast_80m_methods",
            calib_truth_arrays=(
                cal_truth_u_b.reshape(
                    -1, cal_truth_u_b.shape[-2], cal_truth_u_b.shape[-1]
                ),
                cal_truth_v_b.reshape(
                    -1, cal_truth_v_b.shape[-2], cal_truth_v_b.shape[-1]
                ),
            ),
            lr_features=(X_test, X_cal),
            calib_loglaw_arrays=(
                cal_u10.reshape(-1, cal_u10.shape[-2], cal_u10.shape[-1]),
                cal_v10.reshape(-1, cal_v10.shape[-2], cal_v10.shape[-1]),
                cal_truth_u_b.reshape(
                    -1, cal_truth_u_b.shape[-2], cal_truth_u_b.shape[-1]
                ),
                cal_truth_v_b.reshape(
                    -1, cal_truth_v_b.shape[-2], cal_truth_v_b.shape[-1]
                ),
            ),
        )
    finally:
        _close_hrrr_sessions(hrrr, getattr(model, "conditioning_data_source", None))


def main() -> None:
    """CLI entrypoint for selecting HRRR, StormCast, or both benchmarks."""
    parser = argparse.ArgumentParser(
        description="Compare 80m wind methods in HRRR and StormCast."
    )
    parser.add_argument(
        "--mode",
        choices=["hrrr", "stormcast", "both"],
        default="both",
        help="Which benchmark to run.",
    )
    args = parser.parse_args()
    if args.mode in ("hrrr", "both"):
        run_hrrr_only()
    if args.mode in ("stormcast", "both"):
        run_stormcast()


if __name__ == "__main__":
    main()
