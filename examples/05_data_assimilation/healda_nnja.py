# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
HealDA Inference with NNJA Conventional Observations
====================================================

Produce a global weather analysis from satellite radiances and NNJA
conventional observations using the HealDA data-assimilation model, and
optionally compare the result against the GFS operational analysis.

This is the NNJA-backed sibling of ``real_time_healda.py``: the only
substantive change is that the conventional observation source is
:class:`earth2studio.data.NNJAObsConv` (the NOAA-NASA Joint Archive on
AWS S3) instead of :class:`earth2studio.data.NomadsGDASObsConv` (the
real-time NOMADS production server).

Data sources (all public, no paid accounts required):

  Satellite (always fetched):
    - JPSS_ATMS    ATMS radiances from NOAA AWS S3 (n20, npp)
    - MetOpMHS     MHS radiances from EUMETSAT (metop-a/b/c)  [needs free EUMETSAT key]
    - MetOpAMSUA   AMSU-A radiances from EUMETSAT (metop-a/b/c) [needs free EUMETSAT key]

  Conventional (fetched unless --skip-conv):
    - NNJAObsConv  Temperature, humidity, pressure, winds from NNJA PrepBUFR (S3)

  Reference for comparison (fetched with --compare):
    - GFS 0.25 deg analysis from NOAA AWS S3

Observations are spatially thinned to ~1 deg (HEALPix nside=64) to
match the training distribution and fit in GPU memory. Use ``--no-thin``
to disable.

Why this script exists alongside ``real_time_healda.py``
--------------------------------------------------------
- ``real_time_healda.py`` uses NOMADS GDAS, which only retains the
  last ~2 days of PrepBUFR. It is the right tool for *real-time* (or
  last-48-hours) inference.
- ``healda_nnja.py`` uses NNJA, a reanalysis-style archive on AWS S3.
  NNJA goes back to **1979-01-01** but publishes new cycles with a
  ~1-2 day latency. It is the right tool for *historical* inference
  and for reproducibility (the same S3 object can be fetched years
  later). The satellite sources (JPSS / MetOp) cap the practical
  start date around 2007-2023 depending on which sensors are
  available; see ``earth2studio/data/jpss_atms.py`` for JPSS's
  ``_SAT_START_DATE`` (2023-09-06) and the MetOp data sources for
  EUMETSAT availability.

Examples
--------
  # ~2-day-old cycle, satellite + NNJA conv, no comparison:
  python healda_nnja.py

  # Specific historical timestamp with GFS comparison:
  python healda_nnja.py --timestamp 2024-06-01T00:00 --compare

  # Multiple timestamps:
  python healda_nnja.py --timestamp 2024-06-01T00:00 2024-06-02T00:00 --compare

  # Satellite-only (skip conventional and EUMETSAT):
  python healda_nnja.py --skip-conv --skip-eumetsat

Environment variables
---------------------
  EUMETSAT_CONSUMER_KEY     Free API key from https://eoportal.eumetsat.int/
  EUMETSAT_CONSUMER_SECRET  Corresponding secret (both needed for MHS + AMSU-A)

Output files (saved to --output-dir, default: outputs/)
-------------------------------------------------------
  healda_nnja_obs_YYYYMMDD_HH.jpg         Observation locations after thinning
  healda_nnja_YYYYMMDD_HH.nc              NetCDF analysis (74 channels, lat-lon grid)
  healda_nnja_YYYYMMDD_HH.jpg             Analysis fields (t2m, z500)
  healda_nnja_vs_gfs_YYYYMMDD_HH.jpg      3-row comparison: HealDA / GFS / Difference
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone

import cartopy.crs as ccrs
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from earth2studio.data import (
    GFS,
    JPSS_ATMS,
    MetOpAMSUA,
    MetOpMHS,
    NNJAObsConv,
    fetch_dataframe,
)
from earth2studio.models.da import HealDA

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

PLOT_VARS = ["t2m", "z500"]
METRIC_VARS = ["t2m", "z500", "u10m", "msl"]
PROJECTION = ccrs.Robinson()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def snap_to_cycle(dt: datetime, cycle_hours: int = 6) -> datetime:
    """Snap a datetime down to the nearest GDAS/GFS cycle boundary."""
    hour = (dt.hour // cycle_hours) * cycle_hours
    return dt.replace(hour=hour, minute=0, second=0, microsecond=0)


def to_numpy(arr: np.ndarray) -> np.ndarray:
    """CuPy -> NumPy if needed."""
    return arr.get() if hasattr(arr, "get") else arr


def thin_to_healpix(
    df: pd.DataFrame,
    nside: int = 64,
    max_per_group: int = 120,
    max_total: int = 50_000_000,
) -> pd.DataFrame:
    """Spatially thin observations to ~1 deg while preserving raw timestamps.

    For each (HEALPix pixel, variable, channel, satellite) group, keeps at
    most *max_per_group* observations (in their original order, so the
    temporal spread of the source stream is retained).  This matches the
    ~1 deg thinning that UFS Replay applies during training, while leaving
    every surviving observation's timestamp at native resolution for
    HealDA's time-conditioned encoder.
    """
    lat = np.clip(df["lat"].values.astype(np.float64), -90.0, 90.0)
    lon = df["lon"].values.astype(np.float64) % 360.0
    pix = hp.ang2pix(nside, np.deg2rad(90.0 - lat), np.deg2rad(lon))

    df = df.copy()
    df["_hpx_pix"] = pix

    spatial_cols = ["_hpx_pix", "variable"]
    if "channel_index" in df.columns:
        spatial_cols.append("channel_index")
    if "satellite" in df.columns:
        spatial_cols.append("satellite")

    if max_per_group and max_per_group > 0:
        thinned = (
            df.groupby(spatial_cols, sort=False)
            .head(max_per_group)
            .reset_index(drop=True)
        )
    else:
        thinned = df.reset_index(drop=True)

    thinned = thinned.drop(columns=["_hpx_pix"])

    if len(thinned) > max_total:
        logger.warning(
            f"Thinned obs ({len(thinned):,}) exceed GPU budget ({max_total:,}), "
            f"random subsampling."
        )
        thinned = thinned.sample(n=max_total, random_state=42)

    return thinned


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
def fetch_satellite_obs(
    time_np: np.ndarray,
    sat_schema: dict,
    window: tuple[timedelta, timedelta],
    use_eumetsat: bool = True,
) -> pd.DataFrame:
    """Fetch and concatenate all available satellite observations."""
    fields = np.array(list(sat_schema.keys()))
    frames: list[pd.DataFrame] = []

    logger.info("Fetching ATMS satellite observations from NOAA AWS...")
    atms_df = fetch_dataframe(
        JPSS_ATMS(satellites=["n20", "npp"], time_tolerance=window),
        time=time_np,
        variable=np.array(["atms"]),
        fields=fields,
    )
    logger.info(f"  ATMS: {len(atms_df):,} obs")
    frames.append(atms_df)
    attrs = atms_df.attrs

    if use_eumetsat and os.environ.get("EUMETSAT_CONSUMER_KEY"):
        logger.info("Fetching MHS observations from EUMETSAT...")
        mhs_df = fetch_dataframe(
            MetOpMHS(time_tolerance=window),
            time=time_np,
            variable=np.array(["mhs"]),
            fields=fields,
        )
        logger.info(f"  MHS: {len(mhs_df):,} obs")
        frames.append(mhs_df)

        logger.info("Fetching AMSU-A observations from EUMETSAT...")
        amsua_df = fetch_dataframe(
            MetOpAMSUA(time_tolerance=window),
            time=time_np,
            variable=np.array(["amsua"]),
            fields=fields,
        )
        logger.info(f"  AMSU-A: {len(amsua_df):,} obs")
        frames.append(amsua_df)
    elif use_eumetsat:
        logger.warning(
            "EUMETSAT credentials not set - skipping MHS and AMSU-A.  "
            "See https://eoportal.eumetsat.int/"
        )

    sat_df = pd.concat(frames, ignore_index=True)
    sat_df.attrs = attrs
    return sat_df


def fetch_conventional_obs(
    time_np: np.ndarray,
    conv_schema: dict,
    window: tuple[timedelta, timedelta],
) -> pd.DataFrame | None:
    """Fetch conventional observations from the NNJA reanalysis archive."""
    fields = np.array(list(conv_schema.keys()))
    variables = np.array(["t", "q", "pres", "u", "v"])

    logger.info("Fetching conventional observations from NNJA (S3 PrepBUFR)...")
    conv_df = fetch_dataframe(
        NNJAObsConv(time_tolerance=window),
        time=time_np,
        variable=variables,
        fields=fields,
    )
    logger.info(f"  Conventional: {len(conv_df):,} obs")
    return conv_df if len(conv_df) > 0 else None


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_obs_locations(
    sat_df: pd.DataFrame,
    conv_df: pd.DataFrame | None,
    analysis_time: datetime,
    output_dir: str,
) -> None:
    """Scatter-plot observation locations per sensor after thinning."""
    panels: list[tuple[str, pd.DataFrame, str]] = []
    if sat_df is not None and len(sat_df) > 0:
        for sensor in sorted(sat_df["variable"].unique()):
            panels.append((sensor, sat_df[sat_df["variable"] == sensor], "tab:orange"))
    if conv_df is not None and len(conv_df) > 0:
        for var in sorted(conv_df["variable"].unique()):
            panels.append(
                (f"conv:{var}", conv_df[conv_df["variable"] == var], "tab:blue")
            )

    n = min(len(panels), 8)
    if n == 0:
        return
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        subplot_kw={"projection": PROJECTION},
        figsize=(5 * ncols, 4 * nrows),
    )
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, (label, odf, color) in zip(axes_flat, panels[:n]):
        ax.set_global()
        ax.coastlines(linewidth=0.5)
        ax.gridlines(linewidth=0.3, alpha=0.5)
        step = max(1, len(odf) // 20_000)
        ax.scatter(
            odf["lon"].values[::step],
            odf["lat"].values[::step],
            s=0.15,
            alpha=0.3,
            c=color,
            transform=ccrs.PlateCarree(),
        )
        ax.set_title(f"{label} (n={len(odf):,})", fontsize=11)
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Observation Locations (after thinning) - {analysis_time:%Y-%m-%d %H} UTC",
        fontsize=14,
    )
    plt.tight_layout()
    path = os.path.join(
        output_dir, f"healda_nnja_obs_{analysis_time:%Y%m%d_%H}.jpg"
    )
    plt.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved obs plot to {path}")
    plt.close()


def plot_analysis(result, analysis_time: datetime, mode: str, output_dir: str) -> None:
    """Plot HealDA analysis fields (t2m, z500)."""
    lat = result.coords["lat"].values
    lon = result.coords["lon"].values

    fig, axes = plt.subplots(
        1, 2, subplot_kw={"projection": PROJECTION}, figsize=(16, 4)
    )
    for ax, var, cmap in zip(axes, PLOT_VARS, ["Spectral_r", "PRGn"]):
        field = to_numpy(result.sel(variable=var).data[0])
        im = ax.pcolormesh(lon, lat, field, transform=ccrs.PlateCarree(), cmap=cmap)
        ax.coastlines(linewidth=0.5)
        ax.gridlines(linewidth=0.3, alpha=0.5)
        fig.colorbar(im, ax=ax, shrink=0.6)
        ax.set_title(var)

    fig.suptitle(
        f"HealDA Analysis ({mode}, NNJA) - {analysis_time:%Y-%m-%d %H} UTC",
        fontsize=14,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, f"healda_nnja_{analysis_time:%Y%m%d_%H}.jpg")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved analysis plot to {path}")
    plt.close()


def plot_comparison(
    result, gfs_interp, analysis_time: datetime, mode: str, output_dir: str
) -> None:
    """3-row comparison: HealDA / GFS / Difference for t2m and z500."""
    lat = result.coords["lat"].values
    lon = result.coords["lon"].values
    diff_ranges = {"t2m": (-10, 10), "z500": (-500, 500)}
    cmaps = {"t2m": "Spectral_r", "z500": "PRGn"}

    fig, axes = plt.subplots(
        3,
        len(PLOT_VARS),
        subplot_kw={"projection": PROJECTION},
        figsize=(8 * len(PLOT_VARS), 12),
    )

    for col, var in enumerate(PLOT_VARS):
        healda_field = to_numpy(result.sel(variable=var).data[0])
        gfs_field = to_numpy(gfs_interp.sel(variable=var).data[0])
        diff = healda_field - gfs_field
        vmin_shared = min(healda_field.min(), gfs_field.min())
        vmax_shared = max(healda_field.max(), gfs_field.max())

        # Row 0: HealDA
        ax = axes[0, col]
        im = ax.pcolormesh(
            lon,
            lat,
            healda_field,
            transform=ccrs.PlateCarree(),
            cmap=cmaps[var],
            vmin=vmin_shared,
            vmax=vmax_shared,
        )
        ax.coastlines(linewidth=0.5)
        ax.gridlines(linewidth=0.3, alpha=0.5)
        fig.colorbar(im, ax=ax, shrink=0.6)
        ax.set_title(f"HealDA - {var}", fontsize=12)

        # Row 1: GFS
        ax = axes[1, col]
        im = ax.pcolormesh(
            lon,
            lat,
            gfs_field,
            transform=ccrs.PlateCarree(),
            cmap=cmaps[var],
            vmin=vmin_shared,
            vmax=vmax_shared,
        )
        ax.coastlines(linewidth=0.5)
        ax.gridlines(linewidth=0.3, alpha=0.5)
        fig.colorbar(im, ax=ax, shrink=0.6)
        ax.set_title(f"GFS - {var}", fontsize=12)

        # Row 2: Difference
        ax = axes[2, col]
        dmin, dmax = diff_ranges[var]
        mae = float(np.abs(diff).mean())
        im = ax.pcolormesh(
            lon,
            lat,
            diff,
            transform=ccrs.PlateCarree(),
            cmap="RdBu_r",
            vmin=dmin,
            vmax=dmax,
        )
        ax.coastlines(linewidth=0.5)
        ax.gridlines(linewidth=0.3, alpha=0.5)
        fig.colorbar(im, ax=ax, shrink=0.6)
        ax.set_title(f"HealDA - GFS - {var}  (MAE={mae:.2f})", fontsize=12)

    fig.suptitle(
        f"HealDA vs GFS ({mode}, NNJA) - {analysis_time:%Y-%m-%d %H} UTC",
        fontsize=16,
        y=0.98,
    )
    plt.tight_layout()
    path = os.path.join(
        output_dir, f"healda_nnja_vs_gfs_{analysis_time:%Y%m%d_%H}.jpg"
    )
    plt.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved comparison plot to {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "HealDA inference using NNJA conventional observations, with "
            "optional GFS comparison."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Analysis time(s) in ISO format, e.g. 2024-06-01T00:00. Snapped "
            "to nearest 6 h cycle. Overrides --hours-ago."
        ),
    )
    parser.add_argument(
        "--hours-ago",
        type=float,
        default=48.0,
        help=(
            "Hours before now (default: 48). NNJA has a ~1-2 day publication "
            "lag, so 48 h is a safe default. Ignored if --timestamp is set."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare HealDA against GFS analysis"
    )
    parser.add_argument(
        "--no-thin", action="store_true", help="Skip spatial thinning (may OOM)"
    )
    parser.add_argument(
        "--skip-eumetsat", action="store_true", help="Skip MHS and AMSU-A"
    )
    parser.add_argument(
        "--skip-conv", action="store_true", help="Skip conventional observations"
    )
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # --- Resolve analysis times -------------------------------------------
    if args.timestamp:
        analysis_times = []
        for ts in args.timestamp:
            dt = datetime.fromisoformat(ts)
            if dt.hour % 6 != 0 or dt.minute != 0:
                dt = snap_to_cycle(dt)
                logger.warning(f"Snapped {ts} -> {dt}")
            analysis_times.append(dt)
    else:
        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        analysis_times = [snap_to_cycle(now_utc - timedelta(hours=args.hours_ago))]

    logger.info(f"Analysis times: {[t.isoformat() for t in analysis_times]}")

    obs_window = (timedelta(hours=-21), timedelta(hours=3))

    # --- Load model -------------------------------------------------------
    logger.info("Loading HealDA model...")
    package = HealDA.load_default_package()
    model = HealDA.load_model(package, lat_lon=True).to(device)
    logger.info(f"HealDA loaded on {device}")
    conv_schema, sat_schema = model.input_coords()

    # --- Process each analysis time ---------------------------------------
    results: dict[datetime, tuple] = {}

    for analysis_time in analysis_times:
        time_np = np.array([np.datetime64(analysis_time)])
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing {analysis_time} UTC")
        logger.info(f"{'=' * 60}")

        # Fetch satellite observations
        sat_df = fetch_satellite_obs(
            time_np, sat_schema, obs_window, use_eumetsat=not args.skip_eumetsat
        )
        logger.info(f"Total satellite: {len(sat_df):,}")
        for sensor, count in sat_df["variable"].value_counts().items():
            n_plat = sat_df.loc[sat_df["variable"] == sensor, "satellite"].nunique()
            logger.info(f"  {sensor}: {count:,} obs from {n_plat} platform(s)")

        # Fetch conventional observations
        conv_df = None
        if not args.skip_conv:
            try:
                conv_df = fetch_conventional_obs(time_np, conv_schema, obs_window)
            except Exception as e:
                logger.warning(
                    f"Conventional fetch failed: {e}. Continuing satellite-only."
                )
        if conv_df is not None:
            for var, count in conv_df["variable"].value_counts().items():
                logger.info(f"  conv/{var}: {count:,} obs")

        # Thin to ~1 deg (match UFS Replay training density)
        if not args.no_thin:
            n_before = len(sat_df)
            sat_df = thin_to_healpix(sat_df)
            sat_df.attrs = {"request_time": time_np}
            logger.info(
                f"Thinned satellite: {n_before:,} -> {len(sat_df):,} "
                f"({100 * len(sat_df) / max(n_before, 1):.1f}%)"
            )

            if conv_df is not None:
                n_before = len(conv_df)
                conv_df = thin_to_healpix(conv_df)
                conv_df.attrs = {"request_time": time_np}
                logger.info(
                    f"Thinned conventional: {n_before:,} -> {len(conv_df):,} "
                    f"({100 * len(conv_df) / max(n_before, 1):.1f}%)"
                )

        # Plot observation network
        plot_obs_locations(sat_df, conv_df, analysis_time, args.output_dir)

        # Run HealDA inference
        mode = "conv+sat" if conv_df is not None else "sat-only"
        n_obs = len(sat_df) + (len(conv_df) if conv_df is not None else 0)
        logger.info(f"Running HealDA ({mode}, {n_obs:,} obs)...")
        torch.manual_seed(42)
        result = model(conv_obs=conv_df, sat_obs=sat_df)
        logger.info(f"Analysis shape: {result.shape}")

        for var in METRIC_VARS:
            f = to_numpy(result.sel(variable=var).data[0])
            logger.info(f"  {var}: mean={f.mean():.2f}  std={f.std():.2f}")

        # Move to CPU for saving / plotting
        if hasattr(result.data, "get"):
            result = result.copy(data=result.data.get())

        # Save NetCDF
        nc_path = os.path.join(
            args.output_dir, f"healda_nnja_{analysis_time:%Y%m%d_%H}.nc"
        )
        result.to_netcdf(nc_path)
        logger.info(f"Saved NetCDF to {nc_path}")

        # Plot analysis fields
        plot_analysis(result, analysis_time, mode, args.output_dir)

        results[analysis_time] = (result, conv_df, sat_df, mode)

    # --- GFS comparison ---------------------------------------------------
    if args.compare:
        gfs_ds = GFS(source="aws")

        for analysis_time, (result, conv_df, sat_df, mode) in results.items():
            time_np = np.array([np.datetime64(analysis_time)])
            logger.info(f"\nFetching GFS analysis for {analysis_time} UTC...")
            gfs_da = gfs_ds(time_np, METRIC_VARS)

            lat = result.coords["lat"].values
            lon = result.coords["lon"].values
            gfs_interp = gfs_da.interp(lat=lat, lon=lon, method="nearest")

            logger.info(f"HealDA vs GFS - {analysis_time} UTC:")
            for var in METRIC_VARS:
                hf = to_numpy(result.sel(variable=var).data[0])
                gf = to_numpy(gfs_interp.sel(variable=var).data[0])
                mae = float(np.abs(hf - gf).mean())
                rmse = float(np.sqrt(((hf - gf) ** 2).mean()))
                bias = float((hf - gf).mean())
                logger.info(
                    f"  {var}: MAE={mae:.4f}  RMSE={rmse:.4f}  Bias={bias:.4f}"
                )

            plot_comparison(result, gfs_interp, analysis_time, mode, args.output_dir)

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
