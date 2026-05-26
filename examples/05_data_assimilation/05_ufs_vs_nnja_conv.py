# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%
"""
UFS vs NNJA conventional observations: source comparison
========================================================

Side-by-side comparison of the two **conventional** observation
archives Earth2Studio exposes for data assimilation, **without**
running any model.

The script fetches the same 6-hourly analysis cycle from both archives
and renders three things per variable:

1. A short text summary (row counts, lat/lon ranges, station coverage,
   raw value range).
2. Overlaid histograms of the ``observation`` column so the value
   distributions can be compared directly.
3. Side-by-side global scatter maps showing where each archive has
   coverage for that variable.

In this example you will learn:

- How to query :py:class:`earth2studio.data.UFSObsConv` and
  :py:class:`earth2studio.data.NNJAObsConv` for a single cycle.
- The practical differences between the two archives: schema columns,
  units, coverage, and value-range biases.

.. note::
   The NNJA satellite archive (``NNJAObsSat``) is no longer available
   from the public bucket, so this example focuses on conventional
   observations only. For satellite radiances use the independent
   :py:class:`earth2studio.data.UFSObsSat` source.
"""
# /// script
# dependencies = [
#   "earth2studio[data] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "cartopy",
#   "matplotlib",
# ]
# ///

# %%
# Set Up
# ------
# Pick a single analysis cycle that both archives carry. NNJA requires a
# 6-hourly cycle (00/06/12/18z); UFS does too in practice once the
# tolerance window is applied.

# %%
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

os.makedirs("outputs", exist_ok=True)

from earth2studio.data import NNJAObsConv, UFSObsConv
from earth2studio.lexicon import GSIConventionalLexicon, NNJAObsConvLexicon

ANALYSIS_TIME = datetime(2024, 6, 1, 0)
# Symmetric +-3 h window: a single 6-hourly cycle centred on
# ``ANALYSIS_TIME``. Wide enough to catch GOES / MODIS AMV types that
# only fire on a subset of synoptic times within the cycle.
TIME_TOLERANCE = (timedelta(hours=-3), timedelta(hours=3))

# Variables that exist in *both* lexicons. We split them into two
# groups because GPS-RO observations are not row-for-row comparable
# between the two archives:
#
# - PrepBUFR conv (``u, v, q, t, pres``) -> values in the same SI
#   units, can be overlaid on the same histogram axis.
# - GPS-RO (``gps, gps_t, gps_q``) -> NNJA returns raw BUFR bending
#   angle / T / Q at impact-parameter levels; UFS returns the same
#   occultation after GSI thinning, QC and interpolation onto pressure
#   levels. Plotted side-by-side, not overlaid.
_SHARED_ALL = sorted(
    set(GSIConventionalLexicon.VOCAB.keys())
    & set(NNJAObsConvLexicon.VOCAB.keys())
)
SHARED_CONV_VARS = [v for v in _SHARED_ALL if not v.startswith("gps")]
SHARED_GPSRO_VARS = [v for v in _SHARED_ALL if v.startswith("gps")]

logger.info(f"Shared conventional variables: {SHARED_CONV_VARS}")
logger.info(f"Shared GPS-RO variables:       {SHARED_GPSRO_VARS}")


# %%
# Fetch
# -----
# One call per archive. Both data sources accept a ``time``,
# ``variable`` and (optional) ``fields`` argument and return a
# :class:`pandas.DataFrame` with the columns named in their schema.

# %%
ufs_conv = UFSObsConv(time_tolerance=TIME_TOLERANCE)
nnja_conv = NNJAObsConv(time_tolerance=TIME_TOLERANCE)


def _safe_call(source, variables: list[str], label: str) -> pd.DataFrame:
    try:
        df = source(ANALYSIS_TIME, variables)
    except Exception as exc:
        logger.warning(f"{label}: fetch failed ({exc!r}); returning empty frame")
        return pd.DataFrame()
    df.attrs["label"] = label
    return df


ufs_conv_df = _safe_call(ufs_conv, SHARED_CONV_VARS, "UFS conv")
nnja_conv_df = _safe_call(nnja_conv, SHARED_CONV_VARS, "NNJA conv")
ufs_gpsro_df = (
    _safe_call(ufs_conv, SHARED_GPSRO_VARS, "UFS GPS-RO")
    if SHARED_GPSRO_VARS
    else pd.DataFrame()
)
nnja_gpsro_df = (
    _safe_call(nnja_conv, SHARED_GPSRO_VARS, "NNJA GPS-RO")
    if SHARED_GPSRO_VARS
    else pd.DataFrame()
)

logger.info(
    f"Row counts -- UFS conv: {len(ufs_conv_df):>8,}  NNJA conv: {len(nnja_conv_df):>8,}"
)
logger.info(
    f"Row counts -- UFS gps:  {len(ufs_gpsro_df):>8,}  NNJA gps:  {len(nnja_gpsro_df):>8,}"
)


# %%
# Schema diff
# -----------
# Show the (set-)difference of column names between the two archives.
# Conventional schemas are deliberately kept identical.

# %%
def _schema_diff(a: pd.DataFrame, b: pd.DataFrame, label_a: str, label_b: str) -> None:
    cols_a = set(a.columns)
    cols_b = set(b.columns)
    only_a = sorted(cols_a - cols_b)
    only_b = sorted(cols_b - cols_a)
    common = sorted(cols_a & cols_b)
    logger.info(f"[{label_a}] only columns: {only_a or '<none>'}")
    logger.info(f"[{label_b}] only columns: {only_b or '<none>'}")
    logger.info(f"common columns ({len(common)}): {common}")


logger.info("--- conventional schema diff ---")
_schema_diff(ufs_conv_df, nnja_conv_df, "UFS conv", "NNJA conv")


# %%
# Per-variable summary
# --------------------
# A compact text table showing per-variable row counts, value range,
# mean, and unique-station coverage.

# %%
def _summarise(df: pd.DataFrame, var: str, label: str) -> dict:
    if df.empty or "variable" not in df.columns:
        return {"label": label, "var": var, "n": 0}
    sub = df[df["variable"] == var]
    if sub.empty:
        return {"label": label, "var": var, "n": 0}
    obs = sub["observation"].astype(np.float64)
    record = {
        "label": label,
        "var": var,
        "n": len(sub),
        "mean": float(obs.mean()),
        "std": float(obs.std()),
        "min": float(obs.min()),
        "max": float(obs.max()),
        "lat_min": float(sub["lat"].min()) if "lat" in sub else float("nan"),
        "lat_max": float(sub["lat"].max()) if "lat" in sub else float("nan"),
    }
    if "station" in sub.columns:
        record["unique_stations"] = int(sub["station"].dropna().nunique())
    return record


summary_rows: list[dict] = []
for var in SHARED_CONV_VARS:
    summary_rows.append(_summarise(ufs_conv_df, var, "UFS"))
    summary_rows.append(_summarise(nnja_conv_df, var, "NNJA"))
for var in SHARED_GPSRO_VARS:
    summary_rows.append(_summarise(ufs_gpsro_df, var, "UFS"))
    summary_rows.append(_summarise(nnja_gpsro_df, var, "NNJA"))

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df[summary_df["n"] > 0].reset_index(drop=True)
print("\nPer-variable summary (rows with n>0):\n")
with pd.option_context("display.max_rows", None, "display.width", 160):
    print(summary_df.to_string(index=False))
summary_df.to_csv("outputs/05_ufs_vs_nnja_conv_summary.csv", index=False)


# %%
# Distribution plots
# ------------------
# One figure with one row per variable; left column overlays the UFS
# and NNJA histograms (log y-axis to keep tail visible), right column
# shows global scatter coverage on a Robinson projection.

# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

plt.close("all")
PROJ = ccrs.Robinson()
SUBSAMPLE_FOR_MAP = 5000  # cap per (archive, var) for readable scatters
RNG = np.random.default_rng(0)


def _values(df: pd.DataFrame, var: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (obs, lat, lon) for ``var``, dropping NaN observations."""
    if df.empty or "variable" not in df.columns:
        return np.empty(0), np.empty(0), np.empty(0)
    sub = df[df["variable"] == var].dropna(subset=["observation", "lat", "lon"])
    if sub.empty:
        return np.empty(0), np.empty(0), np.empty(0)
    return (
        sub["observation"].to_numpy(dtype=np.float64),
        sub["lat"].to_numpy(dtype=np.float64),
        sub["lon"].to_numpy(dtype=np.float64),
    )


def _shared_bins(a: np.ndarray, b: np.ndarray, n: int = 60) -> np.ndarray:
    """Histogram bins that span both arrays. Falls back gracefully if empty."""
    pieces = [x for x in (a, b) if x.size]
    if not pieces:
        return np.linspace(0, 1, n + 1)
    cat = np.concatenate(pieces)
    lo, hi = np.nanquantile(cat, [0.005, 0.995])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(cat.min()), float(cat.max())
        if lo == hi:
            hi = lo + 1.0
    return np.linspace(lo, hi, n + 1)


def _subsample(values: np.ndarray, lat: np.ndarray, lon: np.ndarray, k: int):
    if values.size <= k:
        return values, lat, lon
    idx = RNG.choice(values.size, size=k, replace=False)
    return values[idx], lat[idx], lon[idx]


def _plot_kind(
    ufs_df: pd.DataFrame,
    nnja_df: pd.DataFrame,
    variables: list[str],
    title: str,
    out_path: str,
) -> None:
    if not variables:
        logger.info(f"{title}: no shared variables to plot")
        return
    n_var = len(variables)
    fig = plt.figure(figsize=(13, 3.0 * n_var))
    fig.suptitle(
        f"{title}  (cycle {ANALYSIS_TIME.isoformat()} UTC)",
        fontsize=14,
        y=1.0,
    )

    for row, var in enumerate(variables):
        ufs_obs, ufs_lat, ufs_lon = _values(ufs_df, var)
        nnja_obs, nnja_lat, nnja_lon = _values(nnja_df, var)

        # Histogram (left).
        ax_hist = fig.add_subplot(n_var, 2, 2 * row + 1)
        bins = _shared_bins(ufs_obs, nnja_obs)
        if ufs_obs.size:
            ax_hist.hist(
                ufs_obs,
                bins=bins,
                density=True,
                alpha=0.45,
                color="C0",
                label=f"UFS (n={ufs_obs.size:,})",
            )
        if nnja_obs.size:
            ax_hist.hist(
                nnja_obs,
                bins=bins,
                density=True,
                alpha=0.45,
                color="C3",
                label=f"NNJA (n={nnja_obs.size:,})",
            )
        ax_hist.set_yscale("log")
        ax_hist.set_title(f"{var}: observation distribution")
        ax_hist.set_xlabel("observation value")
        ax_hist.set_ylabel("density (log)")
        ax_hist.grid(alpha=0.3)
        ax_hist.legend(fontsize=8, loc="upper right")

        # Coverage (right).
        ax_map = fig.add_subplot(n_var, 2, 2 * row + 2, projection=PROJ)
        ax_map.set_global()
        ax_map.coastlines(linewidth=0.4)
        ax_map.gridlines(linewidth=0.3, alpha=0.4)
        ufs_obs_s, ufs_lat_s, ufs_lon_s = _subsample(
            ufs_obs, ufs_lat, ufs_lon, SUBSAMPLE_FOR_MAP
        )
        nnja_obs_s, nnja_lat_s, nnja_lon_s = _subsample(
            nnja_obs, nnja_lat, nnja_lon, SUBSAMPLE_FOR_MAP
        )
        if ufs_lat_s.size:
            ax_map.scatter(
                ufs_lon_s,
                ufs_lat_s,
                s=2,
                color="C0",
                alpha=0.4,
                transform=ccrs.PlateCarree(),
                label="UFS",
            )
        if nnja_lat_s.size:
            ax_map.scatter(
                nnja_lon_s,
                nnja_lat_s,
                s=2,
                color="C3",
                alpha=0.4,
                transform=ccrs.PlateCarree(),
                label="NNJA",
            )
        ax_map.set_title(f"{var}: coverage (subsample <= {SUBSAMPLE_FOR_MAP:,})")
        if ufs_lat_s.size or nnja_lat_s.size:
            ax_map.legend(fontsize=8, loc="lower left")

    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


_plot_kind(
    ufs_conv_df,
    nnja_conv_df,
    SHARED_CONV_VARS,
    "UFS vs NNJA -- conventional",
    "outputs/05_ufs_vs_nnja_conv.jpg",
)


# %%
# GPS-RO: side-by-side, not overlaid
# ----------------------------------
# NNJA and UFS expose GPS-RO observations with fundamentally different
# semantics (raw BUFR at impact-parameter levels vs GSI-thinned values
# on pressure levels), so overlaying their histograms would be
# misleading. Instead, render two separate histograms and two separate
# coverage maps per variable.

# %%
def _plot_gpsro(
    ufs_df: pd.DataFrame,
    nnja_df: pd.DataFrame,
    variables: list[str],
    title: str,
    out_path: str,
) -> None:
    if not variables:
        logger.info(f"{title}: no shared GPS-RO variables to plot")
        return
    n_var = len(variables)
    # Four columns per variable: UFS hist | NNJA hist | UFS map | NNJA map
    fig = plt.figure(figsize=(20, 3.2 * n_var))
    fig.suptitle(
        f"{title}  (cycle {ANALYSIS_TIME.isoformat()} UTC)",
        fontsize=14,
        y=1.0,
    )

    for row, var in enumerate(variables):
        ufs_obs, ufs_lat, ufs_lon = _values(ufs_df, var)
        nnja_obs, nnja_lat, nnja_lon = _values(nnja_df, var)

        ax_ufs_h = fig.add_subplot(n_var, 4, 4 * row + 1)
        if ufs_obs.size:
            ax_ufs_h.hist(ufs_obs, bins=60, color="C0", alpha=0.7)
        ax_ufs_h.set_yscale("log")
        ax_ufs_h.set_title(f"UFS {var} (n={ufs_obs.size:,})")
        ax_ufs_h.set_xlabel("observation value")
        ax_ufs_h.set_ylabel("count (log)")
        ax_ufs_h.grid(alpha=0.3)

        ax_nnja_h = fig.add_subplot(n_var, 4, 4 * row + 2)
        if nnja_obs.size:
            ax_nnja_h.hist(nnja_obs, bins=60, color="C3", alpha=0.7)
        ax_nnja_h.set_yscale("log")
        ax_nnja_h.set_title(f"NNJA {var} (n={nnja_obs.size:,})")
        ax_nnja_h.set_xlabel("observation value")
        ax_nnja_h.set_ylabel("count (log)")
        ax_nnja_h.grid(alpha=0.3)

        ax_ufs_m = fig.add_subplot(n_var, 4, 4 * row + 3, projection=PROJ)
        ax_ufs_m.set_global()
        ax_ufs_m.coastlines(linewidth=0.4)
        ax_ufs_m.gridlines(linewidth=0.3, alpha=0.4)
        ufs_s, ufs_lat_s, ufs_lon_s = _subsample(
            ufs_obs, ufs_lat, ufs_lon, SUBSAMPLE_FOR_MAP
        )
        if ufs_lat_s.size:
            ax_ufs_m.scatter(
                ufs_lon_s,
                ufs_lat_s,
                s=3,
                color="C0",
                alpha=0.5,
                transform=ccrs.PlateCarree(),
            )
        ax_ufs_m.set_title(f"UFS {var}: coverage")

        ax_nnja_m = fig.add_subplot(n_var, 4, 4 * row + 4, projection=PROJ)
        ax_nnja_m.set_global()
        ax_nnja_m.coastlines(linewidth=0.4)
        ax_nnja_m.gridlines(linewidth=0.3, alpha=0.4)
        nnja_s, nnja_lat_s, nnja_lon_s = _subsample(
            nnja_obs, nnja_lat, nnja_lon, SUBSAMPLE_FOR_MAP
        )
        if nnja_lat_s.size:
            ax_nnja_m.scatter(
                nnja_lon_s,
                nnja_lat_s,
                s=3,
                color="C3",
                alpha=0.5,
                transform=ccrs.PlateCarree(),
            )
        ax_nnja_m.set_title(f"NNJA {var}: coverage")

    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


_plot_gpsro(
    ufs_gpsro_df,
    nnja_gpsro_df,
    SHARED_GPSRO_VARS,
    "UFS vs NNJA -- GPS-RO (side-by-side, not directly comparable)",
    "outputs/05_ufs_vs_nnja_gpsro.jpg",
)


# %%
# Coverage by PrepBUFR observation type (UFS and NNJA)
# ----------------------------------------------------
# Group the conventional observations by their numeric PrepBUFR
# ``type`` code and render three category-specific maps for *each*
# archive (NNJA first, then UFS) so the same legend can be used to
# eyeball coverage differences between the two sources.

# %%
# PrepBUFR observation-type code -> human-readable description.
# Aligned with NCEP PrepBUFR Table 4 ("Report Types"):
# https://emc.ncep.noaa.gov/mmb/data_processing/prepbufr.doc/table_4.htm
TYPE_DESC_MAP: dict[int, str] = {
    # -- Mass observations (1xx) --
    120: "Rawinsonde",
    122: "Class-1 auto-launched dropsonde",
    126: "RASS (Radio Acoustic Sounding System)",
    130: "AIREP and PIREP aircraft (manual)",
    131: "AMDAR aircraft",
    132: "Flight-level recon / profile dropsonde",
    133: "MDCRS ACARS aircraft",
    134: "TAMDAR aircraft",
    135: "Canadian AMDAR aircraft",
    180: "Surface marine (ship / buoy / C-MAN)",
    181: "Surface land synoptic / METAR",
    182: "Splash-level dropsonde over ocean",
    183: "Surface marine with missing pressure",
    187: "Surface land METAR (US format)",
    188: "Mesonet land surface",
    # -- Synthetic / bogus --
    210: "Synthetic TC storm-center bogus winds",
    # -- Wind observations (2xx) --
    220: "Rawinsonde winds",
    221: "PIBAL (Pilot balloon) winds",
    223: "NOAA Profiler Network (NPN) winds",
    224: "NEXRAD VAD radar winds",
    227: "Wind profiler (Cooperative Agency Profilers)",
    228: "JMA wind profiler",
    229: "Wind profiler from PILOT/PIBAL bulletins",
    230: "AIREP and PIREP aircraft winds",
    231: "AMDAR aircraft winds",
    232: "Flight-level recon / profile dropsonde winds",
    233: "MDCRS ACARS aircraft winds",
    234: "TAMDAR aircraft winds",
    235: "Canadian AMDAR aircraft winds",
    # -- Satellite-derived AMV winds (24x-26x) --
    240: "NESDIS GOES IR (LW) cloud-drift AMV",
    241: "INSAT (India) satellite-derived AMV",
    242: "JMA visible cloud-drift AMV",
    243: "EUMETSAT visible cloud-drift AMV",
    244: "AVHRR polar AMV (NESDIS)",
    245: "NESDIS GOES IR (LW) AMV (alternate)",
    246: "NESDIS GOES water-vapor cloud-top AMV",
    247: "NESDIS GOES water-vapor clear-sky AMV",
    248: "NESDIS GOES picture-triplet visible AMV",
    250: "JMA water-vapor cloud-top AMV",
    251: "NESDIS GOES visible cloud-drift AMV",
    252: "JMA IR (LW) cloud-drift AMV",
    253: "EUMETSAT IR (LW) cloud-drift AMV",
    254: "EUMETSAT water-vapor AMV",
    257: "MODIS (Aqua/Terra) IR cloud-drift AMV",
    258: "MODIS (Aqua/Terra) WV cloud-top AMV",
    259: "MODIS (Aqua/Terra) WV deep-layer AMV",
    260: "VIIRS polar IR AMV",
    # -- Surface-wind types (28x-29x) --
    280: "Surface marine winds (ship / buoy / C-MAN)",
    281: "Surface land synoptic / METAR winds",
    282: "ATLAS buoy winds (TAO / PIRATA / RAMA)",
    283: "SSM/I superobed wind speed",
    284: "Surface marine winds with missing pressure",
    287: "Surface land METAR winds",
    288: "Mesonet land surface winds",
    289: "SSM/I neural-net wind speed",
    290: "ASCAT scatterometer winds",
    291: "QuikSCAT scatterometer winds",
    296: "RapidSCAT scatterometer winds",
}

# Each category lists the ``type`` codes to keep from the conventional
# dataframes (NNJA and UFS share the same PrepBUFR ``type`` schema).
CATEGORY_DEFS: list[dict[str, object]] = [
    {
        "name": "upper_air_aircraft",
        "title": "Upper Air & Aircraft Observations",
        "types": [120, 126, 130, 131, 133, 135, 220, 221, 224, 229, 230, 231, 233, 235],
    },
    {
        "name": "surface_marine",
        "title": "Surface & Marine Observations",
        "types": [180, 181, 183, 187, 280, 281, 282, 284, 287, 290],
    },
    {
        "name": "satellite_synthetic",
        "title": "Satellite & Synthetic Observations",
        "types": [
            210,
            240, 245, 246, 247, 248, 251,
            242, 252,
            243, 253, 254,
            250,
            244, 260,
            257, 258, 259,
        ],
    },
]


def _coords_for_type(df: pd.DataFrame, type_code: int) -> tuple[np.ndarray, np.ndarray]:
    if df.empty or "type" not in df.columns:
        return np.empty(0), np.empty(0)
    sub = df[df["type"] == type_code].dropna(subset=["lat", "lon"])
    if sub.empty:
        return np.empty(0), np.empty(0)
    return (
        sub["lon"].to_numpy(dtype=np.float64),
        sub["lat"].to_numpy(dtype=np.float64),
    )


def _plot_category(
    df: pd.DataFrame,
    archive: str,
    title: str,
    types: list[int],
    out_path: str,
) -> None:
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 1, 1, projection=PROJ)
    ax.set_global()
    ax.coastlines(linewidth=0.4)
    ax.gridlines(linewidth=0.3, alpha=0.4)

    # Categorical palette big enough to cover any single category.
    palette = (
        list(plt.get_cmap("tab10").colors)
        + list(plt.get_cmap("Dark2").colors)
        + list(plt.get_cmap("Set2").colors)
    )

    plotted_any = False
    for color_idx, type_code in enumerate(types):
        lon, lat = _coords_for_type(df, type_code)
        if lon.size == 0:
            continue
        if lon.size > SUBSAMPLE_FOR_MAP:
            sel = RNG.choice(lon.size, size=SUBSAMPLE_FOR_MAP, replace=False)
            lon, lat = lon[sel], lat[sel]
        desc = TYPE_DESC_MAP.get(type_code, f"Type {type_code}")
        ax.scatter(
            lon,
            lat,
            s=6,
            color=palette[color_idx % len(palette)],
            alpha=0.6,
            transform=ccrs.PlateCarree(),
            label=f"{type_code}: {desc}",
        )
        plotted_any = True

    ax.set_title(f"{archive} -- {title}", fontsize=13)

    if not plotted_any:
        logger.info(f"{archive} {title}: no rows in any of {types}; skipping plot")
        plt.close(fig)
        return

    legend = ax.legend(
        title="Observation Types",
        fontsize=12,
        title_fontsize=14,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        markerscale=3.5,
        framealpha=0.9,
        borderaxespad=0.0,
        labelspacing=0.6,
        handletextpad=0.8,
    )
    legend.get_frame().set_edgecolor("0.7")
    legend.get_title().set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


# Render NNJA + UFS for each category so the two archives can be
# compared side-by-side using the same colour mapping per type code.
ARCHIVES: list[tuple[str, str, pd.DataFrame]] = [
    ("nnja", "NNJA", nnja_conv_df),
    ("ufs", "UFS", ufs_conv_df),
]

for _cfg in CATEGORY_DEFS:
    for _slug, _archive, _df in ARCHIVES:
        _plot_category(
            df=_df,
            archive=_archive,
            title=str(_cfg["title"]),
            types=list(_cfg["types"]),  # type: ignore[arg-type]
            out_path=f"outputs/05_{_slug}_{_cfg['name']}.jpg",
        )


# %%
# Notes
# -----
# - The UFS conv archive serves *GSI diagnostic* output, so values have
#   already been QC'd and unit-normalised. NNJA serves the underlying
#   PrepBUFR / GPS-RO archive; unit conversion is performed by
#   :py:class:`earth2studio.lexicon.NNJAObsConvLexicon` modifiers
#   (TOB degC->K, QOB mg/kg->kg/kg, POB hPa->Pa).
# - The NNJA satellite radiance archive (``NNJAObsSat``) is no longer
#   available from the public bucket. For satellite brightness
#   temperatures use :py:class:`earth2studio.data.UFSObsSat`, which
#   serves GSI-diagnostic radiances directly.
# - GPS-RO (``gps``, ``gps_t``, ``gps_q``) is rendered in a separate
#   ``05_ufs_vs_nnja_gpsro.jpg`` figure with side-by-side panels rather
#   than overlays. The two archives are not row-for-row comparable for
#   GPS-RO: NNJA reports raw BUFR bending angle / T / Q at impact-
#   parameter levels (``gpsro::15037``, ``gpsro::12001``,
#   ``gpsro::13001``), while UFS reports the same occultations after
#   GSI thinning, QC and interpolation onto pressure levels.
# - If a variable has zero rows in one archive (e.g. an aircraft-only
#   variable in a small tolerance window), only the present archive is
#   shown for that row.
