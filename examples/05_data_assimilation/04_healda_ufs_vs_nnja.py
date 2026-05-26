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
HealDA: UFS vs NNJA conventional observations
=============================================

Compare HealDA analyses produced from two independent **conventional**
observation archives.

This example runs the :py:class:`earth2studio.models.da.HealDA` data
assimilation model twice for the same analysis cycle, with each run
fed observations from its "matching" archive:

- **UFS run** — :py:class:`earth2studio.data.UFSObsConv` for conv and
  :py:class:`earth2studio.data.UFSObsSat` for satellite. Both stream
  from the UFS GEFS-v13 replay GSI diagnostics (already QC'd, unit-
  normalised, and pre-thinned to ~1 deg).
- **NNJA run** — :py:class:`earth2studio.data.NNJAObsConv` for conv
  plus raw radiances from the single-satellite sources:

  - :py:class:`earth2studio.data.JPSS_ATMS` (NOAA AWS S3, n20 + npp)
  - :py:class:`earth2studio.data.MetOpMHS` (EUMETSAT, metop-a/b/c)
  - :py:class:`earth2studio.data.MetOpAMSUA` (EUMETSAT, metop-a/b/c)

Both analyses are then compared against ERA5 reanalysis fetched through
the Copernicus Climate Data Store (CDS).

In this example you will learn:

- How to swap the conventional observation back-end in a HealDA
  workflow without changing the model or the rest of the pipeline.
- How the same cycle's conventional observations from two different
  curated archives affect the resulting analysis.
- How to use :py:class:`earth2studio.data.CDS` as a reanalysis reference.

.. note::
   The NNJA satellite radiance archive (``NNJAObsSat``) is no longer
   available from the public bucket. The NNJA run therefore uses the
   independent single-satellite raw radiance sources (ATMS on JPSS,
   MHS / AMSU-A on MetOp) instead, while the UFS run keeps using
   ``UFSObsSat`` (the GSI-diag pipeline that paired with ``UFSObsConv``
   during HealDA training).

Prerequisites
-------------
- A free CDS API key in ``~/.cdsapirc`` (see
  https://cds.climate.copernicus.eu/how-to-api) for ERA5.
- Free EUMETSAT credentials in ``EUMETSAT_CONSUMER_KEY`` /
  ``EUMETSAT_CONSUMER_SECRET`` (https://eoportal.eumetsat.int/) for
  MHS and AMSU-A. If unset, the run falls back to ATMS-only satellite
  observations.
"""
# /// script
# dependencies = [
#   "earth2studio[da-healda,data] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "cartopy",
#   "healpy",
# ]
# ///

# %%
# Set Up
# ------
# This example requires the following components:
#
# - Assimilation Model: HealDA :py:class:`earth2studio.models.da.HealDA`.
# - Datasource (UFS run, conv): :py:class:`earth2studio.data.UFSObsConv`.
# - Datasource (UFS run, sat):  :py:class:`earth2studio.data.UFSObsSat`.
# - Datasource (NNJA run, conv): :py:class:`earth2studio.data.NNJAObsConv`.
# - Datasource (NNJA run, sat):  JPSS_ATMS + MetOpMHS + MetOpAMSUA.
# - Reference: :py:class:`earth2studio.data.CDS` ERA5 reanalysis.
#
# The (-21h, +3h) observation window matches the time bracket HealDA was
# trained on for the UFS replay archive. The NNJA archive uses the same
# convention (6-hourly GDAS cycle files), so the same window applies.

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()

from datetime import timedelta

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

import healpy as hp
import pandas as pd

from earth2studio.data import (
    CDS,
    JPSS_ATMS,
    MetOpAMSUA,
    MetOpMHS,
    NNJAObsConv,
    UFSObsConv,
    UFSObsSat,
    fetch_dataframe,
)
from earth2studio.lexicon import NNJAObsConvLexicon
from earth2studio.models.da import HealDA

# Load the default HealDA package and regrid to a regular lat-lon grid.
# IMPORTANT: keep the model on CPU during all observation fetching.  The NNJA
# conv decoder spawns BUFR worker processes with ``ProcessPoolExecutor`` (which
# defaults to ``fork`` on Linux); forking after CUDA has been initialised in
# the parent leaves the workers with a corrupt driver state and produces a
# native segfault when they exit.  We move the model to CUDA after fetching.
package = HealDA.load_default_package()
model = HealDA.load_model(package, lat_lon=True)

conv_schema, sat_schema = model.input_coords()
required_conv_vars = list(conv_schema["variable"])
required_sat_vars = list(sat_schema["variable"])
conv_fields = list(conv_schema.keys())
sat_fields = list(sat_schema.keys())


def _intersect(requested: list[str], lexicon_vocab: dict) -> list[str]:
    """Drop variable ids that are not present in a lexicon, with a warning."""
    available = [v for v in requested if v in lexicon_vocab]
    missing = sorted(set(requested) - set(available))
    if missing:
        logger.warning(f"Skipping unsupported variables: {missing}")
    return available


def thin_to_healpix(
    df: pd.DataFrame,
    nside: int = 64,
    max_per_group: int = 20,
    max_total: int = 3_000_000,
) -> pd.DataFrame:
    """Spatially thin observations to ~1 deg, matching HealDA's training density.

    Per (HEALPix pixel, variable, channel, satellite) group, keeps the first
    ``max_per_group`` rows after sorting by quality ascending so the best-quality
    observations within each spatial cell are retained. UFS Replay serves
    pre-thinned satellite data (~2M ATMS, ~500k MHS, ~1.8M AMSU-A per cycle)
    while the raw single-satellite sources return full-resolution radiances
    (100M+ rows per cycle). Defaults here match UFS density - going much
    above that pushes the HealDA scatter aggregator past the bucket density
    it was trained on and trips a CUDA scatter assertion.

    Memory-frugal: works in float32 throughout, attaches the pixel column with
    ``assign`` (no extra full-frame copy), and bails out of the groupby for
    frames already below the per-group budget.
    """
    if len(df) == 0:
        return df

    lat = np.clip(df["lat"].to_numpy(dtype=np.float32, copy=False), -90.0, 90.0)
    lon = df["lon"].to_numpy(dtype=np.float32, copy=False) % np.float32(360.0)
    pix = hp.ang2pix(
        nside,
        np.deg2rad(90.0 - lat, dtype=np.float32),
        np.deg2rad(lon, dtype=np.float32),
    )
    del lat, lon

    # Sort ascending so head() keeps the best-quality obs within each cell.
    # Convention (all sources): 0 = best/clean, higher = progressively worse.
    #   - PrepBUFR QMs (NNJA conv): 0 = no QC issue, 2 = confirmed good,
    #     3 = questionable, 13-15 = bad/missing.
    #   - Satellite bitmasks (ATMS, MetOp MHS/AMSU-A): 0 = no flags set,
    #     non-zero bits = quality flags raised.
    #   - UFS sources carry no quality column (GSI data is already QC'd).
    if "quality" in df.columns:
        order = np.argsort(df["quality"].to_numpy(), kind="stable")
        df = df.iloc[order]
        pix = pix[order]
        del order

    df = df.assign(_hpx_pix=pix)
    del pix

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
    del df

    if len(thinned) > max_total:
        logger.warning(
            f"Thinned obs ({len(thinned):,}) exceed budget ({max_total:,}); "
            f"random subsampling."
        )
        thinned = thinned.sample(n=max_total, random_state=42).reset_index(drop=True)
    return thinned


def filter_trained_channels(df: pd.DataFrame, model: HealDA) -> pd.DataFrame:
    """Drop rows whose ``channel_index`` is not in the model's training set.

    Raw satellite BUFR sources (ATMS, MetOp MHS/AMSU-A) return every physical
    channel of each instrument, but HealDA was only trained on a subset.  The
    UFS GSI pipeline pre-filters to those trained channels; the raw sources do
    not.  This helper uses the model's ``raw_to_local`` table (``raw_to_local
    == 0`` marks an untrained channel) to drop unrecognised channels per
    sensor before passing the frame to the model, which avoids a negative
    ``local_channel`` and the resulting GPU scatter out-of-bounds.
    """
    if "channel_index" not in df.columns or "variable" not in df.columns:
        return df

    keep_parts: list[pd.DataFrame] = []
    for sensor, sensor_df in df.groupby("variable", sort=False):
        if sensor not in model._sensor_stats:
            logger.warning(
                f"  filter_trained_channels: sensor={sensor!r} not in model "
                f"stats (known: {sorted(model._sensor_stats.keys())}); "
                f"passing {len(sensor_df):,} rows through unfiltered"
            )
            keep_parts.append(sensor_df)
            continue
        raw_to_local = np.asarray(model._sensor_stats[sensor]["raw_to_local"])
        raw_ch = sensor_df["channel_index"].values.astype(int)
        ch_min = int(raw_ch.min()) if len(raw_ch) else -1
        ch_max = int(raw_ch.max()) if len(raw_ch) else -1
        trained_raw_ids = np.flatnonzero(raw_to_local > 0).tolist()
        logger.info(
            f"  filter_trained_channels[{sensor}]: incoming channel_index "
            f"range=[{ch_min}, {ch_max}], unique={sorted(set(raw_ch.tolist()))[:20]} "
            f"(showing up to 20); model trained channels (raw ids)={trained_raw_ids}"
        )
        max_raw = len(raw_to_local) - 1
        valid = (raw_ch >= 0) & (raw_ch <= max_raw) & (raw_to_local[np.clip(raw_ch, 0, max_raw)] > 0)
        n_dropped = int((~valid).sum())
        if n_dropped:
            dropped = sorted(set(raw_ch[~valid].tolist()))
            logger.info(
                f"  {sensor}: dropping {n_dropped:,} obs on untrained "
                f"channel(s) {dropped}"
            )
        keep_parts.append(sensor_df.loc[valid])

    out = pd.concat(keep_parts, ignore_index=True)
    out.attrs = df.attrs
    return out


def _purge_corrupt_metop_cache(subdir: str, min_bytes: int = 100_000) -> None:
    """Delete obviously-truncated ``.nat`` files from a MetOp cache directory.

    The MetOp Native format begins with a MPHR ASCII header that declares
    ``TOTAL_FILE_SIZE = NNNNNNNNNN``.  If a previous run was killed mid-download
    (e.g. by a segfault elsewhere in the pipeline) the cached ``.nat`` is
    partial; the source has no integrity check and will re-use it forever,
    deterministically crashing the binary parser with bizarre errors such as
    ``ValueError: input operand has more dimensions than allowed``.

    This helper scans the cache once and deletes any file that is either
    suspiciously small or smaller than its declared ``TOTAL_FILE_SIZE``.
    """
    from earth2studio.data.utils import datasource_cache_root

    cache_dir = os.path.join(datasource_cache_root(), subdir)
    if not os.path.isdir(cache_dir):
        return

    removed = 0
    for name in os.listdir(cache_dir):
        if not name.endswith(".nat"):
            continue
        path = os.path.join(cache_dir, name)
        try:
            actual = os.path.getsize(path)
            if actual < min_bytes:
                os.remove(path)
                removed += 1
                continue
            # MPHR is ASCII at the very start; TOTAL_FILE_SIZE appears within
            # the first few KB.  Peek at 8 KB which always covers it.
            with open(path, "rb") as f:
                head = f.read(8192)
            declared = None
            for line in head.splitlines():
                if line.startswith(b"TOTAL_FILE_SIZE"):
                    try:
                        declared = int(line.split(b"=", 1)[1].strip())
                    except (ValueError, IndexError):
                        declared = None
                    break
            if declared is not None and actual < declared:
                os.remove(path)
                removed += 1
        except OSError:
            continue

    if removed:
        logger.warning(
            "Purged {} truncated .nat file(s) from {} (will be re-downloaded).",
            removed,
            cache_dir,
        )


def fetch_satellite_obs(
    analysis_time: np.ndarray,
    sat_fields: list,
    tolerance: tuple[timedelta, timedelta],
    model: HealDA,
) -> pd.DataFrame:
    """Fetch ATMS + MetOp MHS + MetOp AMSU-A and concatenate into one frame.

    MetOp sources require ``EUMETSAT_CONSUMER_KEY`` / ``EUMETSAT_CONSUMER_SECRET``
    in the environment (free key from https://eoportal.eumetsat.int/). If the
    credentials are missing, we fall back to ATMS-only.

    Each per-sensor frame is filtered to the model's trained channels and
    HEALPix-thinned **before** concatenation so peak memory stays bounded to
    one (raw) source at a time.  Raw ATMS alone can be 100M+ rows.
    """
    # ATMS / MetOp sources use sensor_index where the model expects channel_index.
    # Substitute channel_index → sensor_index in the request so the source returns
    # that column, then rename it back to channel_index after concatenation.
    has_channel_index = "channel_index" in sat_fields
    fields = np.array(
        ["sensor_index" if f == "channel_index" else f for f in sat_fields]
    )

    def _reduce(df: pd.DataFrame, label: str) -> pd.DataFrame:
        """Rename sensor_index → channel_index, drop untrained channels, thin."""
        if has_channel_index and "sensor_index" in df.columns:
            df = df.rename(columns={"sensor_index": "channel_index"})
        n0 = len(df)
        df = filter_trained_channels(df, model)
        n1 = len(df)
        df = thin_to_healpix(df)
        n2 = len(df)
        logger.info(
            f"  {label}: {n0:,} fetched → {n1:,} after channel filter → "
            f"{n2:,} after HEALPix thin"
        )
        return df

    frames: list[pd.DataFrame] = []
    attrs: dict = {}

    logger.info("Fetching ATMS satellite observations from NOAA AWS...")
    atms_df = fetch_dataframe(
        JPSS_ATMS(satellites=["n20", "npp"], time_tolerance=tolerance, async_timeout=3600),
        time=analysis_time,
        variable=np.array(["atms"]),
        fields=fields,
    )
    attrs = dict(atms_df.attrs)
    frames.append(_reduce(atms_df, "ATMS"))
    del atms_df

    if os.environ.get("EUMETSAT_CONSUMER_KEY"):
        _purge_corrupt_metop_cache("metop_mhs")
        _purge_corrupt_metop_cache("metop_amsua")

        # MetOp sources occasionally die on a single corrupt download
        # (truncated .nat from EUMETSAT, transient network/RAM glitch
        # producing bizarre errors like "input operand has more
        # dimensions than allowed by the axis remapping" deep inside
        # the native MHS/AMSU-A parser). Don't kill the whole run for
        # that - log, skip the sensor, and proceed with whatever else
        # we managed to fetch (ATMS alone is enough for the model).
        def _fetch_metop_sensor(
            source_cls, sensor_label: str, variable: str
        ) -> pd.DataFrame | None:
            logger.info(f"Fetching {sensor_label} observations from EUMETSAT...")
            try:
                df = fetch_dataframe(
                    source_cls(time_tolerance=tolerance),
                    time=analysis_time,
                    variable=np.array([variable]),
                    fields=fields,
                )
            except Exception as exc:
                logger.warning(
                    f"{sensor_label} fetch failed ({type(exc).__name__}: {exc}); "
                    f"skipping {sensor_label} for this run."
                )
                return None
            return _reduce(df, sensor_label)

        mhs_df = _fetch_metop_sensor(MetOpMHS, "MHS", "mhs")
        if mhs_df is not None:
            frames.append(mhs_df)
        del mhs_df

        amsua_df = _fetch_metop_sensor(MetOpAMSUA, "AMSU-A", "amsua")
        if amsua_df is not None:
            frames.append(amsua_df)
        del amsua_df
    else:
        logger.warning(
            "EUMETSAT_CONSUMER_KEY not set - MHS / AMSU-A skipped. "
            "Register at https://eoportal.eumetsat.int/ for a free key."
        )

    sat_df = pd.concat(frames, ignore_index=True)
    sat_df.attrs = attrs
    return sat_df


# %%
# Fetch Observations
# ------------------
# Pull conventional and satellite observation DataFrames from each archive
# for the same analysis time. The NNJA conv data source covers all eight
# variables HealDA can ingest (``u, v, q, t, pres, gps, gps_t, gps_q``)
# by routing PrepBUFR variables to ``conv/prepbufr/`` and GPS RO
# variables to ``gps/gpsro/``.

# %%
analysis_time = np.array([np.datetime64("2024-01-01T00:00")])
tolerance = (timedelta(hours=-21), timedelta(hours=3))

# --- UFS run: UFS conv + UFS sat (both from GSI-diag pipeline) -------
ufs_conv_source = UFSObsConv(time_tolerance=tolerance)
ufs_conv_df = fetch_dataframe(
    ufs_conv_source,
    time=analysis_time,
    variable=np.array(required_conv_vars),
    fields=np.array(conv_fields),
)
logger.info(f"Fetched {len(ufs_conv_df):,} UFS conventional observations")

ufs_sat_source = UFSObsSat(time_tolerance=tolerance)
ufs_sat_df = fetch_dataframe(
    ufs_sat_source,
    time=analysis_time,
    variable=np.array(required_sat_vars),
    fields=np.array(sat_fields),
)
# Drop rows with BUFR fill-value sentinel observations (physically > 1000 K).
ufs_sat_df = ufs_sat_df[ufs_sat_df["observation"] < 1000.0].reset_index(drop=True)
logger.info(f"Fetched {len(ufs_sat_df):,} UFS satellite observations")

# --- NNJA run: NNJA conv + MetOp/JPSS single-satellite radiances ----
# As of NNJAObsConvLexicon v1, ``u, v, q, t, pres, gps, gps_t, gps_q``
# are all supported. ``_intersect`` is kept for forward compatibility
# if the lexicon ever falls behind HealDA's required variable list.
nnja_conv_vars = _intersect(required_conv_vars, NNJAObsConvLexicon.VOCAB)

# NNJA cycles are decoded one at a time and merged afterwards.  Decoding all
# 5 cycles in a single fetch_dataframe() call lets NNJAObsConv accumulate
# every cycle's DataFrame in memory until the final concat, which combined
# with the peak of pd.DataFrame(all_rows) on a 4M+ row GPSRO cycle can OOM
# the box.  Per-cycle calls keep peak resident set to ~one cycle's worth.
def _cycle_anchors(t0: np.datetime64, tol: tuple[timedelta, timedelta]) -> list[np.datetime64]:
    """List of 6-hourly NNJA cycle datetimes covered by ``t0 + tol``."""
    base = pd.Timestamp(t0).to_pydatetime()
    lo = base + tol[0]
    hi = base + tol[1]
    start = lo.replace(minute=0, second=0, microsecond=0)
    start = start.replace(hour=(start.hour // 6) * 6)
    out: list[np.datetime64] = []
    cur = start
    while cur <= hi:
        out.append(np.datetime64(cur))
        cur += timedelta(hours=6)
    return out


import gc
import tempfile

# Canonical NCEP convention: each 6-hourly cycle file ``cHHz`` carries obs
# in ``[c-3h, c+3h)``.  We fetch one cycle per call (the library internally
# may read the previous cycle's file too but only emits rows in the window
# below); to keep peak memory bounded we spill each cycle's frame to a
# Parquet shard immediately and reload them all at the end.
nnja_conv_source = NNJAObsConv(
    time_tolerance=(timedelta(hours=-3), timedelta(hours=3)),
    # 8 parallel ``pybufrkit`` workers, each holding hundreds of thousands of
    # in-flight Python dicts, push total RSS past the limit on a busy machine
    # and the C decoder segfaults non-deterministically.  GPSRO subsets are
    # large (~1700 rows each) so the headroom matters.  4 workers is the
    # sweet spot: ~2x slower than 8 but half the peak memory, which keeps
    # the cumulative cycle-by-cycle pipeline well below the OOM threshold.
    decode_workers=8,
)
nnja_cycle_files: list[str] = []
nnja_shard_dir = tempfile.mkdtemp(prefix="nnja_conv_shards_")
# Drop the first anchor: with canonical ±3h windows the leftmost anchor
# only contributes its lower half [c-3h, t-21h), which lies *outside* the
# UFS observation window [t-21h, t+3h].  Skipping it keeps NNJA conv
# coverage exactly aligned with the UFS run.
_anchors = _cycle_anchors(analysis_time[0], tolerance)[1:]
for cycle in _anchors:
    cycle_df = fetch_dataframe(
        nnja_conv_source,
        time=np.array([cycle]),
        variable=np.array(nnja_conv_vars),
        fields=np.array(conv_fields),
    )
    logger.info(
        f"  NNJA cycle {pd.Timestamp(cycle):%Y-%m-%d %HZ}: "
        f"{len(cycle_df):,} obs"
    )
    if len(cycle_df):
        shard_path = os.path.join(
            nnja_shard_dir, f"cycle_{pd.Timestamp(cycle):%Y%m%d_%H}.parquet"
        )
        # ``df.attrs`` may contain numpy arrays (e.g. ``request_time``) which
        # pandas tries to JSON-serialize into the parquet metadata.  Clear it.
        cycle_df.attrs = {}
        cycle_df.to_parquet(shard_path, index=False)
        nnja_cycle_files.append(shard_path)
    del cycle_df
    gc.collect()

nnja_conv_df = (
    pd.concat([pd.read_parquet(p) for p in nnja_cycle_files], ignore_index=True)
    if nnja_cycle_files
    else pd.DataFrame()
)
for p in nnja_cycle_files:
    try:
        os.remove(p)
    except OSError:
        pass
try:
    os.rmdir(nnja_shard_dir)
except OSError:
    pass
nnja_conv_df.attrs = {"request_time": analysis_time}
logger.info(f"Fetched {len(nnja_conv_df):,} NNJA conventional observations")

nnja_sat_df = fetch_satellite_obs(analysis_time, sat_fields, tolerance, model)
nnja_sat_df.attrs = {"request_time": analysis_time}
logger.info(
    f"Fetched {len(nnja_sat_df):,} NNJA-run satellite observations "
    f"(ATMS + MHS + AMSU-A, thinned to ~1 deg)"
)

# %%
# Compare Observation Sources
# ---------------------------
# Before running HealDA, compare the UFS and NNJA observation archives directly:
# a per-variable count/stats table and 2×2 global density maps (conv + sat).

# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


def to_numpy(arr):
    """CuPy / NumPy helper."""
    return arr.get() if hasattr(arr, "get") else arr


def _obs_summary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for var, grp in df.groupby("variable", sort=True):
        obs = grp["observation"]
        rows.append(
            {
                "source": label,
                "variable": var,
                "count": len(grp),
                "mean": float(obs.mean()),
                "std": float(obs.std()),
                "min": float(obs.min()),
                "max": float(obs.max()),
            }
        )
    return pd.DataFrame(rows)


def _density_ax(
    ax, df: pd.DataFrame, title: str, cmap: str = "YlOrRd"
) -> None:
    lats = df["lat"].values.astype(np.float32)
    lons = (df["lon"].values.astype(np.float32) % 360.0) - 180.0
    counts, lat_edges, lon_edges = np.histogram2d(
        lats, lons, bins=[90, 180], range=[[-90, 90], [-180, 180]]
    )
    lon_c = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_c = (lat_edges[:-1] + lat_edges[1:]) / 2
    LON, LAT = np.meshgrid(lon_c, lat_c)
    im = ax.pcolormesh(
        LON,
        LAT,
        np.log1p(counts),
        transform=ccrs.PlateCarree(),
        cmap=cmap,
    )
    ax.coastlines(linewidth=0.4)
    ax.set_global()
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.55, label="log(1+count) per 2° cell")


conv_summary = pd.concat(
    [_obs_summary(ufs_conv_df, "UFS"), _obs_summary(nnja_conv_df, "NNJA")],
    ignore_index=True,
).sort_values(["variable", "source"])
logger.info(
    "Conventional obs summary (UFS vs NNJA):\n" + conv_summary.to_string(index=False)
)

sat_summary = pd.concat(
    [_obs_summary(ufs_sat_df, "UFS"), _obs_summary(nnja_sat_df, "NNJA")],
    ignore_index=True,
).sort_values(["variable", "source"])
logger.info(
    "Satellite obs summary (UFS vs NNJA):\n" + sat_summary.to_string(index=False)
)

projection = ccrs.Robinson()
fig, axes = plt.subplots(
    2,
    2,
    subplot_kw={"projection": projection},
    figsize=(18, 8),
)
fig.subplots_adjust(wspace=0.06, hspace=0.12, top=0.90)
_density_ax(axes[0, 0], ufs_conv_df, "Conv — UFS")
_density_ax(axes[0, 1], nnja_conv_df, "Conv — NNJA")
_density_ax(axes[1, 0], ufs_sat_df, "Sat — UFS")
_density_ax(axes[1, 1], nnja_sat_df, "Sat — NNJA (thinned)")
fig.suptitle(
    f"Observation density  {str(analysis_time[0])[:16]} UTC",
    fontsize=14,
)
plt.savefig("outputs/04_obs_density_ufs_vs_nnja.jpg", dpi=150, bbox_inches="tight")
plt.close(fig)
logger.info("Wrote outputs/04_obs_density_ufs_vs_nnja.jpg")


# %%
# GPS RO Comparison (UFS vs NNJA)
# -------------------------------
# Side-by-side look at the GPS bending-angle observations: spatial
# density, vertical distribution by impact parameter, and the
# observation-value histogram. ``gps`` is the bending angle (rad);
# ``elev`` carries the impact parameter (m) for the bending-angle
# subprofile (set by the GPSRO decoder).

# %%
def _plot_gps_compare(
    ufs_df: pd.DataFrame,
    nnja_df: pd.DataFrame,
    out_path: str,
) -> None:
    """Compare UFS vs NNJA GPS RO observations.

    Layout (2×2):
        (0,0) UFS GPS spatial density
        (0,1) NNJA GPS spatial density
        (1,0) Vertical distribution (count vs impact parameter)
        (1,1) Bending-angle value histogram
    """
    ufs_g = ufs_df[ufs_df["variable"] == "gps"]
    nnja_g = nnja_df[nnja_df["variable"] == "gps"]
    if len(ufs_g) == 0 and len(nnja_g) == 0:
        logger.warning("No GPS observations in either source - skipping plot")
        return

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.0, 0.85], hspace=0.30, wspace=0.18
    )

    ax_ufs = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson())
    ax_nnja = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson())
    _density_ax(ax_ufs, ufs_g, f"GPS — UFS ({len(ufs_g):,} obs)", cmap="Blues")
    _density_ax(ax_nnja, nnja_g, f"GPS — NNJA ({len(nnja_g):,} obs)", cmap="Reds")

    # Vertical distribution: count vs impact parameter (km).
    ax_vert = fig.add_subplot(gs[1, 0])
    elev_km_ufs = ufs_g["elev"].to_numpy(dtype=np.float32) / 1000.0
    elev_km_nnja = nnja_g["elev"].to_numpy(dtype=np.float32) / 1000.0
    elev_km_ufs = elev_km_ufs[np.isfinite(elev_km_ufs)]
    elev_km_nnja = elev_km_nnja[np.isfinite(elev_km_nnja)]
    bins = np.linspace(0, 80, 81)  # 1 km bins, 0-80 km
    if len(elev_km_ufs):
        ax_vert.hist(
            elev_km_ufs,
            bins=bins,
            orientation="horizontal",
            histtype="step",
            color="C0",
            linewidth=1.4,
            label=f"UFS (n={len(elev_km_ufs):,})",
        )
    if len(elev_km_nnja):
        ax_vert.hist(
            elev_km_nnja,
            bins=bins,
            orientation="horizontal",
            histtype="step",
            color="C3",
            linewidth=1.4,
            label=f"NNJA (n={len(elev_km_nnja):,})",
        )
    ax_vert.set_xlabel("count per 1 km bin")
    ax_vert.set_ylabel("impact parameter (km)")
    ax_vert.set_title("Vertical distribution")
    ax_vert.set_xscale("log")
    ax_vert.grid(True, alpha=0.3)
    ax_vert.legend(loc="upper right", fontsize=9)

    # Bending-angle value histogram.
    ax_val = fig.add_subplot(gs[1, 1])
    val_ufs = ufs_g["observation"].to_numpy(dtype=np.float64)
    val_nnja = nnja_g["observation"].to_numpy(dtype=np.float64)
    val_ufs = val_ufs[np.isfinite(val_ufs)]
    val_nnja = val_nnja[np.isfinite(val_nnja)]
    # Use combined range, clipped to physical bending-angle scale.
    combined = np.concatenate([val_ufs, val_nnja])
    if combined.size:
        lo = max(float(np.nanmin(combined)), -0.005)
        hi = min(float(np.nanmax(combined)), 0.06)
        bins_val = np.linspace(lo, hi, 81)
        if len(val_ufs):
            ax_val.hist(
                val_ufs,
                bins=bins_val,
                histtype="step",
                color="C0",
                linewidth=1.4,
                label=f"UFS  μ={val_ufs.mean():.2e}  σ={val_ufs.std():.2e}",
            )
        if len(val_nnja):
            ax_val.hist(
                val_nnja,
                bins=bins_val,
                histtype="step",
                color="C3",
                linewidth=1.4,
                label=f"NNJA μ={val_nnja.mean():.2e}  σ={val_nnja.std():.2e}",
            )
    ax_val.set_xlabel("bending angle (rad)")
    ax_val.set_ylabel("count")
    ax_val.set_title("Observation-value distribution")
    ax_val.set_yscale("log")
    ax_val.grid(True, alpha=0.3)
    ax_val.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"GPS RO bending angle — UFS vs NNJA   "
        f"{str(analysis_time[0])[:16]} UTC",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


_plot_gps_compare(
    ufs_conv_df, nnja_conv_df, "outputs/04_gps_ufs_vs_nnja.jpg"
)


# %%
# Run HealDA twice
# ----------------
# Each run is stateless and fully determined by the input observations.

# HealDA's obs embedder converts (lat, lon) to a HEALPix pixel index via
# ``self._grid.ang2pix(lon, lat).int()`` with no NaN / range guard.  Any
# row with non-finite or out-of-range coordinates therefore produces an
# out-of-bounds pixel index and trips a CUDA device-side assertion deep
# inside the scatter aggregator.  The UFS GSI pipeline is pre-QC'd so
# this never bites, but the raw NNJA decoders (especially GPSRO) can
# emit such rows.  Filter them out before handing observations to the
# model.  We apply the same guard to UFS as a cheap safety net.


# %%
def _drop_bad_coords(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Drop rows with non-finite or out-of-range ``lat`` / ``lon``."""
    if (
        df is None
        or len(df) == 0
        or "lat" not in df.columns
        or "lon" not in df.columns
    ):
        return df
    lat = df["lat"].to_numpy(dtype=np.float64, copy=False)
    lon = df["lon"].to_numpy(dtype=np.float64, copy=False)
    ok = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (lat >= -90.0)
        & (lat <= 90.0)
    )
    n_drop = int((~ok).sum())
    if n_drop == 0:
        return df
    bad_lat = lat[~ok]
    bad_lon = lon[~ok]
    logger.warning(
        f"{label}: dropping {n_drop:,}/{len(df):,} rows with bad coords "
        f"(lat range=[{np.nanmin(bad_lat):.3f}, {np.nanmax(bad_lat):.3f}], "
        f"lon range=[{np.nanmin(bad_lon):.3f}, {np.nanmax(bad_lon):.3f}])"
    )
    out = df.loc[ok].reset_index(drop=True)
    out.attrs = dict(df.attrs)
    return out


ufs_conv_df = _drop_bad_coords(ufs_conv_df, "UFS conv")
ufs_sat_df = _drop_bad_coords(ufs_sat_df, "UFS sat")
nnja_conv_df = _drop_bad_coords(nnja_conv_df, "NNJA conv")
nnja_sat_df = _drop_bad_coords(nnja_sat_df, "NNJA sat")

# %%
# All BUFR worker pools are done by this point; safe to initialise CUDA.
model = model.to("cuda:0")

torch.manual_seed(42)
result_ufs = model(conv_obs=ufs_conv_df, sat_obs=ufs_sat_df)
logger.info(f"UFS analysis shape:  {result_ufs.shape}")

torch.manual_seed(42)
result_nnja = model(conv_obs=nnja_conv_df, sat_obs=nnja_sat_df)
logger.info(f"NNJA analysis shape: {result_nnja.shape}")


# %%
# Fetch ERA5 reference from CDS
# -----------------------------
# Both analyses are compared against ERA5 reanalysis pulled from the
# Copernicus Climate Data Store. CDS returns a regular 0.25-degree
# lat-lon grid which we interpolate to the HealDA output grid for a
# direct pixel-wise difference.

# %%
plot_vars = ["t2m", "z500"]

cds_ds = CDS(cache=True, verbose=True)
era5_da = cds_ds(analysis_time, plot_vars)

lat = result_ufs.coords["lat"].values
lon = result_ufs.coords["lon"].values
era5_interp = era5_da.interp(lat=lat, lon=lon, method="nearest")


# %%
# Post Processing
# ---------------
# Plot a 3-row figure (UFS-HealDA, NNJA-HealDA, ERA5) for ``t2m`` and
# ``z500``, then a 2x2 figure of differences against ERA5.

# %%
plt.close("all")

projection = ccrs.Robinson()
titles = ["HealDA (UFS)", "HealDA (NNJA)", "ERA5 (CDS)"]
fields = [result_ufs, result_nnja, era5_interp]
cmaps = ["Spectral_r", "PRGn"]

fig, axes = plt.subplots(
    len(fields),
    len(plot_vars),
    subplot_kw={"projection": projection},
    figsize=(14, 9),
)
fig.subplots_adjust(wspace=0.02, hspace=0.08, left=0.08, right=0.94)

for row, (title, da) in enumerate(zip(titles, fields)):
    for col, var in enumerate(plot_vars):
        ax = axes[row, col]
        ax.set_global()
        field = to_numpy(da.sel(variable=var).data[0])
        im = ax.pcolormesh(
            lon,
            lat,
            field,
            transform=ccrs.PlateCarree(),
            cmap=cmaps[col],
        )
        ax.coastlines(linewidth=0.5)
        ax.gridlines(linewidth=0.3, alpha=0.4)
        fig.colorbar(im, ax=ax, shrink=0.6)
        if row == 0:
            ax.set_title(var, fontsize=14)
        if col == 0:
            ax.text(
                -0.05,
                0.5,
                title,
                fontsize=12,
                va="center",
                ha="center",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
            )

fig.suptitle(
    f"HealDA UFS vs NNJA vs ERA5  {str(analysis_time[0])[:16]} UTC",
    fontsize=16,
    y=0.96,
)
plt.savefig("outputs/04_healda_ufs_vs_nnja_analysis.jpg", dpi=150)
plt.close(fig)


# %%
# Difference vs ERA5
# ------------------
# Compute the analysis - ERA5 difference per variable and per source,
# print a global-mean MAE summary, and plot maps.

# %%
diff_titles = ["UFS - ERA5", "NNJA - ERA5"]
diff_results = [result_ufs, result_nnja]

for title, da_pred in zip(diff_titles, diff_results):
    for var in plot_vars:
        f_pred = to_numpy(da_pred.sel(variable=var).data[0])
        f_era5 = to_numpy(era5_interp.sel(variable=var).data[0])
        mae = float(np.abs(f_pred - f_era5).mean())
        logger.info(f"{title} | {var} MAE: {mae:.4f}")

plt.close("all")
diff_ranges = {"t2m": (-10.0, 10.0), "z500": (-500.0, 500.0)}
fig, axes = plt.subplots(
    len(diff_results),
    len(plot_vars),
    subplot_kw={"projection": projection},
    figsize=(14, 6),
)
fig.subplots_adjust(wspace=0.02, hspace=0.08, left=0.08, right=0.94)

for row, (title, da_pred) in enumerate(zip(diff_titles, diff_results)):
    for col, var in enumerate(plot_vars):
        ax = axes[row, col]
        ax.set_global()
        f_pred = to_numpy(da_pred.sel(variable=var).data[0])
        f_era5 = to_numpy(era5_interp.sel(variable=var).data[0])
        diff = f_pred - f_era5
        im = ax.pcolormesh(
            lon,
            lat,
            diff,
            transform=ccrs.PlateCarree(),
            cmap="RdBu_r",
            vmin=diff_ranges[var][0],
            vmax=diff_ranges[var][1],
        )
        ax.coastlines(linewidth=0.5)
        ax.gridlines(linewidth=0.3, alpha=0.4)
        fig.colorbar(im, ax=ax, shrink=0.6)
        if row == 0:
            ax.set_title(var, fontsize=14)
        if col == 0:
            ax.text(
                -0.05,
                0.5,
                title,
                fontsize=12,
                va="center",
                ha="center",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.transAxes,
            )

fig.suptitle(
    f"HealDA - ERA5 differences  {str(analysis_time[0])[:16]} UTC",
    fontsize=16,
    y=0.97,
)
plt.savefig("outputs/04_healda_ufs_vs_nnja_diff.jpg", dpi=150, bbox_inches="tight")
plt.close(fig)
