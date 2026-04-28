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
HealDA: UFS vs NNJA observations
=================================

Compare HealDA analyses produced from two independent observation archives.

This example runs the :py:class:`earth2studio.models.da.HealDA` data
assimilation model twice for the same analysis cycle, once with
observations sourced from the NOAA UFS GEFS-v13 replay archive and once
with observations from the NOAA-NASA Joint Archive (NNJA), and then
compares both analyses against ERA5 reanalysis fetched through the
Copernicus Climate Data Store (CDS).

In this example you will learn:

- How to swap observation back-ends in a HealDA workflow without
  changing the model or the rest of the pipeline.
- How the same cycle's observations from two different curated archives
  affect the resulting analysis.
- How to use :py:class:`earth2studio.data.CDS` as a reanalysis reference.

Prerequisites
-------------
A free CDS API key in ``~/.cdsapirc`` (see
https://cds.climate.copernicus.eu/how-to-api).
"""
# /// script
# dependencies = [
#   "earth2studio[da-healda,data] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "cartopy",
# ]
# ///

# %%
# Set Up
# ------
# This example requires the following components:
#
# - Assimilation Model: HealDA :py:class:`earth2studio.models.da.HealDA`.
# - Datasource (UFS conv): :py:class:`earth2studio.data.UFSObsConv`.
# - Datasource (UFS sat):  :py:class:`earth2studio.data.UFSObsSat`.
# - Datasource (NNJA conv): :py:class:`earth2studio.data.NNJAObsConv`.
# - Datasource (NNJA sat):  :py:class:`earth2studio.data.NNJAObsSat`.
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

from earth2studio.data import (
    CDS,
    NNJAObsConv,
    NNJAObsSat,
    UFSObsConv,
    UFSObsSat,
    fetch_dataframe,
)
from earth2studio.lexicon import NNJAObsConvLexicon, NNJASatelliteLexicon
from earth2studio.models.da import HealDA

# Load the default HealDA package and regrid to a regular lat-lon grid.
package = HealDA.load_default_package()
model = HealDA.load_model(package, lat_lon=True)
model = model.to("cuda:0")

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

# UFS observations
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
logger.info(f"Fetched {len(ufs_sat_df):,} UFS satellite observations")

# NNJA observations.  As of NNJAObsConvLexicon v1, ``u, v, q, t, pres,
# gps, gps_t, gps_q`` are all supported and the satellite lexicon covers
# all four sensors HealDA needs (``atms``, ``amsua``, ``amsub``, ``mhs``).
# ``_intersect`` is kept for forward compatibility if the lexicon ever
# falls behind HealDA's required variable list.
nnja_conv_vars = _intersect(required_conv_vars, NNJAObsConvLexicon.VOCAB)
nnja_sat_vars = _intersect(required_sat_vars, NNJASatelliteLexicon.VOCAB)

nnja_conv_source = NNJAObsConv(time_tolerance=tolerance)
nnja_conv_df = fetch_dataframe(
    nnja_conv_source,
    time=analysis_time,
    variable=np.array(nnja_conv_vars),
    fields=np.array(conv_fields),
)
logger.info(f"Fetched {len(nnja_conv_df):,} NNJA conventional observations")

nnja_sat_source = NNJAObsSat(time_tolerance=tolerance)
nnja_sat_df = fetch_dataframe(
    nnja_sat_source,
    time=analysis_time,
    variable=np.array(nnja_sat_vars),
    fields=np.array(sat_fields),
)
logger.info(f"Fetched {len(nnja_sat_df):,} NNJA satellite observations")

# %%
# Run HealDA twice
# ----------------
# Each run is stateless and fully determined by the input observations.

# %%
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
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

plt.close("all")


def to_numpy(arr):
    """CuPy / NumPy helper."""
    return arr.get() if hasattr(arr, "get") else arr


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
