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
NNJA Observations
=================

Fetching and visualising NOAA-NASA Joint Archive (NNJA) observations.

This example pulls a single 6-hourly cycle of conventional (PrepBUFR) and
satellite (ATMS BUFR) observations from the NOAA-NASA Joint Archive on
AWS S3 and plots their global distribution. NNJA is a curated joint
observation archive maintained by NOAA and NASA spanning 1979 to
present, intended for Earth System reanalyses; see
https://psl.noaa.gov/data/nnja_obs/.

In this example you will learn:

- How to instantiate :py:class:`earth2studio.data.NNJAObsConv` and
  :py:class:`earth2studio.data.NNJAObsSat`.
- Fetching conventional (in-situ) and satellite-radiance observations
  for a given assimilation cycle.
- Plotting global distributions of temperature, wind, and microwave
  brightness temperature on Cartopy maps.
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
# This example requires the following components:
#
# - Datasource (conv): :py:class:`earth2studio.data.NNJAObsConv` for
#   conventional in-situ PrepBUFR observations.
# - Datasource (sat): :py:class:`earth2studio.data.NNJAObsSat` for
#   satellite-radiance BUFR observations.
#
# Both pull from the public AWS S3 bucket
# ``s3://noaa-reanalyses-pds/observations/reanalysis/`` and decode BUFR
# in-process (eccodes for satellite radiance, pybufrkit for PrepBUFR).
#
# .. note::
#   NNJA observation data is distributed under CC BY 4.0; please cite
#   https://psl.noaa.gov/data/nnja_obs/ when using this data.

# %%
import os
from datetime import datetime, timedelta

os.makedirs("outputs", exist_ok=True)

from dotenv import load_dotenv

load_dotenv()

from earth2studio.data import NNJAObsConv, NNJAObsSat

cycle = datetime(2024, 1, 1, 0)

conv_ds = NNJAObsConv(
    source="prepbufr",
    time_tolerance=timedelta(hours=3),
    cache=True,
    verbose=True,
)
sat_ds = NNJAObsSat(
    satellites=["n20"],
    time_tolerance=timedelta(minutes=30),
    cache=True,
    verbose=True,
)

# %%
# Execute the Workflow
# --------------------
# Fetch a single cycle (00z 2024-01-01) of conventional temperature
# and winds, plus ATMS brightness-temperature observations from
# NOAA-20.

# %%
df_t = conv_ds(cycle, ["t"])
df_uv = conv_ds(cycle, ["u", "v"])
df_atms = sat_ds(cycle, ["atms"])

print(f"Temperature observations: {len(df_t):,}")
print(f"Wind (u/v) observations:  {len(df_uv):,}")
print(f"ATMS observations:        {len(df_atms):,}")
print(df_t.head(3))

# %%
# Post Processing
# ---------------
# Plot the global distribution of each observation type on a single
# 2x2 Cartopy figure.
#
# - Top-left: temperature observations colored by value (Kelvin).
# - Top-right: wind speed at observation locations.
# - Bottom-left: ATMS channel-1 brightness temperature (23.8 GHz, surface window).
# - Bottom-right: total observation density (2D histogram of the
#   conventional temperature observation locations).

# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.close("all")


def _maybe_sample(df: pd.DataFrame, max_rows: int = 100_000) -> pd.DataFrame:
    """Cap the number of plotted points for responsiveness."""
    if len(df) > max_rows:
        return df.sample(max_rows, random_state=0)
    return df


projection = ccrs.Robinson()
fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(16, 9),
    subplot_kw={"projection": projection},
)

# Convert PrepBUFR longitudes from [0, 360) to (-180, 180] for plotting
def _to_180(lon: pd.Series) -> pd.Series:
    return ((lon + 180.0) % 360.0) - 180.0


# Top-left: temperature observations
ax = axes[0, 0]
ax.set_global()
df_t_plot = _maybe_sample(df_t)
sc = ax.scatter(
    _to_180(df_t_plot["lon"]),
    df_t_plot["lat"],
    c=df_t_plot["observation"],
    cmap="Spectral_r",
    s=2,
    transform=ccrs.PlateCarree(),
)
ax.coastlines()
ax.gridlines(draw_labels=False, alpha=0.3)
ax.set_title(f"NNJA conventional T  ({len(df_t):,} obs)\n{cycle.isoformat()} UTC")
plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05, label="T (K)")

# Top-right: wind speed at observation locations
ax = axes[0, 1]
ax.set_global()
df_u = df_uv[df_uv["variable"] == "u"]
df_v = df_uv[df_uv["variable"] == "v"]
# Join u/v by (time, lat, lon, pres, station) to compute wind speed at each obs
df_w = (
    df_u[["time", "lat", "lon", "pres", "observation"]]
    .rename(columns={"observation": "u"})
    .merge(
        df_v[["time", "lat", "lon", "pres", "observation"]].rename(
            columns={"observation": "v"}
        ),
        on=["time", "lat", "lon", "pres"],
        how="inner",
    )
)
df_w["speed"] = np.sqrt(df_w["u"] ** 2 + df_w["v"] ** 2)
df_w_plot = _maybe_sample(df_w)
sc = ax.scatter(
    _to_180(df_w_plot["lon"]),
    df_w_plot["lat"],
    c=df_w_plot["speed"],
    cmap="viridis",
    s=2,
    transform=ccrs.PlateCarree(),
)
ax.coastlines()
ax.gridlines(draw_labels=False, alpha=0.3)
ax.set_title(f"NNJA conventional |U|  ({len(df_w):,} obs)\n{cycle.isoformat()} UTC")
plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05, label="|U| (m/s)")

# Bottom-left: ATMS channel-1 brightness temperature
ax = axes[1, 0]
df_ch1 = df_atms[df_atms["channel_index"] == 1]
df_ch1_plot = _maybe_sample(df_ch1)
ax.set_global()
sc = ax.scatter(
    _to_180(df_ch1_plot["lon"]),
    df_ch1_plot["lat"],
    c=df_ch1_plot["observation"],
    cmap="plasma",
    s=2,
    transform=ccrs.PlateCarree(),
)
ax.coastlines()
ax.gridlines(draw_labels=False, alpha=0.3)
ax.set_title(
    f"NNJA NOAA-20 ATMS ch.1 (23.8 GHz)  ({len(df_ch1):,} FOVs)\n"
    f"{cycle.isoformat()} UTC \u00b130min"
)
plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05, label="BT (K)")

# Bottom-right: observation density (2D histogram of temperature obs)
ax = axes[1, 1]
ax.set_global()
H, xedges, yedges = np.histogram2d(
    _to_180(df_t["lon"]).to_numpy(),
    df_t["lat"].to_numpy(),
    bins=[np.linspace(-180, 180, 73), np.linspace(-90, 90, 37)],
)
im = ax.pcolormesh(
    xedges,
    yedges,
    H.T,
    cmap="cividis",
    transform=ccrs.PlateCarree(),
)
ax.coastlines()
ax.gridlines(draw_labels=False, alpha=0.3)
ax.set_title("NNJA conventional T density (5°x5° bins)")
plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, label="obs / bin")

fig.suptitle("NOAA-NASA Joint Archive (NNJA) observations", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97])

plt.savefig("outputs/03_nnja_observations.jpg", dpi=120)
plt.close(fig)
