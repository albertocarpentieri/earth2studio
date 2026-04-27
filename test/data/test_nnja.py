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

import pathlib
import shutil
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from earth2studio.data import NNJAObsConv, NNJAObsSat
from earth2studio.lexicon import NNJAObsConvLexicon, NNJASatelliteLexicon

# ─────────────────────────────────────────────────────────────────────
# Lexicon coverage and modifier tests (offline)
# ─────────────────────────────────────────────────────────────────────


def test_nnja_obs_conv_lexicon_parse():
    """Every NNJAObsConvLexicon entry resolves to a non-empty key plus a callable modifier."""
    for var in NNJAObsConvLexicon.VOCAB:
        key, modifier = NNJAObsConvLexicon[var]
        assert isinstance(key, str)
        assert key
        assert callable(modifier)


def test_nnja_obs_conv_lexicon_modifiers():
    """The conv lexicon modifiers convert raw PrepBUFR units to Earth2Studio standards."""
    # t: degrees C -> Kelvin
    _, mod = NNJAObsConvLexicon["t"]
    df = mod(pd.DataFrame({"observation": [0.0]}))
    assert df["observation"].iloc[0] == pytest.approx(273.15)

    # q: mg/kg -> kg/kg
    _, mod = NNJAObsConvLexicon["q"]
    df = mod(pd.DataFrame({"observation": [1e6]}))
    assert df["observation"].iloc[0] == pytest.approx(1.0)

    # pres: hPa -> Pa
    _, mod = NNJAObsConvLexicon["pres"]
    df = mod(pd.DataFrame({"observation": [1.0]}))
    assert df["observation"].iloc[0] == pytest.approx(100.0)

    # u, v: identity
    for var in ("u", "v"):
        _, mod = NNJAObsConvLexicon[var]
        df = mod(pd.DataFrame({"observation": [3.14]}))
        assert df["observation"].iloc[0] == pytest.approx(3.14)


def test_nnja_satellite_lexicon_parse():
    """Every NNJASatelliteLexicon entry splits into 4 ``::`` parts with non-empty platforms."""
    for var, vocab in NNJASatelliteLexicon.VOCAB.items():
        parts = vocab.split("::")
        assert len(parts) == 4, f"{var}: expected 4 parts, got {len(parts)}"
        sensor, source, platforms_csv, bufr_key = parts
        assert sensor and source and bufr_key
        platforms = platforms_csv.split(",")
        assert all(p.strip() for p in platforms), f"{var}: empty platform"


def test_nnja_satellite_lexicon_identity_modifier():
    """The satellite lexicon modifier is the identity on DataFrames."""
    for var in NNJASatelliteLexicon.VOCAB:
        _, mod = NNJASatelliteLexicon[var]
        df = pd.DataFrame({"observation": [1.0, 2.0]})
        out = mod(df)
        assert out["observation"].tolist() == [1.0, 2.0]


# ─────────────────────────────────────────────────────────────────────
# URI templating / task creation (offline, bypasses async S3 init)
# ─────────────────────────────────────────────────────────────────────


def _bare_conv(source: str = "prepbufr") -> NNJAObsConv:
    """Build an NNJAObsConv without triggering async S3 init (for offline tests)."""
    obj = NNJAObsConv.__new__(NNJAObsConv)
    obj._source = source
    obj._tolerance_lower = timedelta(0)
    obj._tolerance_upper = timedelta(0)
    obj._verbose = False
    obj._cache = False
    obj._max_workers = 1
    obj.async_timeout = 60
    obj._tmp_cache_hash = None
    obj.fs = None
    return obj


def _bare_sat(satellites: list[str] | None = None) -> NNJAObsSat:
    obj = NNJAObsSat.__new__(NNJAObsSat)
    obj._satellites = satellites
    obj._tolerance_lower = timedelta(0)
    obj._tolerance_upper = timedelta(0)
    obj._verbose = False
    obj._cache = False
    obj._max_workers = 1
    obj.async_timeout = 60
    obj._tmp_cache_hash = None
    obj.fs = None
    return obj


def test_nnja_obs_conv_build_uri():
    obj = _bare_conv("prepbufr")
    uri = obj._build_uri(datetime(2024, 1, 1, 0))
    assert uri == (
        "s3://noaa-reanalyses-pds/observations/reanalysis/conv/prepbufr/"
        "2024/01/prepbufr/gdas.20240101.t00z.prepbufr.nr"
    )

    obj2 = _bare_conv("convbufr")
    assert (
        "convbufr/2024/01/convbufr/gdas.20240101.t00z.convbufr.nr"
        in obj2._build_uri(datetime(2024, 1, 1, 0))
    )


def test_nnja_obs_sat_build_uri():
    obj = _bare_sat()
    uri = obj._build_uri(datetime(2024, 1, 1, 6), "atms", "atms")
    assert uri == (
        "s3://noaa-reanalyses-pds/observations/reanalysis/atms/atms/"
        "2024/01/bufr/gdas.20240101.t06z.atms.tm00.bufr_d"
    )

    uri2 = obj._build_uri(datetime(2024, 1, 1, 12), "amsua", "1bamua")
    assert (
        "amsua/1bamua/2024/01/bufr/gdas.20240101.t12z.1bamua.tm00.bufr_d"
        in uri2
    )


def test_nnja_obs_conv_create_tasks_dedupes_and_aligns():
    obj = _bare_conv()
    tasks = obj._create_tasks([datetime(2024, 1, 1, 0)], ["t"])
    assert len(tasks) == 1
    assert tasks[0].datetime_file == datetime(2024, 1, 1, 0)
    assert "t" in tasks[0].var_plan

    # With tolerance spanning two cycles
    obj._tolerance_lower = timedelta(hours=-3)
    obj._tolerance_upper = timedelta(hours=3)
    tasks = obj._create_tasks([datetime(2024, 1, 1, 0)], ["t", "u"])
    cycle_hours = sorted({t.datetime_file.hour for t in tasks})
    # window [21z prev day, 03z] hits 18z, 00z (ceil to 6h grid)
    assert 0 in cycle_hours


def test_nnja_obs_sat_create_tasks_filters_satellites():
    obj = _bare_sat(satellites=["n20"])
    tasks = obj._create_tasks([datetime(2024, 1, 1, 0)], ["atms"])
    assert len(tasks) == 1
    # ATMS NNJA carries npp,n20 — restricting to n20 should keep only n20
    assert tasks[0].platforms == ("n20",)

    obj_invalid = _bare_sat(satellites=["nonexistent"])
    # No task created when no platform overlap (warning, no exception)
    tasks_empty = obj_invalid._create_tasks([datetime(2024, 1, 1, 0)], ["atms"])
    assert tasks_empty == []


# ─────────────────────────────────────────────────────────────────────
# Time validation (offline)
# ─────────────────────────────────────────────────────────────────────


def test_nnja_obs_conv_validate_time():
    NNJAObsConv._validate_time([datetime(2024, 1, 1, 0)])
    NNJAObsConv._validate_time([datetime(2024, 1, 1, 6)])
    NNJAObsConv._validate_time([datetime(2024, 1, 1, 12)])
    NNJAObsConv._validate_time([datetime(2024, 1, 1, 18)])

    with pytest.raises(ValueError):
        NNJAObsConv._validate_time([datetime(2024, 1, 1, 1)])

    with pytest.raises(ValueError):
        NNJAObsConv._validate_time([datetime(2024, 1, 1, 0, 30)])

    with pytest.raises(ValueError):
        NNJAObsConv._validate_time([datetime(1970, 1, 1, 0)])


def test_nnja_obs_sat_validate_time():
    NNJAObsSat._validate_time([datetime(2024, 1, 1, 0)])
    with pytest.raises(ValueError):
        NNJAObsSat._validate_time([datetime(2024, 1, 1, 3)])


# ─────────────────────────────────────────────────────────────────────
# resolve_fields (offline)
# ─────────────────────────────────────────────────────────────────────


def test_nnja_obs_conv_resolve_fields():
    schema_full = NNJAObsConv.resolve_fields(None)
    assert schema_full.names == NNJAObsConv.SCHEMA.names

    schema_subset = NNJAObsConv.resolve_fields(
        ["time", "lat", "lon", "observation", "variable"]
    )
    assert schema_subset.names == ["time", "lat", "lon", "observation", "variable"]

    schema_str = NNJAObsConv.resolve_fields("time")
    assert schema_str.names == ["time"]

    # passing pa.Schema unchanged
    sub = pa.schema(
        [
            NNJAObsConv.SCHEMA.field("time"),
            NNJAObsConv.SCHEMA.field("observation"),
        ]
    )
    out = NNJAObsConv.resolve_fields(sub)
    assert out.names == ["time", "observation"]

    with pytest.raises(KeyError):
        NNJAObsConv.resolve_fields(["nonexistent"])

    bad_schema = pa.schema([pa.field("nonexistent", pa.float32())])
    with pytest.raises(KeyError):
        NNJAObsConv.resolve_fields(bad_schema)

    wrong_type = pa.schema([pa.field("time", pa.string())])
    with pytest.raises(TypeError):
        NNJAObsConv.resolve_fields(wrong_type)


def test_nnja_obs_sat_resolve_fields():
    schema_full = NNJAObsSat.resolve_fields(None)
    assert schema_full.names == NNJAObsSat.SCHEMA.names

    subset = NNJAObsSat.resolve_fields(
        ["time", "lat", "lon", "channel_index", "satellite", "observation", "variable"]
    )
    assert "channel_index" in subset.names

    with pytest.raises(KeyError):
        NNJAObsSat.resolve_fields(["nonexistent"])


# ─────────────────────────────────────────────────────────────────────
# Construction-time validation (offline)
# ─────────────────────────────────────────────────────────────────────


def test_nnja_obs_conv_invalid_source():
    with pytest.raises(ValueError):
        NNJAObsConv(source="not_a_source", cache=False, verbose=False)


def test_nnja_obs_sat_invalid_satellite():
    with pytest.raises(ValueError):
        NNJAObsSat(satellites=["not_a_satellite"], cache=False, verbose=False)


# ─────────────────────────────────────────────────────────────────────
# Tolerance conversion (offline)
# ─────────────────────────────────────────────────────────────────────


def test_nnja_obs_conv_tolerance_conversion():
    ds = NNJAObsConv(
        time_tolerance=timedelta(hours=1), cache=False, verbose=False
    )
    assert ds._tolerance_lower == timedelta(hours=-1)
    assert ds._tolerance_upper == timedelta(hours=1)

    ds_np = NNJAObsConv(
        time_tolerance=np.timedelta64(2, "h"), cache=False, verbose=False
    )
    assert ds_np._tolerance_lower == timedelta(hours=-2)
    assert ds_np._tolerance_upper == timedelta(hours=2)

    ds_asym = NNJAObsConv(
        time_tolerance=(np.timedelta64(-3, "h"), np.timedelta64(1, "h")),
        cache=False,
        verbose=False,
    )
    assert ds_asym._tolerance_lower == timedelta(hours=-3)
    assert ds_asym._tolerance_upper == timedelta(hours=1)


# ─────────────────────────────────────────────────────────────────────
# Slow live-S3 integration tests
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "time",
    [datetime(year=2024, month=1, day=1, hour=0)],
)
@pytest.mark.parametrize(
    "variable, tol",
    [
        (["t"], timedelta(0)),
        (["u", "v"], timedelta(0)),
    ],
)
def test_nnja_obs_conv_fetch(time, variable, tol):
    ds = NNJAObsConv(time_tolerance=tol, cache=False, verbose=False)
    df = ds(time, variable)

    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["variable"].unique()).issubset(set(variable))
    assert "observation" in df.columns
    assert not df.empty


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(180)
@pytest.mark.parametrize("cache", [True, False])
def test_nnja_obs_conv_cache(cache):
    ds = NNJAObsConv(time_tolerance=timedelta(0), cache=cache, verbose=False)
    df = ds(datetime(2024, 1, 1, 0), ["t"])
    assert list(df.columns) == ds.SCHEMA.names
    assert pathlib.Path(ds.cache).is_dir() == cache

    df2 = ds(datetime(2024, 1, 1, 0), ["t"])
    assert list(df2.columns) == ds.SCHEMA.names

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(180)
def test_nnja_obs_conv_schema_fields():
    ds = NNJAObsConv(time_tolerance=timedelta(0), cache=False, verbose=False)
    df_full = ds(datetime(2024, 1, 1, 0), ["t"])
    assert list(df_full.columns) == ds.SCHEMA.names

    subset = ["time", "lat", "lon", "observation", "variable"]
    df_sub = ds(datetime(2024, 1, 1, 0), ["t"], fields=subset)
    assert list(df_sub.columns) == subset


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "variable, satellites",
    [
        (["atms"], ["n20"]),
        (["amsua"], ["n19", "metop-b"]),
    ],
)
def test_nnja_obs_sat_fetch(variable, satellites):
    ds = NNJAObsSat(
        satellites=satellites,
        time_tolerance=timedelta(0),
        cache=False,
        verbose=False,
    )
    df = ds(datetime(2024, 1, 1, 0), variable)

    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["variable"].unique()).issubset(set(variable))
    if not df.empty:
        assert set(df["satellite"].unique()).issubset(set(satellites))
