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

<<<<<<< HEAD
import pathlib
import shutil
from datetime import datetime, timedelta
=======
import shutil
from datetime import datetime, timedelta
from unittest.mock import patch
>>>>>>> 3e27b0e34a010228ea22e466bab9a8cd66ec4dc8

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

<<<<<<< HEAD
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

    # u, v, gps, gps_t, gps_q: identity (already SI)
    for var in ("u", "v", "gps", "gps_t", "gps_q"):
        _, mod = NNJAObsConvLexicon[var]
        df = mod(pd.DataFrame({"observation": [3.14]}))
        assert df["observation"].iloc[0] == pytest.approx(3.14)


def test_nnja_obs_conv_lexicon_routes():
    """Conv lexicon entries are route-prefixed with 'prepbufr::' or 'gpsro::'."""
    for var, vocab in NNJAObsConvLexicon.VOCAB.items():
        route, _, rest = vocab.partition("::")
        assert route in ("prepbufr", "gpsro"), f"{var}: unexpected route '{route}'"
        assert rest, f"{var}: empty payload after route prefix"
        if route == "gpsro":
            # rest must parse as an int BUFR descriptor id
            int(rest)


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
    uri = obj._build_prepbufr_uri(datetime(2024, 1, 1, 0))
    assert uri == (
        "s3://noaa-reanalyses-pds/observations/reanalysis/conv/prepbufr/"
        "2024/01/prepbufr/gdas.20240101.t00z.prepbufr.nr"
    )

    obj2 = _bare_conv("convbufr")
    assert (
        "convbufr/2024/01/convbufr/gdas.20240101.t00z.convbufr.nr"
        in obj2._build_prepbufr_uri(datetime(2024, 1, 1, 0))
    )

    # Back-compat alias still works
    assert obj._build_uri(datetime(2024, 1, 1, 0)) == uri


def test_nnja_obs_conv_build_gpsro_uri():
    obj = _bare_conv("prepbufr")
    uri = obj._build_gpsro_uri(datetime(2024, 1, 1, 6))
    assert uri == (
        "s3://noaa-reanalyses-pds/observations/reanalysis/gps/gpsro/"
        "2024/01/bufr/gdas.20240101.t06z.gpsro.tm00.bufr_d"
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
    from earth2studio.data.nnja import _NNJAConvTask

    obj = _bare_conv()
    tasks = obj._create_tasks([datetime(2024, 1, 1, 0)], ["t"])
    assert len(tasks) == 1
    assert isinstance(tasks[0], _NNJAConvTask)
    assert tasks[0].datetime_file == datetime(2024, 1, 1, 0)
    assert "t" in tasks[0].var_plan

    # With tolerance spanning two cycles
    obj._tolerance_lower = timedelta(hours=-3)
    obj._tolerance_upper = timedelta(hours=3)
    tasks = obj._create_tasks([datetime(2024, 1, 1, 0)], ["t", "u"])
    cycle_hours = sorted({t.datetime_file.hour for t in tasks})
    assert 0 in cycle_hours


def test_nnja_obs_conv_create_tasks_routes_gpsro_separately():
    """Requesting prepbufr + gpsro variables creates two task types per cycle."""
    from earth2studio.data.nnja import _NNJAConvTask, _NNJAGpsRoTask

    obj = _bare_conv()
    tasks = obj._create_tasks([datetime(2024, 1, 1, 0)], ["t", "gps_t", "gps_q"])
    types = {type(task).__name__ for task in tasks}
    assert {"_NNJAConvTask", "_NNJAGpsRoTask"} == types

    conv_tasks = [t for t in tasks if isinstance(t, _NNJAConvTask)]
    gps_tasks = [t for t in tasks if isinstance(t, _NNJAGpsRoTask)]
    assert len(conv_tasks) == 1
    assert len(gps_tasks) == 1
    assert "t" in conv_tasks[0].var_plan
    assert "gps_t" in gps_tasks[0].var_plan
    assert "gps_q" in gps_tasks[0].var_plan
    # The gpsro var_plan stores (descriptor_id, modifier)
    desc_id, _ = gps_tasks[0].var_plan["gps_t"]
    assert desc_id == 12001
    desc_id, _ = gps_tasks[0].var_plan["gps_q"]
    assert desc_id == 13001
    # Verify URIs
    assert "/conv/prepbufr/" in conv_tasks[0].s3_uri
    assert "/gps/gpsro/" in gps_tasks[0].s3_uri


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
=======
from earth2studio.data import NNJAObsConv

pytest.importorskip("pybufrkit", reason="pybufrkit not installed")


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
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
    ds = NNJAObsConv(time_tolerance=tol, cache=False, verbose=False, decode_workers=16)
    df = ds(time, variable)

    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["variable"].unique()).issubset(set(variable))
    assert "observation" in df.columns
    assert not df.empty


@pytest.mark.parametrize("cache", [True, False])
def test_nnja_obs_conv_cache_mock(cache, tmp_path):
    """Test NNJAObsConv cache behavior with mocked S3 fetch."""
    # Create a minimal mock DataFrame matching NNJA output schema
    mock_df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:00"]),
            "pres": [85000.0, 92500.0],
            "elev": [100.0, 50.0],
            "type": [120, 120],
            "class": ["ADPUPA", "ADPUPA"],
            "lat": [40.0, 41.0],
            "lon": [250.0, 251.0],
            "station": ["72469", "72469"],
            "station_elev": [1000.0, 1000.0],
            "quality": [2, 2],
            "observation": [273.15, 280.0],
            "variable": ["t", "t"],
        }
    )

    with patch("earth2studio.data.nnja._sync_async") as mock_sync:
        mock_sync.return_value = mock_df

        ds = NNJAObsConv(time_tolerance=timedelta(0), cache=cache, verbose=False)

        # First fetch
        df = ds(datetime(2024, 1, 1, 0), ["t"])
        assert list(df.columns) == ds.SCHEMA.names

        # Second fetch (should use cache if enabled)
        df2 = ds(datetime(2024, 1, 1, 0), ["t"])
        assert list(df2.columns) == ds.SCHEMA.names

    # Clean up
    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


def test_nnja_obs_conv_exceptions():

    # Invalid source
    with pytest.raises(ValueError):
        NNJAObsConv(source="not_a_source", cache=False, verbose=False)

    # Invalid variable - test via lexicon lookup directly (avoids network)
    with pytest.raises(KeyError):
        from earth2studio.lexicon import NNJAObsConvLexicon

        NNJAObsConvLexicon["invalid_variable"]

    # Invalid fields - test via resolve_fields directly (avoids network)
    with pytest.raises(KeyError):
        NNJAObsConv.resolve_fields(["observation", "variable", "invalid_field"])

    invalid_schema = pa.schema(
        [
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
            pa.field("nonexistent", pa.float32()),
        ]
    )
    with pytest.raises(KeyError):
        NNJAObsConv.resolve_fields(invalid_schema)

    wrong_type_schema = pa.schema(
        [
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
            pa.field("time", pa.string()),
        ]
    )
    with pytest.raises(TypeError):
        NNJAObsConv.resolve_fields(wrong_type_schema)
>>>>>>> 3e27b0e34a010228ea22e466bab9a8cd66ec4dc8


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


<<<<<<< HEAD
def test_nnja_obs_sat_validate_time():
    NNJAObsSat._validate_time([datetime(2024, 1, 1, 0)])
    with pytest.raises(ValueError):
        NNJAObsSat._validate_time([datetime(2024, 1, 1, 3)])


# ─────────────────────────────────────────────────────────────────────
# resolve_fields (offline)
# ─────────────────────────────────────────────────────────────────────
=======
def test_nnja_obs_conv_tolerance_conversion():

    ds = NNJAObsConv(time_tolerance=timedelta(hours=1), cache=False, verbose=False)
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
>>>>>>> 3e27b0e34a010228ea22e466bab9a8cd66ec4dc8


def test_nnja_obs_conv_resolve_fields():
    schema_full = NNJAObsConv.resolve_fields(None)
    assert schema_full.names == NNJAObsConv.SCHEMA.names

    schema_subset = NNJAObsConv.resolve_fields(
        ["time", "lat", "lon", "observation", "variable"]
    )
    assert schema_subset.names == ["time", "lat", "lon", "observation", "variable"]

    schema_str = NNJAObsConv.resolve_fields("time")
    assert schema_str.names == ["time"]

<<<<<<< HEAD
    # passing pa.Schema unchanged
=======
>>>>>>> 3e27b0e34a010228ea22e466bab9a8cd66ec4dc8
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


<<<<<<< HEAD
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
=======
def test_nnja_obs_conv_mock_fetch():
    """Test NNJAObsConv data processing with mocked S3 fetch."""

    # Create a minimal mock DataFrame matching NNJA output schema
    mock_df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:00"]),
            "pres": [85000.0, 92500.0],
            "elev": [100.0, 50.0],
            "type": [120, 120],
            "class": ["ADPUPA", "ADPUPA"],
            "lat": [40.0, 41.0],
            "lon": [250.0, 251.0],
            "station": ["72469", "72469"],
            "station_elev": [1000.0, 1000.0],
            "quality": [2, 2],
            "observation": [273.15, 280.0],
            "variable": ["t", "t"],
        }
    )

    with patch.object(NNJAObsConv, "fetch") as mock_fetch:
        mock_fetch.return_value = mock_df

        ds = NNJAObsConv(time_tolerance=timedelta(0), cache=False, verbose=False)

        # Patch _sync_async to call the mock directly
        with patch("earth2studio.data.nnja._sync_async") as mock_sync:
            mock_sync.return_value = mock_df
            df = ds(datetime(2024, 1, 1, 0), ["t"])

    assert list(df.columns) == ds.SCHEMA.names
    assert len(df) == 2
    assert set(df["variable"].unique()) == {"t"}
    assert df["observation"].iloc[0] == pytest.approx(273.15)


def test_nnja_obs_conv_available():
    """Test NNJAObsConv.available() classmethod with both datetime types."""
    # Valid 6-hourly cycle times
    assert NNJAObsConv.available(datetime(2024, 1, 1, 0)) is True
    assert NNJAObsConv.available(datetime(2024, 1, 1, 6)) is True
    assert NNJAObsConv.available(datetime(2024, 1, 1, 12)) is True
    assert NNJAObsConv.available(datetime(2024, 1, 1, 18)) is True

    # Invalid hours
    assert NNJAObsConv.available(datetime(2024, 1, 1, 1)) is False
    assert NNJAObsConv.available(datetime(2024, 1, 1, 7)) is False

    # Before MIN_DATE
    assert NNJAObsConv.available(datetime(1970, 1, 1, 0)) is False

    # np.datetime64 input - valid
    assert NNJAObsConv.available(np.datetime64("2024-01-01T00:00:00")) is True
    assert NNJAObsConv.available(np.datetime64("2024-01-01T06:00:00")) is True

    # np.datetime64 input - invalid
    assert NNJAObsConv.available(np.datetime64("2024-01-01T01:00:00")) is False
    assert NNJAObsConv.available(np.datetime64("1970-01-01T00:00:00")) is False


# These are unit tests for BUFR decoding


def test_nnja_safe_int():
    """Test _safe_int helper function with various input types."""
    from earth2studio.data.utils_bufr import safe_int as _safe_int

    # int/float inputs
    assert _safe_int(42) == 42
    assert _safe_int(3.14) == 3
    assert _safe_int(-7.9) == -7

    # bytes input
    assert _safe_int(b"123") == 123
    assert _safe_int(b"  456  ") == 456
    assert _safe_int(b"") == 0
    assert _safe_int(b"abc") == 0  # non-numeric bytes

    # None input
    assert _safe_int(None) == 0

    # string input
    assert _safe_int("789") == 789
    assert _safe_int("  012  ") == 12
    assert _safe_int("") == 0
    assert _safe_int("not_a_number") == 0


def test_nnja_extract_dx_tables_empty():
    """Test _extract_dx_tables with empty/minimal inputs."""
    from earth2studio.data.utils_bufr import extract_dx_tables as _extract_dx_tables

    table_b: dict = {}
    table_d: dict = {}

    # Empty flat list
    _extract_dx_tables([], table_b, table_d)
    assert table_b == {}
    assert table_d == {}

    # Only n_a (no entries)
    _extract_dx_tables([0], table_b, table_d)
    assert table_b == {}
    assert table_d == {}

    # n_a=0, n_b=0 (skip to table D)
    _extract_dx_tables([0, 0], table_b, table_d)
    assert table_b == {}
    assert table_d == {}

    # n_a=0, n_b=0, n_d=0 (all empty)
    _extract_dx_tables([0, 0, 0], table_b, table_d)
    assert table_b == {}
    assert table_d == {}


def test_nnja_extract_dx_tables_truncated():
    """Test _extract_dx_tables with truncated/partial data."""
    from earth2studio.data.utils_bufr import extract_dx_tables as _extract_dx_tables

    table_b: dict = {}
    table_d: dict = {}

    # n_a entries truncated (should return early)
    _extract_dx_tables([1], table_b, table_d)  # n_a=1 but no entries
    assert table_b == {}
    assert table_d == {}

    # Table B entry truncated (less than 11 fields)
    # n_a=0, n_b=1, then only partial entry
    table_b.clear()
    table_d.clear()
    _extract_dx_tables([0, 1, 0, 1, 2], table_b, table_d)  # truncated B entry
    assert table_b == {}

    # Table D entry truncated
    # n_a=0, n_b=0, n_d=1, then only partial entry
    table_b.clear()
    table_d.clear()
    _extract_dx_tables([0, 0, 1, 0, 1], table_b, table_d)  # truncated D entry
    assert table_d == {}


def test_nnja_extract_dx_tables_valid_entries():
    """Test _extract_dx_tables with valid Table B and D entries."""
    from earth2studio.data.utils_bufr import extract_dx_tables as _extract_dx_tables

    table_b: dict = {}
    table_d: dict = {}

    # Create a valid Table B entry
    # Format: n_a, [a_entries...], n_b, [b_entries (11 fields each)...], n_d, [d_entries...]
    # B entry: f, x, y, mnemonic, desc(skip), unit, sign_scale, scale, sign_ref, ref, width
    flat = [
        0,  # n_a = 0
        1,  # n_b = 1
        0,
        1,
        2,
        b"TOB",
        b"",
        b"K",
        b"+",
        b"2",
        b"+",
        b"0",
        b"16",  # 11 fields
        0,  # n_d = 0
    ]
    _extract_dx_tables(flat, table_b, table_d)
    # desc_id = 0*100000 + 1*1000 + 2 = 1002
    assert 1002 in table_b
    assert table_b[1002][0] == "TOB"  # mnemonic

    # Test with zero descriptor (should be skipped)
    table_b.clear()
    table_d.clear()
    flat_zero = [
        0,  # n_a = 0
        1,  # n_b = 1
        0,
        0,
        0,
        b"ZERO",
        b"",
        b"K",
        b"+",
        b"0",
        b"+",
        b"0",
        b"8",  # desc_id = 0
        0,  # n_d = 0
    ]
    _extract_dx_tables(flat_zero, table_b, table_d)
    assert 0 not in table_b  # Zero descriptor skipped

    # Test with negative scale/reference
    table_b.clear()
    table_d.clear()
    flat_neg = [
        0,  # n_a = 0
        1,  # n_b = 1
        0,
        12,
        1,
        b"LAT",
        b"",
        b"DEG",
        b"-",
        b"5",
        b"-",
        b"9000000",
        b"25",
        0,  # n_d = 0
    ]
    _extract_dx_tables(flat_neg, table_b, table_d)
    # desc_id = 0*100000 + 12*1000 + 1 = 12001
    assert 12001 in table_b
    assert table_b[12001][2] == -5  # scale is negative
    assert table_b[12001][3] == -9000000  # reference is negative


def test_nnja_extract_dx_tables_table_d():
    """Test _extract_dx_tables Table D sequence entries."""
    from earth2studio.data.utils_bufr import extract_dx_tables as _extract_dx_tables

    table_b: dict = {}
    table_d: dict = {}

    # Table D entry: f, x, y, seq_mnemonic, n_members, [member_mnemonics...]
    flat = [
        0,  # n_a = 0
        0,  # n_b = 0
        1,  # n_d = 1
        3,
        1,
        1,
        b"HEADR",  # seq entry header (f=3, x=1, y=1)
        2,  # n_members = 2
        b"SID",
        b"XOB",  # member mnemonics
    ]
    _extract_dx_tables(flat, table_b, table_d)
    # seq_id = 3*100000 + 1*1000 + 1 = 301001
    assert 301001 in table_d
    assert table_d[301001][0] == "HEADR"
    assert table_d[301001][1] == ["SID", "XOB"]

    # Test zero seq_id (should be skipped)
    table_b.clear()
    table_d.clear()
    flat_zero = [
        0,  # n_a = 0
        0,  # n_b = 0
        1,  # n_d = 1
        0,
        0,
        0,
        b"ZERO",  # seq_id = 0
        1,
        b"X",
    ]
    _extract_dx_tables(flat_zero, table_b, table_d)
    assert 0 not in table_d  # Zero seq_id skipped

    # Test truncated member list
    table_b.clear()
    table_d.clear()
    flat_trunc = [
        0,  # n_a = 0
        0,  # n_b = 0
        1,  # n_d = 1
        3,
        1,
        2,
        b"SEQ",
        3,  # n_members = 3 but only 1 provided
        b"ONLY_ONE",
    ]
    _extract_dx_tables(flat_trunc, table_b, table_d)
    # Should still create entry with partial members
    assert 301002 in table_d
    assert table_d[301002][1] == ["ONLY_ONE"]


def test_nnja_obs_conv_build_uris():
    """Test URI building methods for prepbufr and gpsro."""
    ds = NNJAObsConv(cache=False, verbose=False)

    # Test prepbufr URI
    cycle = datetime(2024, 1, 15, 6)
    prepbufr_uri = ds._build_prepbufr_uri(cycle)
    assert "2024" in prepbufr_uri
    assert "01" in prepbufr_uri
    assert "20240115" in prepbufr_uri
    assert "t06z" in prepbufr_uri
    assert "prepbufr" in prepbufr_uri

    # Test gpsro URI
    gpsro_uri = ds._build_gpsro_uri(cycle)
    assert "2024" in gpsro_uri
    assert "01" in gpsro_uri
    assert "20240115" in gpsro_uri
    assert "t06z" in gpsro_uri
    assert "gpsro" in gpsro_uri

    # Test backward compat alias
    assert ds._build_uri(cycle) == ds._build_prepbufr_uri(cycle)


def test_nnja_obs_conv_create_tasks():
    """Test _create_tasks method for prepbufr variables."""
    from earth2studio.data.nnja import _NNJAConvTask

    # Use zero tolerance to get exactly one task per cycle
    ds = NNJAObsConv(time_tolerance=timedelta(0), cache=False, verbose=False)

    # Test prepbufr-only variable
    tasks = ds._create_tasks([datetime(2024, 1, 1, 0)], ["t"])
    assert len(tasks) == 1
    assert isinstance(tasks[0], _NNJAConvTask)
    assert "prepbufr" in tasks[0].s3_uri
    assert tasks[0].datetime_file == datetime(2024, 1, 1, 0)

    # Test multiple variables (same route)
    tasks_multi = ds._create_tasks([datetime(2024, 1, 1, 0)], ["t", "u", "v"])
    assert len(tasks_multi) == 1  # Same file, combined var_plan
    assert "t" in tasks_multi[0].var_plan
    assert "u" in tasks_multi[0].var_plan
    assert "v" in tasks_multi[0].var_plan

    # Test multiple times
    tasks_times = ds._create_tasks(
        [datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 6)], ["t"]
    )
    assert len(tasks_times) == 2  # Two different cycles


def test_nnja_obs_conv_finalize_decoded_df():
    """Test _finalize_decoded_df with various inputs."""
    ds = NNJAObsConv(cache=False, verbose=False)

    # Import modifier function for testing
    from earth2studio.lexicon import NNJAObsConvLexicon

    _, modifier = NNJAObsConvLexicon["t"]

    # Empty rows
    result = ds._finalize_decoded_df(
        [], {"t": ("TOB", modifier)}, convert_pres_mb_to_pa=True
    )
    assert result.empty

    # Rows with missing variable (should be filtered out)
    rows = [
        {
            "time": datetime(2024, 1, 1, 0),
            "lat": 40.0,
            "lon": 250.0,
            "pres": 850.0,
            "elev": None,
            "type": 120,
            "class": "ADPUPA",
            "station": "72469",
            "station_elev": 1000.0,
            "observation": 273.15,
            "variable": "other_var",  # Not in var_plan
        }
    ]
    result = ds._finalize_decoded_df(
        rows, {"t": ("TOB", modifier)}, convert_pres_mb_to_pa=True
    )
    assert result.empty  # No rows match "t"

    # Rows with matching variable
    rows_match = [
        {
            "time": datetime(2024, 1, 1, 0),
            "lat": 40.0,
            "lon": 250.0,
            "pres": 850.0,
            "elev": None,
            "type": 120,
            "class": "ADPUPA",
            "station": "72469",
            "station_elev": 1000.0,
            "observation": 273.15,
            "variable": "t",
        }
    ]
    result = ds._finalize_decoded_df(
        rows_match, {"t": ("TOB", modifier)}, convert_pres_mb_to_pa=True
    )
    assert len(result) == 1
    # Check pressure conversion (850 mb -> 85000 Pa)
    assert result["pres"].iloc[0] == pytest.approx(85000.0)

    # Test without pressure conversion (gpsro path)
    result_no_conv = ds._finalize_decoded_df(
        rows_match, {"t": ("TOB", modifier)}, convert_pres_mb_to_pa=False
    )
    assert result_no_conv["pres"].iloc[0] == pytest.approx(850.0)


def test_nnja_obs_conv_finalize_adds_missing_columns():
    """Test _finalize_decoded_df adds missing nullable columns."""
    ds = NNJAObsConv(cache=False, verbose=False)

    from earth2studio.lexicon import NNJAObsConvLexicon

    _, modifier = NNJAObsConvLexicon["t"]

    # Rows missing some optional columns
    rows = [
        {
            "time": datetime(2024, 1, 1, 0),
            "lat": 40.0,
            "lon": 250.0,
            "pres": 850.0,
            # "elev" missing
            "type": 120,
            "class": "ADPUPA",
            # "station" missing
            # "station_elev" missing
            "observation": 273.15,
            "variable": "t",
        }
    ]
    result = ds._finalize_decoded_df(
        rows, {"t": ("TOB", modifier)}, convert_pres_mb_to_pa=True
    )

    # All schema columns should be present
    assert list(result.columns) == list(ds.SCHEMA.names)
    # Missing columns should be NaN/None
    assert pd.isna(result["elev"].iloc[0])
    assert result["station"].iloc[0] is None
    assert pd.isna(result["station_elev"].iloc[0])
>>>>>>> 3e27b0e34a010228ea22e466bab9a8cd66ec4dc8
