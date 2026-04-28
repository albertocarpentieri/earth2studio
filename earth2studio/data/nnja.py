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

# NOAA-NASA Joint Archive (NNJA) of Observations for Earth System Reanalysis.
#
# Reference: https://psl.noaa.gov/data/nnja_obs/
# Public S3 bucket: s3://noaa-reanalyses-pds/observations/reanalysis/
#
# This module is intentionally self-contained: the PrepBUFR decoding
# helpers below duplicate concepts already present in
# ``earth2studio.data.gdas`` (which decodes the same NCEP PrepBUFR file
# format from NOMADS). A future PR may extract a shared ``_prepbufr``
# module; for now the duplication keeps this PR isolated and avoids
# changing GDAS behaviour.

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import hashlib
import os
import pathlib
import shutil
import struct
import sys
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import nest_asyncio
import numpy as np
import pandas as pd
import pyarrow as pa
import s3fs
from loguru import logger
from tqdm.asyncio import tqdm

from earth2studio.data.utils import datasource_cache_root, prep_data_inputs
from earth2studio.lexicon import NNJAObsConvLexicon, NNJASatelliteLexicon
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.time import normalize_time_tolerance
from earth2studio.utils.type import TimeArray, TimeTolerance, VariableArray

try:
    import eccodes  # type: ignore[import-untyped]
except ImportError:
    OptionalDependencyFailure("data")
    eccodes = None  # type: ignore[assignment]

try:
    from pybufrkit.decoder import Decoder as BufrDecoder
    from pybufrkit.tables import TableGroupCacheManager
except ImportError:
    OptionalDependencyFailure("data")
    BufrDecoder = None  # type: ignore[assignment,misc]
    TableGroupCacheManager = None  # type: ignore[assignment,misc]


NNJA_BUCKET = "noaa-reanalyses-pds"
NNJA_PREFIX = "observations/reanalysis"


@contextlib.contextmanager
def _silence_bufr_noise() -> Iterator[None]:
    """Suppress chatty C-library stderr from pybufrkit and eccodes.

    Both libraries write informational messages straight to file
    descriptor 2 (e.g. ``Cannot find sub-centre 3 nor valid default``
    from pybufrkit, ``ECCODES ERROR : unable to get descriptor``
    from eccodes) when the file uses NCEP-local descriptors. We rely
    on the DX tables embedded in each NNJA file to decode those
    correctly, so these messages are spurious and would otherwise
    flood the log with one line per BUFR message.

    The redirect only covers C-level writes; Python ``print``,
    ``logger`` and exceptions still propagate normally. We also
    flush ``sys.stderr`` first so any pending Python-side stderr
    is preserved.
    """
    sys.stderr.flush()
    saved_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        try:
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved_fd, 2)
    finally:
        os.close(devnull_fd)
        os.close(saved_fd)


# ── PrepBUFR descriptor IDs (NCEP-local) ─────────────────────────────
# Header field descriptors
_HDR_SID = 1194  # Station ID
_HDR_XOB = 6240  # Longitude (deg E)
_HDR_YOB = 5002  # Latitude (deg N)
_HDR_DHR = 4215  # Obs time minus cycle time (h)
_HDR_ELV = 10199  # Station elevation (m)
_HDR_TYP = 55007  # Report type code
_HDR_T29 = 55008  # Data dump report type code

# Observation field descriptors
_OBS_CAT = 8193  # Observation category code
_OBS_POB = 7245  # Pressure observation (MB)
_OBS_ZOB = 10007  # Height (m)
_OBS_TOB = 12245  # Temperature (DEG C)
_OBS_QOB = 13245  # Specific humidity (MG/KG)
_OBS_UOB = 11003  # U-wind component (m/s)
_OBS_VOB = 11004  # V-wind component (m/s)

_OBSERVATION_DESCR_IDS: set[int] = {
    _OBS_POB,
    _OBS_ZOB,
    _OBS_TOB,
    _OBS_QOB,
    _OBS_UOB,
    _OBS_VOB,
}

# Lexicon mnemonic -> descriptor ID for non-wind variables
_MNEMONIC_TO_DESCR: dict[str, int] = {
    "TOB": _OBS_TOB,
    "QOB": _OBS_QOB,
    "POB": _OBS_POB,
    "ZOB": _OBS_ZOB,
    "UOB": _OBS_UOB,
    "VOB": _OBS_VOB,
}

# PrepBUFR section-1 dataCategory -> NCEP message-type class string
_PREPBUFR_OBS_TYPES: dict[int, str] = {
    102: "ADPUPA",  # Upper air: radiosondes, pilot balloons, dropsondes
    104: "AIRCFT",  # Aircraft
    105: "SATWND",  # Satellite-derived winds
    107: "VADWND",  # VAD (NEXRAD) winds
    109: "ADPSFC",  # Surface land
    110: "SFCSHP",  # Surface marine
    112: "GPSIPW",  # GPS precipitable water
    113: "SYNDAT",  # Synthetic bogus data
    119: "RASSDA",  # RASS virtual temperature
    121: "ASCATW",  # ASCAT scatterometer winds
}


# ── GPS RO BUFR descriptor IDs (NCEP gpsro encoding) ─────────────────
# Header descriptors (per-occultation, scalar)
_GPSRO_SAID = 1007  # Satellite identifier (receiver)
_GPSRO_PTID = 1050  # Platform transmitter ID (GPS satellite)
_GPSRO_QFRO = 33039  # Quality flags for radio occultation
_GPSRO_LAT = 5001  # Latitude (deg)
_GPSRO_LON = 6001  # Longitude (deg)
_GPSRO_YEAR = 4001
_GPSRO_MONTH = 4002
_GPSRO_DAY = 4003
_GPSRO_HOUR = 4004
_GPSRO_MIN = 4005
_GPSRO_SEC = 4006

# Per-level descriptors
_GPSRO_IMPP = 7040  # Impact parameter (m), bending-angle level marker
_GPSRO_BNDA = 15037  # Bending angle (rad)
_GPSRO_HEIT = 7007  # Height (m), refractivity level marker
_GPSRO_ARFR = 15036  # Atmospheric refractivity
_GPSRO_GPHTST = 7009  # Geopotential height (m), retrieval level marker
_GPSRO_PRES = 10004  # Pressure (Pa)
_GPSRO_TEMP = 12001  # Air temperature (K)
_GPSRO_SPFH = 13001  # Specific humidity (kg/kg)

# Descriptor IDs the gpsro decoder pulls out as observations
_GPSRO_OBS_DESCRS: set[int] = {_GPSRO_BNDA, _GPSRO_TEMP, _GPSRO_SPFH}


# ── WMO satellite identifier code (Table 0-01-007) → NNJA platform ──
_SAT_ID_MAP: dict[int, str] = {
    3: "metop-b",
    4: "metop-a",
    5: "metop-c",
    206: "n15",
    207: "n16",
    208: "n17",
    209: "n18",
    223: "n19",
    224: "npp",
    225: "n20",
    226: "n21",
    784: "aqua",
    854: "megha-tropiques",
    825: "gpm",
}


# ── Schemas ─────────────────────────────────────────────────────────

_NNJA_CONV_SCHEMA = pa.schema(
    [
        pa.field("time", pa.timestamp("ns"), metadata={"nnja_name": "Time"}),
        pa.field(
            "pres",
            pa.float32(),
            nullable=True,
            metadata={"nnja_name": "Pressure"},
        ),
        pa.field(
            "elev",
            pa.float32(),
            nullable=True,
            metadata={"nnja_name": "Height"},
        ),
        pa.field(
            "type",
            pa.uint16(),
            nullable=True,
            metadata={"nnja_name": "Observation_Type"},
        ),
        pa.field(
            "class",
            pa.string(),
            nullable=True,
            metadata={"nnja_name": "Observation_Class"},
        ),
        pa.field("lat", pa.float32(), metadata={"nnja_name": "Latitude"}),
        pa.field("lon", pa.float32(), metadata={"nnja_name": "Longitude"}),
        pa.field(
            "station",
            pa.string(),
            nullable=True,
            metadata={"nnja_name": "Station_ID"},
        ),
        pa.field(
            "station_elev",
            pa.float32(),
            nullable=True,
            metadata={"nnja_name": "Station_Elevation"},
        ),
        pa.field("observation", pa.float32()),
        pa.field("variable", pa.string()),
    ]
)

_NNJA_SAT_SCHEMA = pa.schema(
    [
        pa.field("time", pa.timestamp("ns")),
        pa.field("lat", pa.float32()),
        pa.field("lon", pa.float32()),
        pa.field("scan_angle", pa.float32(), nullable=True),
        pa.field("channel_index", pa.uint16(), nullable=True),
        pa.field("solza", pa.float32(), nullable=True),
        pa.field("solaza", pa.float32(), nullable=True),
        pa.field("satellite_za", pa.float32(), nullable=True),
        pa.field("satellite_aza", pa.float32(), nullable=True),
        pa.field("satellite", pa.string()),
        pa.field("observation", pa.float32()),
        pa.field("variable", pa.string()),
    ]
)


# ── Async-task dataclasses ──────────────────────────────────────────


@dataclass
class _NNJAConvTask:
    """Async task for a single PrepBUFR cycle file (route ``prepbufr``)."""

    s3_uri: str
    datetime_file: datetime
    datetime_min: datetime
    datetime_max: datetime
    var_plan: dict[str, tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = field(
        default_factory=dict
    )


@dataclass
class _NNJAGpsRoTask:
    """Async task for a single gps/gpsro cycle BUFR file (route ``gpsro``)."""

    s3_uri: str
    datetime_file: datetime
    datetime_min: datetime
    datetime_max: datetime
    # Map var_name -> (bufr_descriptor_id, modifier)
    var_plan: dict[str, tuple[int, Callable[[pd.DataFrame], pd.DataFrame]]] = field(
        default_factory=dict
    )


@dataclass
class _NNJASatTask:
    """Async task for a single satellite-radiance BUFR cycle file."""

    s3_uri: str
    datetime_file: datetime
    datetime_min: datetime
    datetime_max: datetime
    sensor: str
    source: str
    platforms: tuple[str, ...]
    bufr_key: str
    e2s_obs_name: str
    modifier: Callable[[pd.DataFrame], pd.DataFrame]


# ─────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────


class _NNJAObsBase:
    """Shared infrastructure for NNJA DataFrame data sources.

    Subclasses must define ``SOURCE_ID``, ``SCHEMA``, ``MIN_DATE``, and
    implement ``_create_tasks(time_list, variable)`` and
    ``_decode_file(local_path, task)``.
    """

    SOURCE_ID: str
    SCHEMA: pa.Schema
    MIN_DATE: datetime = datetime(1979, 1, 1)

    def __init__(
        self,
        time_tolerance: TimeTolerance = np.timedelta64(0, "m"),
        max_workers: int = 24,
        cache: bool = True,
        async_timeout: int = 600,
        verbose: bool = True,
    ) -> None:
        self._verbose = verbose
        self._cache = cache
        self._max_workers = max_workers
        self.async_timeout = async_timeout
        self._tmp_cache_hash: str | None = None

        try:
            nest_asyncio.apply()
            loop = asyncio.get_running_loop()
            loop.run_until_complete(self._async_init())
        except RuntimeError:
            self.fs = None

        lower, upper = normalize_time_tolerance(time_tolerance)
        self._tolerance_lower = pd.to_timedelta(lower).to_pytimedelta()
        self._tolerance_upper = pd.to_timedelta(upper).to_pytimedelta()

    async def _async_init(self) -> None:
        """Async initialization of S3 filesystem."""
        self.fs = s3fs.S3FileSystem(
            anon=True, client_kwargs={}, asynchronous=True, skip_instance_cache=True
        )

    # ------------------------------------------------------------------
    # Synchronous entry point
    # ------------------------------------------------------------------
    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Fetch observations for a set of timestamps.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Cycle timestamps (UTC). Must align to a 6-hour cycle (00, 06,
            12, 18z); the time tolerance is used to bracket the cycle when
            selecting observations.
        variable : str | list[str] | VariableArray
            Variable ids defined in the source-specific lexicon
            (:py:class:`earth2studio.lexicon.NNJAObsConvLexicon` or
            :py:class:`earth2studio.lexicon.NNJASatelliteLexicon`).
        fields : str | list[str] | pa.Schema | None, optional
            Output column subset. ``None`` (default) returns all schema
            fields.

        Returns
        -------
        pd.DataFrame
            Observation DataFrame with columns matching the resolved schema.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.set_default_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers)
        )

        if self.fs is None:
            loop.run_until_complete(self._async_init())

        df = loop.run_until_complete(
            asyncio.wait_for(
                self.fetch(time, variable, fields), timeout=self.async_timeout
            )
        )

        if not self._cache:
            shutil.rmtree(self.cache, ignore_errors=True)

        return df

    # ------------------------------------------------------------------
    # Async fetch (downloads + decode)
    # ------------------------------------------------------------------
    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Async function to get data."""
        if self.fs is None:
            raise ValueError(
                "File store is not initialized! If calling this function "
                "directly make sure the data source is initialized inside "
                "the async loop."
            )

        session = await self.fs.set_session(refresh=True)

        time_list, variable_list = prep_data_inputs(time, variable)
        self._validate_time(time_list)
        schema = self.resolve_fields(fields)
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        async_tasks = self._create_tasks(time_list, variable_list)
        file_uri_set = {task.s3_uri for task in async_tasks}
        fetch_jobs = [self._fetch_remote_file(uri) for uri in file_uri_set]
        await tqdm.gather(
            *fetch_jobs, desc="Fetching NNJA files", disable=(not self._verbose)
        )

        if session:
            await session.close()

        df = self._compile_dataframe(async_tasks, variable_list, schema)
        return df

    # ------------------------------------------------------------------
    # File fetch
    # ------------------------------------------------------------------
    async def _fetch_remote_file(self, path: str) -> None:
        """Download a single remote file into the cache directory."""
        if self.fs is None:
            raise ValueError("File system is not initialized")

        cache_path = self._cache_path(path)
        if pathlib.Path(cache_path).is_file():
            return
        try:
            data = await self.fs._cat_file(path)
            with open(cache_path, "wb") as fh:
                fh.write(data)
        except FileNotFoundError:
            self._handle_missing_file(path)

    def _handle_missing_file(self, path: str) -> None:
        """Handle missing file during fetch. Override in subclasses if a
        warn-only behaviour is preferred."""
        logger.error(f"File {path} not found")
        raise FileNotFoundError(f"File {path} not found")

    # ------------------------------------------------------------------
    # Compile DataFrame
    # ------------------------------------------------------------------
    def _compile_dataframe(
        self,
        async_tasks: list,
        variables: list[str],
        schema: pa.Schema,
    ) -> pd.DataFrame:
        """Decode each fetched file and concatenate into a single DataFrame."""
        frames: list[pd.DataFrame] = []
        for task in async_tasks:
            local_path = self._cache_path(task.s3_uri)
            if not pathlib.Path(local_path).is_file():
                logger.warning(f"Cached file missing for {task.s3_uri}, skipping")
                continue
            try:
                df = self._decode_file(local_path, task)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(f"Failed to decode {local_path}: {exc}")
                continue
            if df is None or df.empty:
                continue
            df.attrs["source"] = self.SOURCE_ID
            frames.append(df)

        if not frames:
            return pd.DataFrame(
                {name: pd.Series(dtype=object) for name in self.SCHEMA.names}
            )[[name for name in schema.names if name in self.SCHEMA.names]]

        result = pd.concat(frames, ignore_index=True)
        return result[[name for name in schema.names if name in result.columns]]

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    def _create_tasks(self, time_list: list[datetime], variable: list[str]) -> list:
        raise NotImplementedError("Subclasses must implement _create_tasks.")

    def _decode_file(self, local_path: str, task: Any) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement _decode_file.")

    # ------------------------------------------------------------------
    # Time validation / cache / fields
    # ------------------------------------------------------------------
    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        """Validate that times align to a 6-hour cycle and are in range."""
        for t in times:
            if t.minute != 0 or t.second != 0 or t.microsecond != 0:
                raise ValueError(
                    f"Requested datetime {t} must be on a whole hour "
                    f"(NNJA cycles are 6-hourly)."
                )
            if t.hour % 6 != 0:
                raise ValueError(
                    f"Requested datetime {t} must align to a 6-hour cycle "
                    f"(00, 06, 12, 18z)."
                )
            if t < cls.MIN_DATE:
                raise ValueError(
                    f"Requested datetime {t} is earlier than {cls.__name__}.MIN_DATE "
                    f"({cls.MIN_DATE.isoformat()})."
                )

    def _cache_path(self, s3_uri: str) -> str:
        """Deterministic cache path for an S3 URI."""
        sha = hashlib.sha256(s3_uri.encode()).hexdigest()
        return os.path.join(self.cache, sha)

    @property
    def cache(self) -> str:
        """Local cache directory for this data source."""
        cache_location = os.path.join(datasource_cache_root(), "nnja")
        if not self._cache:
            if self._tmp_cache_hash is None:
                self._tmp_cache_hash = uuid.uuid4().hex[:8]
            cache_location = os.path.join(
                cache_location, f"tmp_nnja_{self._tmp_cache_hash}"
            )
        return cache_location

    @classmethod
    def resolve_fields(cls, fields: str | list[str] | pa.Schema | None) -> pa.Schema:
        """Resolve ``fields`` into a validated PyArrow schema subset."""
        if fields is None:
            return cls.SCHEMA
        if isinstance(fields, str):
            fields = [fields]
        if isinstance(fields, pa.Schema):
            for f in fields:
                if f.name not in cls.SCHEMA.names:
                    raise KeyError(
                        f"Field '{f.name}' not in {cls.__name__} SCHEMA. "
                        f"Available: {cls.SCHEMA.names}"
                    )
                expected = cls.SCHEMA.field(f.name).type
                if f.type != expected:
                    raise TypeError(
                        f"Field '{f.name}' has type {f.type}, expected "
                        f"{expected} from class SCHEMA"
                    )
            return fields
        selected = []
        for name in fields:
            if name not in cls.SCHEMA.names:
                raise KeyError(
                    f"Field '{name}' not in {cls.__name__} SCHEMA. "
                    f"Available: {cls.SCHEMA.names}"
                )
            selected.append(cls.SCHEMA.field(name))
        return pa.schema(selected)


# ─────────────────────────────────────────────────────────────────────
# Self-contained PrepBUFR decoder (used by NNJAObsConv)
# ─────────────────────────────────────────────────────────────────────


def _safe_int(v: Any) -> int:
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, bytes):
        s = v.decode("ascii", errors="replace").strip()
    elif v is None:
        s = ""
    else:
        s = str(v).strip()
    if not s:
        return 0
    try:
        return int(s)
    except ValueError:
        return 0


def _parse_prepbufr_messages(
    file_data: bytes,
) -> tuple[
    dict[int, tuple[Any, ...]],
    dict[int, tuple[Any, ...]],
    list[tuple[bytes, int]],
]:
    """Split a PrepBUFR byte stream into messages and extract DX tables.

    The first several messages of a PrepBUFR file are DX-table messages
    (dataCategory=11) carrying the NCEP-local Table B / Table D
    descriptor definitions needed to decode subsequent data messages.
    """
    table_b: dict[int, tuple[Any, ...]] = {}
    table_d: dict[int, tuple[Any, ...]] = {}
    data_messages: list[tuple[bytes, int]] = []
    dx_messages: list[bytes] = []

    pos = 0
    while pos < len(file_data):
        idx = file_data.find(b"BUFR", pos)
        if idx == -1:
            break
        msg_len = struct.unpack(">I", b"\x00" + file_data[idx + 4 : idx + 7])[0]
        if msg_len < 8:
            pos = idx + 4
            continue
        msg_bytes = file_data[idx : idx + msg_len]

        # BUFR ed3/4: section-0 = 8 bytes, section-1 octet-9 (offset 16) is dataCategory
        data_cat = file_data[idx + 16] if idx + 16 < len(file_data) else 0
        if data_cat == 11:
            dx_messages.append(msg_bytes)
        else:
            data_messages.append((msg_bytes, data_cat))
        pos = idx + msg_len

    if dx_messages:
        with _silence_bufr_noise():
            try:
                dx_decoder = BufrDecoder()
                for dx_bytes in dx_messages:
                    try:
                        dx_msg = dx_decoder.process(dx_bytes)
                    except Exception:  # noqa: S112
                        logger.debug("Skipping unparseable NNJA DX-table message")
                        continue
                    td = dx_msg.template_data.value
                    dvas = td.decoded_values_all_subsets
                    if not dvas:
                        continue
                    _extract_dx_tables(dvas[0], table_b, table_d)
            except Exception as e:
                logger.warning(f"Failed to extract NNJA DX tables: {e}")

    return table_b, table_d, data_messages


def _extract_dx_tables(
    flat: list[Any],
    table_b: dict[int, tuple[Any, ...]],
    table_d: dict[int, tuple[Any, ...]],
) -> None:
    """Extract NCEP Table B and D entries from a DX-message subset.

    Encoding layout (per NCEP BUFRLIB):

    - n_table_a, [table_a entries (3 fields each) ...]
    - n_table_b, [table_b entries (11 fields each) ...]
    - n_table_d, [table_d entries (variable length) ...]
    """

    def _str(v: Any) -> str:
        if isinstance(v, bytes):
            return v.decode("ascii", errors="replace").strip()
        if v is None:
            return ""
        return str(v).strip()

    def _fxy(f: Any, x: Any, y: Any) -> int:
        return _safe_int(f) * 100000 + _safe_int(x) * 1000 + _safe_int(y)

    n = len(flat)
    idx = 0
    if idx >= n:
        return
    n_a = _safe_int(flat[idx])
    idx += 1
    idx += n_a * 3
    if idx >= n:
        return

    # Table B
    n_b = _safe_int(flat[idx])
    idx += 1
    for _ in range(n_b):
        if idx + 10 >= n:
            return
        f_v = flat[idx]
        x_v = flat[idx + 1]
        y_v = flat[idx + 2]
        mnemonic = _str(flat[idx + 3])
        unit = _str(flat[idx + 5])
        sign_scale = _str(flat[idx + 6])
        scale_s = _str(flat[idx + 7])
        sign_ref = _str(flat[idx + 8])
        ref_s = _str(flat[idx + 9])
        width_s = _str(flat[idx + 10])
        idx += 11

        desc_id = _fxy(f_v, x_v, y_v)
        if desc_id == 0:
            continue
        scale = _safe_int(scale_s)
        if sign_scale == "-":
            scale = -scale
        reference = _safe_int(ref_s)
        if sign_ref == "-":
            reference = -reference
        width = _safe_int(width_s)
        table_b[desc_id] = (
            mnemonic,
            unit,
            scale,
            reference,
            width,
            unit,
            scale,
            max(1, (width + 3) // 4),
        )

    # Table D
    if idx >= n:
        return
    n_d = _safe_int(flat[idx])
    idx += 1
    for _ in range(n_d):
        if idx + 3 >= n:
            return
        f_v = flat[idx]
        x_v = flat[idx + 1]
        y_v = flat[idx + 2]
        seq_mnemonic = _str(flat[idx + 3])
        idx += 4
        seq_id = _fxy(f_v, x_v, y_v)
        if seq_id == 0:
            continue
        if idx >= n:
            return
        n_members = _safe_int(flat[idx])
        idx += 1
        members: list[str] = []
        for _ in range(n_members):
            if idx >= n:
                break
            members.append(_str(flat[idx]))
            idx += 1
        if members:
            table_d[seq_id] = (seq_mnemonic, members)


# Worker globals (set per-process by _init_worker)
_worker_decoder: Any = None


def _init_worker(
    table_b: dict[int, tuple[Any, ...]],
    table_d: dict[int, tuple[Any, ...]],
) -> None:
    """ProcessPoolExecutor initializer: register NCEP DX tables and
    create a per-process pybufrkit decoder. Also redirect this
    worker's stderr to /dev/null to silence the per-message
    sub-centre warnings printed by pybufrkit."""
    global _worker_decoder  # noqa: PLW0603
    # Permanently redirect the worker's stderr; the worker only does
    # BUFR decoding, so legitimate errors come back via exceptions /
    # return values rather than stderr.
    try:
        sys.stderr.flush()
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
    except OSError:
        pass
    TableGroupCacheManager.clear_extra_entries()
    TableGroupCacheManager._TABLE_GROUP_CACHE.invalidate()
    if table_b or table_d:
        TableGroupCacheManager.add_extra_entries(table_b, table_d)
    _worker_decoder = BufrDecoder()


def _decode_message_worker(
    msg_bytes: bytes,
    obs_class: str,
    var_keys: list[tuple[str, str]],
    dt_min: datetime,
    dt_max: datetime,
) -> list[dict[str, Any]]:
    """Decode a single BUFR message in a worker process."""
    return _decode_message(_worker_decoder, msg_bytes, obs_class, var_keys, dt_min, dt_max)


def _decode_message(
    decoder: Any,
    msg_bytes: bytes,
    obs_class: str,
    var_keys: list[tuple[str, str]],
    dt_min: datetime,
    dt_max: datetime,
) -> list[dict[str, Any]]:
    """Decode a single PrepBUFR message and emit observation rows.

    ``var_keys`` is a list of ``(var_name, lexicon_key)`` pairs where
    ``lexicon_key`` is one of ``TOB``, ``QOB``, ``POB``, ``ZOB``,
    ``wind::u``, ``wind::v``.
    """
    try:
        msg = decoder.process(msg_bytes)
    except Exception:
        return []

    n_subsets = msg.n_subsets.value
    if n_subsets == 0:
        return []

    td = msg.template_data.value
    ddas = td.decoded_descriptors_all_subsets
    dvas = td.decoded_values_all_subsets

    msg_year = msg.year.value
    if msg_year < 100:
        msg_year += 2000 if msg_year < 70 else 1900
    try:
        base_time = datetime(
            msg_year, msg.month.value, msg.day.value, msg.hour.value, msg.minute.value
        )
    except (ValueError, OverflowError):
        return []

    rows: list[dict[str, Any]] = []
    for s_idx in range(n_subsets):
        rows.extend(
            _extract_subset(
                ddas[s_idx], dvas[s_idx], base_time, obs_class, var_keys, dt_min, dt_max
            )
        )
    return rows


def _extract_subset(
    descs: list[Any],
    vals: list[Any],
    base_time: datetime,
    obs_class: str,
    var_keys: list[tuple[str, str]],
    dt_min: datetime,
    dt_max: datetime,
) -> list[dict[str, Any]]:
    """Extract observation rows from a single decoded PrepBUFR subset.

    The subset is a flat list of (descriptor, value) pairs.  We first
    walk the header (SID/XOB/YOB/DHR/ELV/TYP) and then iterate over the
    repeated CAT/POB level blocks, emitting one row per (level,
    requested variable) where the variable's descriptor has a
    non-missing value.
    """
    rows: list[dict[str, Any]] = []

    header: dict[str, Any] = {
        "sid": "",
        "xob": None,
        "yob": None,
        "dhr": 0.0,
        "elv": None,
        "typ": None,
    }
    for d, v in zip(descs, vals):
        did = d.id
        if did == _HDR_SID:
            header["sid"] = (
                v.decode("ascii", errors="replace").strip()
                if isinstance(v, bytes)
                else (str(v).strip() if v is not None else "")
            )
        elif did == _HDR_XOB:
            header["xob"] = v
        elif did == _HDR_YOB:
            header["yob"] = v
        elif did == _HDR_DHR:
            header["dhr"] = v if v is not None else 0.0
        elif did == _HDR_ELV:
            header["elv"] = v
        elif did == _HDR_TYP:
            header["typ"] = v
        elif did == _OBS_CAT:
            break

    lat = header["yob"]
    lon = header["xob"]
    if lat is None or lon is None:
        return rows
    if lat < -90.0 or lat > 90.0:
        return rows

    try:
        obs_time = base_time + timedelta(hours=float(header["dhr"]))
    except (ValueError, OverflowError, TypeError):
        obs_time = base_time
    if obs_time < dt_min or obs_time > dt_max:
        return rows

    lon_360 = float(lon) % 360.0

    # Build the per-variable descriptor lookup once
    needed_ids: dict[str, int] = {}
    need_wind = False
    for var_name, key in var_keys:
        if key.startswith("wind::"):
            need_wind = True
        elif key in _MNEMONIC_TO_DESCR:
            needed_ids[var_name] = _MNEMONIC_TO_DESCR[key]

    base_row: dict[str, Any] = {
        "time": obs_time,
        "lat": np.float32(lat),
        "lon": np.float32(lon_360),
        "pres": None,
        "elev": None,
        "type": np.uint16(int(header["typ"])) if header["typ"] is not None else None,
        "class": obs_class if obs_class else None,
        "station": header["sid"] if header["sid"] else None,
        "station_elev": (
            np.float32(header["elv"]) if header["elv"] is not None else None
        ),
    }

    # Walk observation levels: a new POB starts a level
    current: dict[int, Any] = {}
    in_obs = False
    for d, v in zip(descs, vals):
        did = d.id
        if did == _OBS_POB:
            if in_obs and current:
                _emit_level_rows(
                    rows, current, base_row, needed_ids, need_wind, var_keys
                )
            current = {_OBS_POB: v}
            in_obs = True
        elif in_obs and did in _OBSERVATION_DESCR_IDS:
            if did not in current:
                current[did] = v
    if in_obs and current:
        _emit_level_rows(rows, current, base_row, needed_ids, need_wind, var_keys)

    return rows


def _extract_gpsro_subset(
    descs: list[Any],
    vals: list[Any],
    wanted_descrs: dict[int, str],
    dt_min: datetime,
    dt_max: datetime,
) -> list[dict[str, Any]]:
    """Extract observation rows from one GPS RO occultation subset.

    ``wanted_descrs`` maps BUFR descriptor id -> Earth2Studio variable
    name (e.g. ``{15037: "gps", 12001: "gps_t", 13001: "gps_q"}``). For
    each non-missing value of a wanted descriptor encountered in the
    subset's flat (descriptor, value) stream we emit one row.

    The NCEP gpsro encoding lays out the per-level data sequentially as
    three sub-profiles in this order:

    1. Bending-angle profile keyed on ``IMPP`` (descriptor 7040), with
       observation in ``BNDA`` (15037).
    2. Refractivity profile keyed on ``HEIT`` (7007), observation in
       ``ARFR`` (15036).
    3. 1D-Var retrieval profile keyed on ``GPHTST`` (7009), with
       ``PRES`` / ``TMDBST`` / ``SPFH`` (10004 / 12001 / 13001).
    """
    rows: list[dict[str, Any]] = []

    # Header pass
    sat_id: Any = None
    tx_id: Any = None
    qf: Any = None
    lat: float | None = None
    lon: float | None = None
    yyyy = mm = dd = hh = mi = None
    sec: float = 0.0
    for d, v in zip(descs, vals):
        did = d.id
        if did == _GPSRO_SAID:
            sat_id = v
        elif did == _GPSRO_PTID:
            tx_id = v
        elif did == _GPSRO_QFRO:
            qf = v
        elif did == _GPSRO_LAT and v is not None:
            lat = float(v)
        elif did == _GPSRO_LON and v is not None:
            lon = float(v)
        elif did == _GPSRO_YEAR and v is not None:
            yyyy = int(v)
        elif did == _GPSRO_MONTH and v is not None:
            mm = int(v)
        elif did == _GPSRO_DAY and v is not None:
            dd = int(v)
        elif did == _GPSRO_HOUR and v is not None:
            hh = int(v)
        elif did == _GPSRO_MIN and v is not None:
            mi = int(v)
        elif did == _GPSRO_SEC and v is not None:
            try:
                sec = float(v)
            except (TypeError, ValueError):
                sec = 0.0
        elif did == _GPSRO_IMPP:
            break

    if lat is None or lon is None or yyyy is None or mm is None or dd is None:
        return rows
    try:
        obs_time = datetime(yyyy, mm, dd, hh or 0, mi or 0, int(sec))
    except (ValueError, OverflowError):
        return rows
    if obs_time < dt_min or obs_time > dt_max:
        return rows

    lon_360 = lon % 360.0
    station_id = (
        f"{int(sat_id)}_{int(tx_id)}"
        if sat_id is not None and tx_id is not None
        else None
    )

    # Per-level pass
    cur_pres: float | None = None
    cur_height: float | None = None
    cur_impp: float | None = None

    for d, v in zip(descs, vals):
        did = d.id
        if v is None:
            if did == _GPSRO_IMPP:
                cur_impp = None
            elif did == _GPSRO_GPHTST or did == _GPSRO_HEIT:
                cur_height = None
                cur_pres = None
            elif did == _GPSRO_PRES:
                cur_pres = None
            continue

        if did == _GPSRO_IMPP:
            try:
                cur_impp = float(v)
            except (TypeError, ValueError):
                cur_impp = None
            continue
        if did == _GPSRO_GPHTST or did == _GPSRO_HEIT:
            try:
                cur_height = float(v)
            except (TypeError, ValueError):
                cur_height = None
            continue
        if did == _GPSRO_PRES:
            try:
                cur_pres = float(v)
            except (TypeError, ValueError):
                cur_pres = None
            continue

        if did not in wanted_descrs or did not in _GPSRO_OBS_DESCRS:
            continue
        try:
            obs_val = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(obs_val):
            continue

        var_name = wanted_descrs[did]
        if did == _GPSRO_BNDA:
            pres_val = None
            elev_val = np.float32(cur_impp) if cur_impp is not None else None
        else:
            pres_val = np.float32(cur_pres) if cur_pres is not None else None
            elev_val = np.float32(cur_height) if cur_height is not None else None

        rows.append(
            {
                "time": obs_time,
                "lat": np.float32(lat),
                "lon": np.float32(lon_360),
                "pres": pres_val,
                "elev": elev_val,
                "type": np.uint16(int(qf)) if qf is not None else None,
                "class": "GPSRO",
                "station": station_id,
                "station_elev": None,
                "observation": np.float32(obs_val),
                "variable": var_name,
            }
        )

    return rows


def _emit_level_rows(
    rows: list[dict[str, Any]],
    level: dict[int, Any],
    base_row: dict[str, Any],
    needed_ids: dict[str, int],
    need_wind: bool,
    var_keys: list[tuple[str, str]],
) -> None:
    """Append one row per requested variable for the current pressure level."""
    pob = level.get(_OBS_POB)
    pres_val = np.float32(pob) if pob is not None else None  # PrepBUFR mb (lexicon mod converts to Pa for `pres`)

    common = base_row.copy()
    common["pres"] = pres_val

    # Non-wind variables
    for var_name, desc_id in needed_ids.items():
        val = level.get(desc_id)
        if val is None:
            continue
        row = common.copy()
        row["variable"] = var_name
        row["observation"] = np.float32(val)
        rows.append(row)

    # Wind decomposition: u from UOB, v from VOB
    if need_wind:
        uob = level.get(_OBS_UOB)
        vob = level.get(_OBS_VOB)
        if uob is not None and vob is not None:
            for var_name, key in var_keys:
                if key == "wind::u":
                    row = common.copy()
                    row["variable"] = var_name
                    row["observation"] = np.float32(uob)
                    rows.append(row)
                elif key == "wind::v":
                    row = common.copy()
                    row["variable"] = var_name
                    row["observation"] = np.float32(vob)
                    rows.append(row)


# ─────────────────────────────────────────────────────────────────────
# NNJAObsConv
# ─────────────────────────────────────────────────────────────────────


@check_optional_dependencies()
class NNJAObsConv(_NNJAObsBase):
    """NNJA conventional (in-situ + GPS RO) observations data source.

    Reads observations from two complementary NNJA archives based on
    the requested variable's lexicon route:

    - ``u, v, q, t, pres`` -> ``conv/<source>/`` PrepBUFR cycle files
      (default ``source="prepbufr"``).
    - ``gps, gps_t, gps_q`` -> ``gps/gpsro/`` BUFR cycle files
      (bending angle and 1D-Var retrieval temperature / specific
      humidity profiles).

    Returns a :class:`pandas.DataFrame` with one row per (cycle,
    observation level, requested variable). Variable routing is
    handled automatically through
    :py:class:`earth2studio.lexicon.NNJAObsConvLexicon`.

    Parameters
    ----------
    source : {"prepbufr", "convbufr", "prepbufr.acft_profiles"}, optional
        Which encoding family of the NNJA conventional archive to read,
        by default ``"prepbufr"``.
    time_tolerance : TimeTolerance, optional
        Time tolerance window for filtering observations. Accepts a single
        value (symmetric ± window) or a tuple ``(lower, upper)`` for
        asymmetric windows, by default ``np.timedelta64(0, 'm')``.
    max_workers : int, optional
        Max workers in async IO thread pool for concurrent S3 downloads,
        by default 24.
    cache : bool, optional
        Cache downloaded files in the local filesystem cache, by default
        ``True``.
    async_timeout : int, optional
        Total timeout in seconds for the async fetch, by default 600.
    verbose : bool, optional
        Show progress bars, by default ``True``.

    Warning
    -------
    This is a remote data source and may download a large amount of data
    for large requests. A single PrepBUFR cycle file is approximately
    100 MB.

    Note
    ----
    NNJA observation data is distributed under CC BY 4.0; please cite
    https://psl.noaa.gov/data/nnja_obs/ when using this data.

    Additional resources:

    - https://psl.noaa.gov/data/nnja_obs/
    - https://registry.opendata.aws/noaa-reanalyses-obs/
    - https://www.emc.ncep.noaa.gov/mmb/data_processing/prepbufr.doc/document.htm

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        from datetime import datetime, timedelta
        from earth2studio.data import NNJAObsConv

        ds = NNJAObsConv(time_tolerance=timedelta(hours=1))
        df = ds(datetime(2024, 1, 1, 0), ["t", "u", "v"])

        # GPS RO bending angle + 1D-Var retrieval profiles
        df_gps = ds(datetime(2024, 1, 1, 0), ["gps", "gps_t", "gps_q"])

    Badges
    ------
    region:global dataclass:observation product:atmos product:insitu
    """

    SOURCE_ID = "earth2studio.data.NNJAObsConv"
    SCHEMA = _NNJA_CONV_SCHEMA
    MIN_DATE = datetime(1979, 1, 1)

    VALID_SOURCES = frozenset(["prepbufr", "convbufr", "prepbufr.acft_profiles"])

    def __init__(
        self,
        source: str = "prepbufr",
        time_tolerance: TimeTolerance = np.timedelta64(0, "m"),
        max_workers: int = 24,
        cache: bool = True,
        async_timeout: int = 600,
        verbose: bool = True,
    ) -> None:
        if source not in self.VALID_SOURCES:
            raise ValueError(
                f"Invalid source '{source}'. Valid sources: {sorted(self.VALID_SOURCES)}"
            )
        self._source = source
        super().__init__(
            time_tolerance=time_tolerance,
            max_workers=max_workers,
            cache=cache,
            async_timeout=async_timeout,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Task creation
    # ------------------------------------------------------------------
    def _create_tasks(
        self, time_list: list[datetime], variable: list[str]
    ) -> list:
        # Partition variables by lexicon route prefix:
        #   "prepbufr::..." -> conv/prepbufr/ tasks (PrepBUFR decoder)
        #   "gpsro::..."    -> gps/gpsro/ tasks (GPS RO BUFR decoder)
        prepbufr_plan: dict[str, tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = {}
        gpsro_plan: dict[str, tuple[int, Callable[[pd.DataFrame], pd.DataFrame]]] = {}

        for v in variable:
            try:
                source_key, modifier = NNJAObsConvLexicon[v]  # type: ignore[misc]
            except KeyError:
                logger.error(f"Variable id '{v}' not found in NNJAObsConvLexicon")
                raise
            route, _, rest = source_key.partition("::")
            if route == "prepbufr":
                prepbufr_plan[v] = (rest, modifier)
            elif route == "gpsro":
                try:
                    desc_id = int(rest)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid gpsro lexicon entry '{source_key}' for {v}: "
                        f"expected an integer BUFR descriptor id"
                    ) from exc
                gpsro_plan[v] = (desc_id, modifier)
            else:
                raise ValueError(
                    f"Unknown route '{route}' in NNJAObsConvLexicon entry "
                    f"'{source_key}' for variable '{v}' (expected 'prepbufr' or 'gpsro')"
                )

        tasks: list = []

        if prepbufr_plan:
            seen: set[str] = set()
            for t in time_list:
                tmin = t + self._tolerance_lower
                tmax = t + self._tolerance_upper
                day = tmin.replace(minute=0, second=0, microsecond=0)
                day = day.replace(hour=(day.hour // 6) * 6)
                while day <= tmax:
                    uri = self._build_prepbufr_uri(day)
                    if uri not in seen:
                        tasks.append(
                            _NNJAConvTask(
                                s3_uri=uri,
                                datetime_file=day,
                                datetime_min=tmin,
                                datetime_max=tmax,
                                var_plan=prepbufr_plan,
                            )
                        )
                        seen.add(uri)
                    day = day + timedelta(hours=6)

        if gpsro_plan:
            seen = set()
            for t in time_list:
                tmin = t + self._tolerance_lower
                tmax = t + self._tolerance_upper
                day = tmin.replace(minute=0, second=0, microsecond=0)
                day = day.replace(hour=(day.hour // 6) * 6)
                while day <= tmax:
                    uri = self._build_gpsro_uri(day)
                    if uri not in seen:
                        tasks.append(
                            _NNJAGpsRoTask(
                                s3_uri=uri,
                                datetime_file=day,
                                datetime_min=tmin,
                                datetime_max=tmax,
                                var_plan=gpsro_plan,
                            )
                        )
                        seen.add(uri)
                    day = day + timedelta(hours=6)

        return tasks

    def _build_prepbufr_uri(self, cycle: datetime) -> str:
        """Build the NNJA S3 URI for a single PrepBUFR cycle."""
        year_key = cycle.strftime("%Y")
        month_key = cycle.strftime("%m")
        date_key = cycle.strftime("%Y%m%d")
        hour_key = f"{cycle.hour:02d}"
        return (
            f"s3://{NNJA_BUCKET}/{NNJA_PREFIX}/conv/{self._source}/"
            f"{year_key}/{month_key}/{self._source}/"
            f"gdas.{date_key}.t{hour_key}z.{self._source}.nr"
        )

    def _build_gpsro_uri(self, cycle: datetime) -> str:
        """Build the NNJA S3 URI for a single gps/gpsro cycle file."""
        year_key = cycle.strftime("%Y")
        month_key = cycle.strftime("%m")
        date_key = cycle.strftime("%Y%m%d")
        hour_key = f"{cycle.hour:02d}"
        return (
            f"s3://{NNJA_BUCKET}/{NNJA_PREFIX}/gps/gpsro/"
            f"{year_key}/{month_key}/bufr/"
            f"gdas.{date_key}.t{hour_key}z.gpsro.tm00.bufr_d"
        )

    # Back-compat alias used by tests that targeted the v1 method name.
    def _build_uri(self, cycle: datetime) -> str:
        return self._build_prepbufr_uri(cycle)

    # ------------------------------------------------------------------
    # File decode (dispatch by task type)
    # ------------------------------------------------------------------
    def _decode_file(self, local_path: str, task) -> pd.DataFrame:
        if isinstance(task, _NNJAGpsRoTask):
            return self._decode_gpsro_file(local_path, task)
        return self._decode_prepbufr_file(local_path, task)

    def _decode_prepbufr_file(
        self, local_path: str, task: _NNJAConvTask
    ) -> pd.DataFrame:
        """Decode a PrepBUFR cycle file into a DataFrame."""
        with open(local_path, "rb") as fh:
            file_data = fh.read()

        table_b, table_d, messages = _parse_prepbufr_messages(file_data)
        var_keys: list[tuple[str, str]] = [
            (var, plan[0]) for var, plan in task.var_plan.items()
        ]

        work_items: list[tuple[bytes, str]] = [
            (msg_bytes, _PREPBUFR_OBS_TYPES[data_cat])
            for msg_bytes, data_cat in messages
            if data_cat in _PREPBUFR_OBS_TYPES
        ]
        if not work_items:
            return pd.DataFrame(columns=self.SCHEMA.names)

        all_rows: list[dict[str, Any]] = []

        if self._max_workers > 1 and len(work_items) > 1:
            with ProcessPoolExecutor(
                max_workers=self._max_workers,
                initializer=_init_worker,
                initargs=(table_b, table_d),
            ) as pool:
                futures = [
                    pool.submit(
                        _decode_message_worker,
                        msg_bytes,
                        obs_class,
                        var_keys,
                        task.datetime_min,
                        task.datetime_max,
                    )
                    for msg_bytes, obs_class in work_items
                ]
                for future in futures:
                    try:
                        rows = future.result()
                        if rows:
                            all_rows.extend(rows)
                    except Exception:
                        logger.debug("NNJA worker failed to decode a BUFR message")
        else:
            with _silence_bufr_noise():
                TableGroupCacheManager.clear_extra_entries()
                TableGroupCacheManager._TABLE_GROUP_CACHE.invalidate()
                if table_b or table_d:
                    TableGroupCacheManager.add_extra_entries(table_b, table_d)
                decoder = BufrDecoder()
                for msg_bytes, obs_class in work_items:
                    rows = _decode_message(
                        decoder,
                        msg_bytes,
                        obs_class,
                        var_keys,
                        task.datetime_min,
                        task.datetime_max,
                    )
                    all_rows.extend(rows)

        if not all_rows:
            return pd.DataFrame(columns=self.SCHEMA.names)

        df = pd.DataFrame(all_rows)
        # Apply per-variable lexicon modifiers (unit conversions)
        result_frames: list[pd.DataFrame] = []
        for var, (_key, modifier) in task.var_plan.items():
            sub = df[df["variable"] == var].copy()
            if sub.empty:
                continue
            sub = modifier(sub)
            result_frames.append(sub)
        if not result_frames:
            return pd.DataFrame(columns=self.SCHEMA.names)
        df = pd.concat(result_frames, ignore_index=True)

        # Convert pres column from MB to Pa (POB is in MB; the lexicon
        # ``pres`` modifier already handles the observation column for
        # the ``pres`` variable; the ``pres`` schema column is the
        # level pressure shared across all variables and should also
        # be in Pa).
        if "pres" in df.columns:
            df["pres"] = (df["pres"].astype(np.float32) * 100.0).astype(np.float32)

        # Coerce column dtypes to schema
        df["time"] = pd.to_datetime(df["time"])
        for name in self.SCHEMA.names:
            if name not in df.columns:
                df[name] = None
        df = df[list(self.SCHEMA.names)]
        return df

    def _decode_gpsro_file(
        self, local_path: str, task: _NNJAGpsRoTask
    ) -> pd.DataFrame:
        """Decode a single NNJA gps/gpsro cycle BUFR file into a DataFrame.

        The NNJA gpsro files use NCEP-local BUFR descriptors that the
        standard ECMWF eccodes tables do not include.  We instead read
        them with pybufrkit using the DX tables embedded at the start of
        each file (same approach as the PrepBUFR decoder).  Each
        occultation profile becomes one BUFR subset; we walk the
        decoded descriptor list and emit one row per (occultation,
        retrieval / impact-parameter level) for each requested variable.
        """
        with open(local_path, "rb") as fh:
            file_data = fh.read()

        table_b, table_d, messages = _parse_prepbufr_messages(file_data)
        if not messages:
            return pd.DataFrame(columns=self.SCHEMA.names)

        wanted_descrs: dict[int, str] = {
            desc_id: var for var, (desc_id, _mod) in task.var_plan.items()
        }

        # pybufrkit's table cache is process-global; register the
        # extracted DX tables and create a single decoder for sequential
        # decode (gpsro files have far fewer messages than PrepBUFR so
        # we skip the process pool here). The decode loop runs under
        # the C-stderr silencer to suppress the per-message
        # ``Cannot find sub-centre 3`` chatter from pybufrkit.
        all_rows: list[dict[str, Any]] = []
        with _silence_bufr_noise():
            TableGroupCacheManager.clear_extra_entries()
            TableGroupCacheManager._TABLE_GROUP_CACHE.invalidate()
            if table_b or table_d:
                TableGroupCacheManager.add_extra_entries(table_b, table_d)
            decoder = BufrDecoder()

            for msg_bytes, _data_cat in messages:
                try:
                    msg = decoder.process(msg_bytes)
                except Exception:  # noqa: S112
                    continue
                try:
                    n_subsets = msg.n_subsets.value
                except Exception:  # noqa: S112
                    continue
                if n_subsets == 0:
                    continue
                td = msg.template_data.value
                ddas = td.decoded_descriptors_all_subsets
                dvas = td.decoded_values_all_subsets
                for s_idx in range(n_subsets):
                    all_rows.extend(
                        _extract_gpsro_subset(
                            ddas[s_idx],
                            dvas[s_idx],
                            wanted_descrs,
                            task.datetime_min,
                            task.datetime_max,
                        )
                    )

        if not all_rows:
            return pd.DataFrame(columns=self.SCHEMA.names)

        df = pd.DataFrame(all_rows)
        # Apply per-variable lexicon modifiers
        result_frames: list[pd.DataFrame] = []
        for var, (_desc_id, modifier) in task.var_plan.items():
            sub = df[df["variable"] == var].copy()
            if sub.empty:
                continue
            sub = modifier(sub)
            result_frames.append(sub)
        if not result_frames:
            return pd.DataFrame(columns=self.SCHEMA.names)
        df = pd.concat(result_frames, ignore_index=True)

        df["time"] = pd.to_datetime(df["time"])
        for name in self.SCHEMA.names:
            if name not in df.columns:
                df[name] = None
        df = df[list(self.SCHEMA.names)]
        return df


# ─────────────────────────────────────────────────────────────────────
# NNJAObsSat (satellite WMO BUFR via eccodes)
# ─────────────────────────────────────────────────────────────────────


@check_optional_dependencies()
class NNJAObsSat(_NNJAObsBase):
    """NNJA satellite-radiance BUFR observations data source.

    Reads ``gdas.YYYYMMDD.tHHz.<sensor_source>.tm00.bufr_d`` files from
    the NNJA S3 archive
    (``s3://noaa-reanalyses-pds/observations/reanalysis/<sensor>/<source>/``)
    and returns a :class:`pandas.DataFrame` with one row per (cycle,
    field-of-view, channel).

    Parameters
    ----------
    satellites : list[str] | None, optional
        Restrict output to a subset of satellite platforms (e.g.
        ``["n20"]``). ``None`` (default) keeps all platforms carried by
        the requested sensor's source folder, as listed in
        :py:class:`earth2studio.lexicon.NNJASatelliteLexicon`.
    time_tolerance : TimeTolerance, optional
        Time tolerance window for filtering observations, by default
        ``np.timedelta64(0, 'm')``.
    max_workers : int, optional
        Max workers in async IO thread pool for concurrent S3 downloads,
        by default 24.
    cache : bool, optional
        Cache downloaded BUFR files locally, by default ``True``.
    async_timeout : int, optional
        Total timeout in seconds for the async fetch, by default 600.
    verbose : bool, optional
        Show progress bars, by default ``True``.

    Warning
    -------
    This is a remote data source and may download a large amount of data
    for large requests. A single satellite-radiance BUFR cycle file can
    range from a few MB to >100 MB depending on the sensor.

    Note
    ----
    NNJA observation data is distributed under CC BY 4.0; please cite
    https://psl.noaa.gov/data/nnja_obs/ when using this data.

    Additional resources:

    - https://psl.noaa.gov/data/nnja_obs/
    - https://registry.opendata.aws/noaa-reanalyses-obs/

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        from datetime import datetime, timedelta
        from earth2studio.data import NNJAObsSat

        ds = NNJAObsSat(satellites=["n20"], time_tolerance=timedelta(minutes=15))
        df = ds(datetime(2024, 1, 1, 0), ["atms"])

    Badges
    ------
    region:global dataclass:observation product:atmos product:sat
    """

    SOURCE_ID = "earth2studio.data.NNJAObsSat"
    SCHEMA = _NNJA_SAT_SCHEMA
    # Earliest reasonable date — most satellite sensors start later, but
    # we allow any post-1979 cycle and let the S3 fetch surface 404s for
    # platform/cycle combinations that don't exist.
    MIN_DATE = datetime(1979, 1, 1)

    def __init__(
        self,
        satellites: list[str] | None = None,
        time_tolerance: TimeTolerance = np.timedelta64(0, "m"),
        max_workers: int = 24,
        cache: bool = True,
        async_timeout: int = 600,
        verbose: bool = True,
    ) -> None:
        if satellites is not None:
            valid_platforms = {p for v in NNJASatelliteLexicon.VOCAB.values() for p in v.split("::")[2].split(",")}
            invalid = set(satellites) - valid_platforms
            if invalid:
                raise ValueError(
                    f"Invalid satellite(s): {sorted(invalid)}. "
                    f"Valid platforms across NNJASatelliteLexicon: {sorted(valid_platforms)}"
                )
        self._satellites = satellites
        super().__init__(
            time_tolerance=time_tolerance,
            max_workers=max_workers,
            cache=cache,
            async_timeout=async_timeout,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Task creation
    # ------------------------------------------------------------------
    def _create_tasks(
        self, time_list: list[datetime], variable: list[str]
    ) -> list[_NNJASatTask]:
        tasks: list[_NNJASatTask] = []
        seen_uris: set[str] = set()

        for v in variable:
            try:
                key, modifier = NNJASatelliteLexicon[v]  # type: ignore[misc]
            except KeyError:
                logger.error(f"Variable id '{v}' not found in NNJASatelliteLexicon")
                raise
            sensor, source, platforms_csv, bufr_key = key.split("::")
            platforms = tuple(platforms_csv.split(","))
            if self._satellites is not None:
                platforms = tuple(p for p in platforms if p in self._satellites)
                if not platforms:
                    logger.warning(
                        f"No requested satellites available for {v}; skipping"
                    )
                    continue

            for t in time_list:
                tmin = t + self._tolerance_lower
                tmax = t + self._tolerance_upper
                day = tmin.replace(minute=0, second=0, microsecond=0)
                day = day.replace(hour=(day.hour // 6) * 6)
                while day <= tmax:
                    uri = self._build_uri(day, sensor, source)
                    task = _NNJASatTask(
                        s3_uri=uri,
                        datetime_file=day,
                        datetime_min=tmin,
                        datetime_max=tmax,
                        sensor=sensor,
                        source=source,
                        platforms=platforms,
                        bufr_key=bufr_key,
                        e2s_obs_name=v,
                        modifier=modifier,
                    )
                    # NB: same URI may legitimately recur if the user
                    # requests overlapping times — keep the per-task
                    # state, dedupe only on download level.
                    tasks.append(task)
                    seen_uris.add(uri)
                    day = day + timedelta(hours=6)
        return tasks

    def _build_uri(self, cycle: datetime, sensor: str, source: str) -> str:
        """Build the NNJA S3 URI for a single satellite-radiance BUFR cycle."""
        year_key = cycle.strftime("%Y")
        month_key = cycle.strftime("%m")
        date_key = cycle.strftime("%Y%m%d")
        hour_key = f"{cycle.hour:02d}"
        token = source  # NNJA filenames embed the source-folder token
        return (
            f"s3://{NNJA_BUCKET}/{NNJA_PREFIX}/{sensor}/{source}/"
            f"{year_key}/{month_key}/bufr/"
            f"gdas.{date_key}.t{hour_key}z.{token}.tm00.bufr_d"
        )

    def _handle_missing_file(self, path: str) -> None:
        """Satellite cycles routinely have missing platform files; warn instead of error."""
        logger.warning(f"NNJA satellite file {path} not found, skipping")

    # ------------------------------------------------------------------
    # File decode
    # ------------------------------------------------------------------
    def _decode_file(self, local_path: str, task: _NNJASatTask) -> pd.DataFrame:
        """Decode a satellite-radiance BUFR file into a DataFrame.

        We iterate every BUFR message in the file with eccodes, request
        the canonical observation arrays (lat/lon/angles/time) and the
        sensor-specific radiance key from the lexicon, then emit one
        row per (FOV, channel) tuple.
        """
        rows: list[dict[str, Any]] = []
        # The whole eccodes loop runs under the C-stderr silencer:
        # NNJA satellite BUFR uses NCEP-local descriptors that
        # eccodes does not ship tables for, so each call would
        # otherwise emit ``ECCODES ERROR : unable to get
        # descriptor`` lines straight to fd-2. We still detect
        # decode failures via the Python exceptions raised by
        # ``codes_set`` and ``codes_get*``.
        with _silence_bufr_noise(), open(local_path, "rb") as fh:
            while True:
                msgid = eccodes.codes_bufr_new_from_file(fh)
                if msgid is None:
                    break
                try:
                    try:
                        eccodes.codes_set(msgid, "unpack", 1)
                    except Exception:  # noqa: S112
                        continue

                    n_subsets = self._safe_get(msgid, "numberOfSubsets", default=0)
                    if not n_subsets:
                        continue

                    lat = self._safe_get_array(msgid, "latitude")
                    lon = self._safe_get_array(msgid, "longitude")
                    if lat is None or lon is None or lat.size == 0:
                        continue

                    obs_flat = self._safe_get_array(msgid, task.bufr_key)
                    if obs_flat is None or obs_flat.size == 0:
                        continue

                    n_fov = lat.size
                    if obs_flat.size % n_fov != 0:
                        logger.debug(
                            f"Unexpected obs-array size {obs_flat.size} for "
                            f"{n_fov} FOVs in {local_path}; skipping message"
                        )
                        continue
                    n_channels = obs_flat.size // n_fov
                    obs = obs_flat.reshape(n_channels, n_fov).T  # (n_fov, n_channels)

                    solza = self._safe_get_array(msgid, "solarZenithAngle", n_fov)
                    solaza = self._safe_get_array(msgid, "solarAzimuth", n_fov)
                    sat_za = self._safe_get_array(msgid, "satelliteZenithAngle", n_fov)
                    sat_aza = self._safe_get_array(msgid, "bearingOrAzimuth", n_fov)

                    # Channel index — array of length n_channels if present
                    chan_arr = self._safe_get_array(msgid, "channelNumber")
                    if chan_arr is None or chan_arr.size != n_channels:
                        chan_arr = np.arange(1, n_channels + 1, dtype=np.uint16)

                    # Satellite identifier (scalar or per-FOV)
                    sat_id_arr = self._safe_get_array(msgid, "satelliteIdentifier")
                    if sat_id_arr is None or sat_id_arr.size == 0:
                        sat_name = task.platforms[0] if task.platforms else ""
                    elif sat_id_arr.size == 1:
                        sat_name = _SAT_ID_MAP.get(
                            int(sat_id_arr[0]),
                            task.platforms[0] if task.platforms else "",
                        )
                    else:
                        # Fall back: use first id
                        sat_name = _SAT_ID_MAP.get(
                            int(sat_id_arr[0]),
                            task.platforms[0] if task.platforms else "",
                        )

                    # Filter by requested satellites
                    if (
                        self._satellites is not None
                        and sat_name
                        and sat_name not in self._satellites
                    ):
                        continue

                    # Time fields (broadcast scalars)
                    years = self._safe_get_time_array(msgid, "year", n_fov, default=task.datetime_file.year)
                    months = self._safe_get_time_array(msgid, "month", n_fov, default=task.datetime_file.month)
                    days = self._safe_get_time_array(msgid, "day", n_fov, default=task.datetime_file.day)
                    hours = self._safe_get_time_array(msgid, "hour", n_fov, default=task.datetime_file.hour)
                    minutes = self._safe_get_time_array(msgid, "minute", n_fov, default=0)
                    seconds = self._safe_get_time_array(msgid, "second", n_fov, default=0)

                    for i in range(n_fov):
                        try:
                            obs_time = datetime(
                                int(years[i]),
                                int(months[i]),
                                int(days[i]),
                                int(hours[i]),
                                int(minutes[i]),
                                int(seconds[i]),
                            )
                        except (ValueError, OverflowError):
                            continue
                        if (
                            obs_time < task.datetime_min
                            or obs_time > task.datetime_max
                        ):
                            continue
                        for ch in range(n_channels):
                            raw = float(obs[i, ch])
                            if not np.isfinite(raw) or raw > 1e8:
                                continue
                            rows.append(
                                {
                                    "time": obs_time,
                                    "lat": float(lat[i]),
                                    "lon": float(lon[i]) % 360.0,
                                    "scan_angle": None,
                                    "channel_index": int(chan_arr[ch]),
                                    "solza": float(solza[i]) if solza is not None else None,
                                    "solaza": float(solaza[i]) if solaza is not None else None,
                                    "satellite_za": float(sat_za[i]) if sat_za is not None else None,
                                    "satellite_aza": float(sat_aza[i]) if sat_aza is not None else None,
                                    "satellite": sat_name,
                                    "observation": raw,
                                    "variable": task.e2s_obs_name,
                                }
                            )
                finally:
                    eccodes.codes_release(msgid)

        if not rows:
            return pd.DataFrame(columns=self.SCHEMA.names)

        df = pd.DataFrame(rows)
        df = task.modifier(df)

        # Coerce dtypes
        df["time"] = pd.to_datetime(df["time"])
        for col, dtype in (
            ("lat", np.float32),
            ("lon", np.float32),
            ("scan_angle", np.float32),
            ("solza", np.float32),
            ("solaza", np.float32),
            ("satellite_za", np.float32),
            ("satellite_aza", np.float32),
            ("observation", np.float32),
        ):
            if col in df.columns and df[col].notna().any():
                df[col] = df[col].astype(dtype)
        if "channel_index" in df.columns:
            df["channel_index"] = df["channel_index"].astype(np.uint16)
        df = df[list(self.SCHEMA.names)]
        return df

    # ------------------------------------------------------------------
    # eccodes helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_get(msgid: int, key: str, default: Any = None) -> Any:
        try:
            return eccodes.codes_get(msgid, key)
        except Exception:
            return default

    @staticmethod
    def _safe_get_array(
        msgid: int, key: str, broadcast_to: int | None = None
    ) -> np.ndarray | None:
        try:
            arr = eccodes.codes_get_array(msgid, key)
        except Exception:
            return None
        if arr is None:
            return None
        if broadcast_to is not None and arr.size == 1 and broadcast_to > 1:
            arr = np.full(broadcast_to, arr[0])
        return arr

    @staticmethod
    def _safe_get_time_array(
        msgid: int, key: str, n_fov: int, default: int = 0
    ) -> np.ndarray:
        try:
            arr = eccodes.codes_get_array(msgid, key)
        except Exception:
            arr = np.full(n_fov, default)
        if arr.size == 1:
            arr = np.full(n_fov, arr[0])
        elif arr.size != n_fov:
            arr = np.full(n_fov, default)
        return arr.astype(int)
