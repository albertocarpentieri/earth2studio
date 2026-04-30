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

"""Debug script for :py:class:`earth2studio.data.NNJAObsSat`.

This is **not** a unit test or a gallery example. It is a probe that
walks the satellite decode pipeline one step at a time so we can see
exactly where a given sensor (e.g. ``iasi`` / ``crisfsr`` / ``mhs``)
fails to produce rows.

For each requested variable id the script:

1. Resolves the lexicon entry to ``(sensor, source, platforms, bufr_key)``.
2. Builds the NNJA S3 URI for the requested cycle and verifies it
   exists in the public bucket (``head_object`` via ``s3fs``).
3. Downloads the cycle file to the standard NNJA cache.
4. Splits it into BUFR messages with the same parser the data source
   uses (``_parse_prepbufr_messages``) and reports DX-table sizes.
5. For a small sample of messages, runs the **eccodes** path used by
   the data source (``_extract_eccodes_message``) and records why each
   message either succeeded, was skipped, or raised.
6. For the same sample, runs the **pybufrkit fallback** path
   (``_extract_satellite_pybufrkit``) and records:

   - whether the observation BUFR descriptor (e.g. ``12063`` for
     brightness temperature, ``14046`` for IASI scaled radiance) was
     present in the message,
   - which lat/lon/satellite-id descriptors decoded,
   - the per-FOV / per-channel shapes pybufrkit recovered.
7. Calls the full ``NNJAObsSat`` data source end-to-end so the actual
   row count for the cycle is reported alongside the per-message
   diagnostics.

Run::

    python examples/05_data_assimilation/debug_nnja_obs_sat.py \
        --variables iasi crisfsr mhs atms \
        --time 2024-06-01T00 \
        --sample-messages 5
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import pathlib
import sys
import time as time_mod
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import s3fs
from loguru import logger

from earth2studio.data import NNJAObsSat
from earth2studio.data.nnja import (
    _SAT_BUFR_KEY_TO_FXY,
    _SAT_FXY_CHAN_NUM,
    _SAT_FXY_LATITUDE,
    _SAT_FXY_LONGITUDE,
    _SAT_FXY_SATID,
    _NNJASatTask,
    _extract_satellite_pybufrkit,
    _paired_channel_obs,
    _parse_prepbufr_messages,
    _register_dx_tables,
)
from earth2studio.lexicon import NNJASatelliteLexicon

try:
    import eccodes  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    eccodes = None  # type: ignore[assignment]

try:
    from pybufrkit.decoder import Decoder as BufrDecoder
except ImportError:  # pragma: no cover
    BufrDecoder = None  # type: ignore[assignment,misc]


NNJA_BUCKET = "noaa-reanalyses-pds"
NNJA_PREFIX = "observations/reanalysis"


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _build_uri(cycle: datetime, sensor: str, source: str) -> str:
    """Mirror :py:meth:`NNJAObsSat._build_uri` (kept inline so the script
    is independent of any future refactor of that method)."""
    return (
        f"s3://{NNJA_BUCKET}/{NNJA_PREFIX}/{sensor}/{source}/"
        f"{cycle:%Y}/{cycle:%m}/bufr/"
        f"gdas.{cycle:%Y%m%d}.t{cycle:%H}z.{source}.tm00.bufr_d"
    )


def _s3_head(fs: s3fs.S3FileSystem, uri: str) -> dict[str, Any] | None:
    """Return ``info(uri)`` or ``None`` if the object is missing."""
    try:
        info = fs.info(uri)
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.warning(f"  S3 info raised: {exc!r}")
        return None
    return info


def _make_task(
    cycle: datetime,
    sensor: str,
    source: str,
    platforms: tuple[str, ...],
    bufr_key: str,
    e2s_obs_name: str,
) -> _NNJASatTask:
    """Build a synthetic ``_NNJASatTask`` matching what ``NNJAObsSat``
    would produce for a single (cycle, variable)."""
    return _NNJASatTask(
        s3_uri=_build_uri(cycle, sensor, source),
        datetime_file=cycle,
        datetime_min=cycle - timedelta(hours=3),
        datetime_max=cycle + timedelta(hours=3),
        sensor=sensor,
        source=source,
        platforms=platforms,
        bufr_key=bufr_key,
        e2s_obs_name=e2s_obs_name,
        modifier=lambda df: df,  # identity
    )


@contextlib.contextmanager
def _capture_eccodes_errors() -> "contextlib.AbstractContextManager[list[str]]":
    """Capture C-level eccodes/pybufrkit stderr noise per message.

    The data source silences these, but here we *want* them so we can
    show the actual underlying errors (e.g.
    ``HashArrayNoMatchError(-37): unable to get descriptor 311192``).
    """
    sys.stderr.flush()
    saved = os.dup(2)
    r, w = os.pipe()
    os.dup2(w, 2)
    os.close(w)
    captured: list[str] = []
    try:
        yield captured
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)
        os.close(r)
        # We can't read the pipe without risking deadlocks on large
        # buffers, so the capture list is left empty here. We rely on
        # the Python exception from eccodes for the actual error, and
        # only suppress noise to keep the report readable.


def _eccodes_probe(msg_bytes: bytes, bufr_key: str) -> dict[str, Any]:
    """Run the eccodes decode path on one message and return a small
    diagnostic record."""
    if eccodes is None:
        return {"status": "eccodes-missing"}
    msgid = None
    try:
        msgid = eccodes.codes_new_from_message(msg_bytes)
        if msgid is None:
            return {"status": "codes_new_from_message returned None"}
        try:
            eccodes.codes_set(msgid, "unpack", 1)
        except Exception as exc:
            return {"status": f"unpack failed: {exc!r}"}

        n_subsets = 0
        try:
            n_subsets = int(eccodes.codes_get(msgid, "numberOfSubsets"))
        except Exception:
            pass

        try:
            obs = eccodes.codes_get_array(msgid, bufr_key)
            obs_size = int(obs.size) if obs is not None else 0
        except Exception as exc:
            return {
                "status": f"missing key '{bufr_key}': {exc!r}",
                "n_subsets": n_subsets,
            }
        return {
            "status": "ok",
            "n_subsets": n_subsets,
            "obs_size": obs_size,
        }
    except Exception as exc:
        return {"status": f"raised: {exc!r}"}
    finally:
        if msgid is not None:
            with contextlib.suppress(Exception):
                eccodes.codes_release(msgid)


def _pybufrkit_probe(
    decoder: Any,
    msg_bytes: bytes,
    bufr_key: str,
    cycle: datetime,
) -> dict[str, Any]:
    """Run the pybufrkit decode path on one message and return a
    detailed diagnostic record."""
    if decoder is None:
        return {"status": "pybufrkit-missing"}
    try:
        msg = decoder.process(msg_bytes)
    except Exception as exc:
        return {"status": f"process failed: {exc!r}"}

    try:
        n_subsets = int(msg.n_subsets.value)
    except Exception as exc:
        return {"status": f"n_subsets unavailable: {exc!r}"}

    if not n_subsets:
        return {"status": "n_subsets=0"}

    try:
        td = msg.template_data.value
        ddas = td.decoded_descriptors_all_subsets
        dvas = td.decoded_values_all_subsets
    except Exception as exc:
        return {"status": f"template_data unavailable: {exc!r}"}

    descs0 = ddas[0]
    vals0 = dvas[0]
    desc_ids = [getattr(d, "id", None) for d in descs0]
    desc_counter = Counter(desc_ids)

    obs_fxy = _SAT_BUFR_KEY_TO_FXY.get(bufr_key, ())
    obs_present_ids = sorted({d for d in desc_ids if d in obs_fxy})
    n_obs_vals = sum(
        1
        for d, v in zip(descs0, vals0)
        if getattr(d, "id", None) in obs_fxy and v is not None
    )

    lat_present = sorted({d for d in desc_ids if d in _SAT_FXY_LATITUDE})
    lon_present = sorted({d for d in desc_ids if d in _SAT_FXY_LONGITUDE})
    sat_present = sorted({d for d in desc_ids if d in _SAT_FXY_SATID})

    # Pairing helper used by the production extractor. Lets us see how
    # many (channel, obs) pairs the message actually yields and how
    # many of those carry a non-missing radiance.
    paired_chans, paired_obs = _paired_channel_obs(
        descs0, vals0, obs_fxy, _SAT_FXY_CHAN_NUM
    )
    n_paired = len(paired_chans)
    n_paired_finite = sum(1 for v in paired_obs if v is not None)

    # Now try the same end-to-end decode the data source does.
    decoded = _extract_satellite_pybufrkit(
        msg, bufr_key, cycle.year, cycle.month, cycle.day, cycle.hour
    )

    return {
        "status": "ok" if decoded is not None else "extract returned None",
        "n_subsets": n_subsets,
        "n_descriptors": len(desc_ids),
        "obs_fxy_expected": list(obs_fxy),
        "obs_fxy_found": obs_present_ids,
        "n_obs_values_first_subset": n_obs_vals,
        "n_paired_chan_obs": n_paired,
        "n_paired_obs_finite": n_paired_finite,
        "lat_descr_found": lat_present,
        "lon_descr_found": lon_present,
        "satid_descr_found": sat_present,
        "extract_n_fov": None if decoded is None else int(decoded["n_fov"]),
        "extract_n_channels": None if decoded is None else int(decoded["n_channels"]),
        "extract_obs_finite": (
            None
            if decoded is None
            else int(np.isfinite(decoded["obs"]).sum())
        ),
        "extract_lat_finite": (
            None
            if decoded is None
            else int(np.isfinite(decoded["lat"]).sum())
        ),
        # Top descriptor frequencies — useful for spotting whether the
        # message is the expected sensor template at all.
        "top_descriptors": desc_counter.most_common(8),
    }


def _print_dict(prefix: str, record: dict[str, Any]) -> None:
    """Pretty-print a diagnostic dict with stable key order."""
    width = max((len(k) for k in record.keys()), default=0)
    for k, v in record.items():
        logger.info(f"{prefix}{k:<{width}} = {v}")


# ─────────────────────────────────────────────────────────────────────
# Per-variable probe
# ─────────────────────────────────────────────────────────────────────


def _plot_coverage(
    df: pd.DataFrame,
    var_id: str,
    cycle: datetime,
    out_path: str,
) -> None:
    """Render a global Robinson scatter map of the fetched observations.

    Imports of cartopy / matplotlib happen here so the rest of the
    script can run on machines without those optional plotting deps.
    """
    import cartopy.crs as ccrs  # type: ignore[import-untyped]
    import matplotlib.pyplot as plt

    sub = df.dropna(subset=["lat", "lon", "observation"])
    if sub.empty:
        logger.warning(f"  plot: no rows with finite (lat, lon, observation)")
        return

    n = len(sub)
    if n > 200_000:
        sub = sub.sample(n=200_000, random_state=0)

    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines(linewidth=0.4)
    ax.gridlines(linewidth=0.3, alpha=0.4)

    sc = ax.scatter(
        sub["lon"].to_numpy(),
        sub["lat"].to_numpy(),
        c=sub["observation"].to_numpy(),
        s=2,
        alpha=0.6,
        cmap="viridis",
        transform=ccrs.PlateCarree(),
    )
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
    cb.set_label("observation")

    sats = sorted(df["satellite"].dropna().unique().tolist())
    chans = sorted(df["channel_index"].dropna().unique().tolist())
    ax.set_title(
        f"NNJAObsSat / {var_id} @ {cycle.isoformat()}Z\n"
        f"rows={n:,}  sats={sats}  channels={chans[:8]}{'…' if len(chans) > 8 else ''}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  wrote {out_path}")


def probe_variable(
    var_id: str,
    cycle: datetime,
    fs: s3fs.S3FileSystem,
    sample_messages: int,
    skip_decode: bool,
    skip_end_to_end: bool,
    plot_dir: str | None,
) -> None:
    """Walk the full download → parse → eccodes → pybufrkit → fetch
    pipeline for one variable and print diagnostics at every step."""
    logger.info("")
    logger.info("=" * 78)
    logger.info(f"VARIABLE: {var_id}")
    logger.info("=" * 78)

    # 1. Lexicon resolution.
    try:
        vocab_key, _modifier = NNJASatelliteLexicon[var_id]
    except KeyError:
        logger.error(f"'{var_id}' not in NNJASatelliteLexicon; skipping")
        return
    sensor, source, platforms_csv, bufr_key = vocab_key.split("::")
    platforms = tuple(platforms_csv.split(","))
    logger.info(f"lexicon       : {vocab_key}")
    logger.info(
        f"  sensor={sensor}  source={source}  bufr_key={bufr_key}  "
        f"platforms={list(platforms)}"
    )
    logger.info(
        f"  obs FXY ids the pybufrkit fallback will look for: "
        f"{_SAT_BUFR_KEY_TO_FXY.get(bufr_key)}"
    )

    # 2. S3 head check.
    uri = _build_uri(cycle, sensor, source)
    logger.info(f"s3 uri        : {uri}")
    info = _s3_head(fs, uri)
    if info is None:
        logger.error("  -> object NOT FOUND in NNJA bucket; nothing to decode")
        return
    size_mb = info["size"] / (1024 * 1024)
    logger.info(f"  size={size_mb:,.1f} MB  last_modified={info.get('LastModified')}")

    # 3. Build the task and rely on the public NNJAObsSat to do the
    #    download (so we hit the same cache + retry behaviour the
    #    production path would use).
    task = _make_task(cycle, sensor, source, platforms, bufr_key, var_id)
    ds = NNJAObsSat(time_tolerance=(timedelta(hours=-3), timedelta(hours=3)))
    cache_path = ds._cache_path(task.s3_uri)  # type: ignore[attr-defined]
    pathlib.Path(ds.cache).mkdir(parents=True, exist_ok=True)
    if not pathlib.Path(cache_path).is_file():
        logger.info("  not in cache; downloading…")
        t0 = time_mod.perf_counter()

        async def _download() -> None:
            await ds._async_init()  # type: ignore[attr-defined]
            await ds._fetch_remote_file(task.s3_uri)  # type: ignore[attr-defined]

        asyncio.run(_download())
        logger.info(
            f"  downloaded in {time_mod.perf_counter() - t0:.1f}s -> {cache_path}"
        )
    else:
        logger.info(f"  cache hit -> {cache_path}")

    if skip_decode:
        return

    # 4. Parse messages + DX tables.
    with open(cache_path, "rb") as fh:
        file_data = fh.read()
    table_b, table_d, messages = _parse_prepbufr_messages(file_data)
    logger.info(
        f"messages      : {len(messages):,}   DX tables: B={len(table_b)} D={len(table_d)}"
    )
    if not messages:
        logger.error("  -> file contains zero data messages; nothing to decode")
        return

    sample = messages[: max(1, sample_messages)]

    # 5. eccodes path.
    logger.info(f"eccodes path  : sampling {len(sample)} message(s)")
    ec_outcomes: Counter = Counter()
    for i, (msg_bytes, _data_cat) in enumerate(sample):
        with _capture_eccodes_errors():
            rec = _eccodes_probe(msg_bytes, bufr_key)
        ec_outcomes[rec["status"][:60]] += 1
        if i < 3:  # full detail for the first 3
            logger.info(f"  msg #{i}:")
            _print_dict("    ", rec)
    logger.info(f"  eccodes status histogram: {dict(ec_outcomes)}")

    # 6. pybufrkit path.
    if BufrDecoder is None:
        logger.warning("pybufrkit not installed; skipping fallback probe")
    else:
        logger.info(f"pybufrkit path: sampling {len(sample)} message(s)")
        _register_dx_tables(table_b, table_d)
        decoder = BufrDecoder()
        py_outcomes: Counter = Counter()
        for i, (msg_bytes, _data_cat) in enumerate(sample):
            rec = _pybufrkit_probe(decoder, msg_bytes, bufr_key, cycle)
            py_outcomes[rec["status"][:60]] += 1
            if i < 3:
                logger.info(f"  msg #{i}:")
                _print_dict("    ", rec)
        logger.info(f"  pybufrkit status histogram: {dict(py_outcomes)}")

    # 7. Full data source call (small window so it doesn't run forever).
    if skip_end_to_end:
        logger.info("end-to-end    : skipped (--skip-end-to-end)")
        return
    logger.info("end-to-end    : NNJAObsSat call")
    t0 = time_mod.perf_counter()
    try:
        df = ds(cycle, [var_id])
    except Exception as exc:
        logger.error(f"  data source raised: {exc!r}")
        return
    elapsed = time_mod.perf_counter() - t0
    if df.empty:
        logger.warning(f"  -> EMPTY DataFrame after {elapsed:.1f}s")
        return
    logger.info(
        f"  -> {len(df):,} rows in {elapsed:.1f}s  "
        f"satellites={sorted(df['satellite'].dropna().unique().tolist())}  "
        f"channels={sorted(df['channel_index'].dropna().unique().tolist())[:8]}…"
    )
    with pd.option_context("display.max_columns", None, "display.width", 160):
        logger.info(f"  preview:\n{df.head(3)}")

    if plot_dir is not None:
        pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
        cycle_tag = cycle.strftime("%Y%m%dT%H")
        out_path = os.path.join(plot_dir, f"debug_nnja_{var_id}_{cycle_tag}.jpg")
        _plot_coverage(df, var_id, cycle, out_path)


# ─────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["iasi", "crisfsr", "mhs", "atms"],
        help=(
            "NNJASatelliteLexicon variable ids to probe. Default targets "
            "the three failing sensors plus ATMS as a known-good control."
        ),
    )
    parser.add_argument(
        "--time",
        default="2024-06-01T00",
        help="Cycle datetime (UTC, ISO 8601). Must align to 00/06/12/18z.",
    )
    parser.add_argument(
        "--sample-messages",
        type=int,
        default=5,
        help="Number of BUFR messages to inspect per file, by default 5",
    )
    parser.add_argument(
        "--skip-decode",
        action="store_true",
        help=(
            "Stop after the S3 head + download step. Useful for "
            "quickly verifying that paths exist for a long list of "
            "variables without paying the decode cost."
        ),
    )
    parser.add_argument(
        "--skip-end-to-end",
        action="store_true",
        help=(
            "Run per-message diagnostics but skip the full NNJAObsSat "
            "fetch at the end. Useful for very large files (e.g. IASI "
            "~800 MB, CrIS-FSR ~1.4 GB) where the full pybufrkit decode "
            "takes minutes."
        ),
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help=(
            "If set, write a global Robinson scatter map of the fetched "
            "observations to this directory (one JPG per variable). "
            "Requires cartopy + matplotlib."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cycle = datetime.fromisoformat(args.time)
    if cycle.hour % 6 != 0 or cycle.minute or cycle.second:
        raise SystemExit(
            f"Cycle {cycle} must align to a 6-hour synoptic time (00/06/12/18z)"
        )

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    fs = s3fs.S3FileSystem(anon=True)
    logger.info(f"Probing cycle {cycle.isoformat()}Z for {len(args.variables)} variable(s)")
    logger.info(f"Variables: {args.variables}")

    for var in args.variables:
        try:
            probe_variable(
                var,
                cycle,
                fs,
                args.sample_messages,
                args.skip_decode,
                args.skip_end_to_end,
                args.plot_dir,
            )
        except Exception:
            logger.exception(f"Unexpected error while probing '{var}'")


if __name__ == "__main__":
    main()
