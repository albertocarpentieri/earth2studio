"""Run HealDA on NNJA observations -- simple, readable pipeline.

Steps
-----
1. Load the HealDA model (CPU first, then move to GPU).
2. Fetch the selected observation sources (``--sources``):

   conv (fed into ``conv_obs``):
     - ``nnja``   -- PrepBUFR conventional obs (t/q/pres/u/v)
     - ``gpsro``  -- GPS Radio Occultation (gps/gps_t/gps_q)

   sat (fed into ``sat_obs``):
     - ``atms``   -- JPSS ATMS radiances, NOAA AWS (free)
     - ``mhs``    -- MetOp MHS radiances, EUMETSAT (needs key)
     - ``amsua``  -- MetOp AMSU-A radiances, EUMETSAT (needs key)

3. Run: ``model(conv_obs=conv_df, sat_obs=sat_df)``.
4. Save NetCDF to ``outputs/<YYYYMMDD_HH>/healda_ic_<sources>[_e2b].nc``.

Conv decoder
------------
By default ``nnja`` and ``gpsro`` are fetched via
:class:`earth2studio.data.NNJAObsConv` (pybufrkit decoder).
Pass ``--earth2bufr`` to use the Rust-based ``earth2bufr`` decoder
(~10× faster, but requires ``pip install earth2bufr``).

Usage
-----
  python run_healda_nnja.py
  python run_healda_nnja.py --analysis-time 2024-01-01T00
  python run_healda_nnja.py --sources nnja gpsro atms   # no EUMETSAT
  python run_healda_nnja.py --sources atms               # sat-only
  python run_healda_nnja.py --earth2bufr                 # fast conv decoder
  python run_healda_nnja.py --earth2bufr --cache-dir /tmp/nnja_cache
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

os.makedirs("outputs", exist_ok=True)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "earth2studio"))

from earth2studio.data import (  # noqa: E402
    JPSS_ATMS,
    MetOpAMSUA,
    MetOpMHS,
    NNJAObsConv,
    fetch_dataframe,
)
from earth2studio.models.da import HealDA  # noqa: E402


# ===========================================================================
# Source registry
# ===========================================================================
@dataclass
class _Source:
    kind: str                # "conv" or "sat"
    label: str               # human-readable name for logs
    build: Callable          # window -> earth2studio DataFrameSource
    variables: list[str]     # variables to request from the source
    requires_env: str | None = None


def _build_sources(window: tuple[timedelta, timedelta]) -> dict[str, _Source]:
    return {
        "nnja": _Source(
            kind="conv",
            label="NNJA conv (PrepBUFR)",
            build=lambda w: NNJAObsConv(time_tolerance=w),
            variables=["t", "q", "pres", "u", "v"],
        ),
        # GPS Radio Occultation uses the same NNJAObsConv class but the
        # lexicon routes gps/gps_t/gps_q to the gps/gpsro/ cycle files.
        "gpsro": _Source(
            kind="conv",
            label="NNJA GPSRO",
            build=lambda w: NNJAObsConv(time_tolerance=w),
            variables=["gps", "gps_t", "gps_q"],
        ),
        "atms": _Source(
            kind="sat",
            label="JPSS ATMS",
            build=lambda w: JPSS_ATMS(satellites=["n20", "npp"], time_tolerance=w),
            variables=["atms"],
        ),
        "mhs": _Source(
            kind="sat",
            label="MetOp MHS",
            build=lambda w: MetOpMHS(time_tolerance=w),
            variables=["mhs"],
            requires_env="EUMETSAT_CONSUMER_KEY",
        ),
        "amsua": _Source(
            kind="sat",
            label="MetOp AMSU-A",
            build=lambda w: MetOpAMSUA(time_tolerance=w),
            variables=["amsua"],
            requires_env="EUMETSAT_CONSUMER_KEY",
        ),
    }


SOURCE_NAMES = ("nnja", "gpsro", "atms", "mhs", "amsua")


# ===========================================================================
# Channel-index helpers
#
# HealDA's sat schema uses ``channel_index`` (the raw channel number it
# was trained on). The JPSS/MetOp sources publish the same quantity as
# ``sensor_index``. For ATMS/MHS/AMSU-A the numbering scheme is identical,
# but the sources include ALL instrument channels while HealDA was trained
# on only a subset.  We:
#   1. Ask sources for ``sensor_index`` instead of ``channel_index``.
#   2. Rename the returned column to ``channel_index``.
#   3. Drop rows whose channel isn't in HealDA's trained set (raw_to_local
#      LUT; entries == 0 are untrained).
# ===========================================================================
def filter_trained_channels(df: pd.DataFrame, model: "HealDA") -> pd.DataFrame:
    """Keep only rows whose ``channel_index`` HealDA was trained on."""
    if "channel_index" not in df.columns or "variable" not in df.columns:
        return df
    keep: list[pd.DataFrame] = []
    for sensor, sdf in df.groupby("variable", sort=False):
        if sensor not in model._sensor_stats:
            keep.append(sdf)
            continue
        raw_to_local = np.asarray(model._sensor_stats[sensor]["raw_to_local"])
        raw_ch = sdf["channel_index"].to_numpy(dtype=int)
        max_raw = len(raw_to_local) - 1
        valid = (
            (raw_ch >= 0)
            & (raw_ch <= max_raw)
            & (raw_to_local[np.clip(raw_ch, 0, max_raw)] > 0)
        )
        keep.append(sdf.loc[valid])
    return pd.concat(keep, ignore_index=True)


# ===========================================================================
# Earth2bufr-based NNJA conv decoder
# Only used when --earth2bufr is passed.
# Implements the same PrepBUFR + GPSRO fetch as NNJAObsConv but calls
# earth2bufr (Rust) instead of pybufrkit (Python) for BUFR decoding.
# ===========================================================================
_NNJA_BUCKET = "noaa-reanalyses-pds"
_NNJA_PREFIX = "observations/reanalysis"

# PrepBUFR mnemonic → (HealDA variable name, unit conversion)
_PREP_VARS: dict[str, tuple[str, Callable[[np.ndarray], np.ndarray]]] = {
    "POB": ("pres", lambda x: x.astype(np.float32) * np.float32(100.0)),  # mb → Pa
    "TOB": ("t",    lambda x: x.astype(np.float32) + np.float32(273.15)),  # C → K
    "QOB": ("q",    lambda x: x.astype(np.float32) * np.float32(1e-6)),    # mg/kg → kg/kg
    "UOB": ("u",    lambda x: x.astype(np.float32)),
    "VOB": ("v",    lambda x: x.astype(np.float32)),
}
_PREP_QM = {"POB": "PQM", "TOB": "TQM", "QOB": "QQM", "UOB": "WQM", "VOB": "WQM"}
_PREP_CAT_FILTER = [100, 101, 102, 104, 105, 107, 109, 110, 119, 120, 121]

# GPSRO BUFR mnemonic → HealDA variable name
_GPS_MNEMONICS = {"BNDA": "gps", "TMDBST": "gps_t", "SPFH": "gps_q"}


def _col(df: pd.DataFrame, *names: str) -> pd.Series | None:
    """Return the first matching column as a numeric Series, or None.

    earth2bufr appends unit strings to column names (e.g. "ELV [m]"), so we
    match by the base name (first whitespace-delimited token) in addition to
    exact match.
    """
    # Build a prefix→actual-column map once per call.
    prefix_map: dict[str, str] = {c.split()[0]: c for c in df.columns}
    for n in names:
        if not n:
            continue
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
        if n in prefix_map:
            return pd.to_numeric(df[prefix_map[n]], errors="coerce")
    return None


def _e2b_cycle_anchors(
    t0: datetime,
    lo: timedelta,
    hi: timedelta,
) -> list[datetime]:
    """Return 6-hourly GDAS cycle anchors that cover [t0+lo, t0+hi]."""
    t_lo = t0 + lo
    t_hi = t0 + hi
    start = t_lo.replace(minute=0, second=0, microsecond=0,
                         hour=(t_lo.hour // 6) * 6)
    anchors: list[datetime] = []
    cur = start
    while cur <= t_hi:
        anchors.append(cur)
        cur += timedelta(hours=6)
    # Drop the first anchor: with the canonical ±3 h NCEP window it only
    # contributes data before t0+lo (i.e. outside the requested window).
    return anchors[1:] if anchors else anchors


def _e2b_s3_fetch(uri: str, cache_dir: Path) -> Path | None:
    """Download a public S3 object to *cache_dir*; return None on failure."""
    import fsspec

    local = cache_dir / Path(uri.split("s3://", 1)[-1].replace("/", "_"))
    if local.exists() and local.stat().st_size > 0:
        return local
    try:
        fsspec.filesystem("s3", anon=True).get(uri, str(local))
        return local
    except Exception as exc:
        logger.warning(f"  S3 fetch failed for {Path(uri).name}: {exc}")
        return None


def _e2b_decode_prepbufr(
    path: Path,
    cycle_dt: datetime,
    wanted: set[str],
    window_h: float = 3.0,
) -> pd.DataFrame:
    """Decode one PrepBUFR file with earth2bufr into a HealDA-compatible DataFrame."""
    import earth2bufr

    batches = earth2bufr.read_prepbufr(
        str(path),
        data_category_filter=_PREP_CAT_FILTER,
        as_pandas=True,
        flatten=True,
    )
    frames: list[pd.DataFrame] = []
    for batch in batches:
        lat = _col(batch, "YOB")
        lon = _col(batch, "XOB")
        dhr = _col(batch, "DHR")
        if any(s is None for s in (lat, lon, dhr)):
            continue
        lon_w = ((lon + 180.0) % 360.0) - 180.0
        times = pd.to_datetime(cycle_dt) + pd.to_timedelta(dhr, unit="h")
        pob = _col(batch, "POB")
        elv = _col(batch, "ELV", "ZOB")
        typ = _col(batch, "TYP")
        for mnemonic, (var_name, convert) in _PREP_VARS.items():
            if var_name not in wanted:
                continue
            obs = _col(batch, mnemonic)
            if obs is None:
                continue
            qm = _col(batch, _PREP_QM.get(mnemonic, ""))
            mask = obs.notna() & lat.notna() & lon.notna() & dhr.notna()
            if not mask.any():
                continue
            n = int(mask.sum())
            pres_pa = (
                (pob[mask] * np.float32(100.0)).to_numpy(dtype=np.float32)
                if pob is not None and mnemonic != "POB"
                else np.full(n, np.nan, dtype=np.float32)
            )
            frames.append(pd.DataFrame({
                "lat":         lat[mask].to_numpy(dtype=np.float32),
                "lon":         lon_w[mask].to_numpy(dtype=np.float32),
                "time":        times[mask].to_numpy(),
                "elev":        elv[mask].to_numpy(dtype=np.float32) if elv is not None else np.full(n, np.nan, dtype=np.float32),
                "pres":        pres_pa,
                "observation": convert(obs[mask].to_numpy()),
                "type":        typ[mask].fillna(0).to_numpy(dtype=np.uint16) if typ is not None else np.zeros(n, dtype=np.uint16),
                "variable":    var_name,
            }))
        del batch
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    t_min = pd.Timestamp(cycle_dt - timedelta(hours=window_h))
    t_max = pd.Timestamp(cycle_dt + timedelta(hours=window_h))
    return df[(df["time"] >= t_min) & (df["time"] < t_max)].reset_index(drop=True)


def _e2b_decode_gpsro(
    path: Path,
    cycle_dt: datetime,
    wanted: set[str],
    window_h: float = 3.0,
) -> pd.DataFrame:
    """Decode one GPSRO BUFR file with earth2bufr into a HealDA-compatible DataFrame."""
    import earth2bufr
    from earth2studio.data.utils_bufr import parse_prepbufr_messages

    # NNJA GPSRO files embed NCEP-specific BUFR table overrides. We extract
    # them and hand them to earth2bufr so it can decode the file correctly.
    with open(path, "rb") as fh:
        raw = fh.read()
    table_b, table_d, _ = parse_prepbufr_messages(raw, silence_noise=True)

    def _key(d: int) -> str:
        return f"{d // 100000}{(d // 1000) % 100:02d}{d % 1000:03d}"

    tb, td = {}, {}
    for desc_id, entry in table_b.items():
        try:
            mnemonic, unit, scale, reference, width = entry[:5]
        except (TypeError, ValueError):
            continue
        tb[_key(int(desc_id))] = {
            "mnemonic": mnemonic or f"B{_key(int(desc_id))}",
            "name": mnemonic or "",
            "units": unit or "NUMERIC",
            "scale": int(scale),
            "reference_value": int(reference),
            "bit_width": int(width),
        }
    for desc_id, entry in table_d.items():
        try:
            mnemonic, members = entry
        except (TypeError, ValueError):
            continue
        mids = []
        for m in members:
            s = str(m).strip().replace("-", "")
            if len(s) == 6 and s.isdigit():
                f, x, y = int(s[0]), int(s[1:3]), int(s[3:])
                mids.append((f << 14) | (x << 8) | y)
        if mids:
            td[_key(int(desc_id))] = {
                "mnemonic": mnemonic or f"D{_key(int(desc_id))}",
                "descriptors": mids,
            }

    local_json = (
        earth2bufr.create_local_tables(
            table_b_overrides=tb or None,
            table_d_overrides=td or None,
        )
        if (tb or td) else None
    )
    batches = earth2bufr.read_bufr(
        str(path), as_pandas=True, flatten=True, local_tables_json=local_json
    )
    if not batches:
        return pd.DataFrame()

    df_raw = pd.concat(batches, ignore_index=True, sort=False)
    lat = _col(df_raw, "CLATH", "CLAT", "latitude")
    lon = _col(df_raw, "CLONH", "CLON", "longitude")
    if lat is None or lon is None:
        return pd.DataFrame()
    lon = ((lon + 180.0) % 360.0) - 180.0

    minute_col = _col(df_raw, "MINU", "MINUTE")
    second_col = _col(df_raw, "SECO", "SECOND")
    parts = {
        "year":   _col(df_raw, "YEAR"),
        "month":  _col(df_raw, "MNTH", "MONTH"),
        "day":    _col(df_raw, "DAYS", "DAY"),
        "hour":   _col(df_raw, "HOUR"),
        "minute": minute_col if minute_col is not None else pd.Series(0, index=df_raw.index),
        "second": second_col if second_col is not None else pd.Series(0, index=df_raw.index),
    }
    if not all(parts[k] is not None for k in ("year", "month", "day", "hour")):
        return pd.DataFrame()
    times = pd.to_datetime(pd.DataFrame(parts), errors="coerce")

    pres = _col(df_raw, "PRLC", "pressure")
    elev = _col(df_raw, "GPHTST", "nonCoordinateGeopotentialHeight", "HEIT", "IMPP")

    frames: list[pd.DataFrame] = []
    for mnemonic, var_name in _GPS_MNEMONICS.items():
        if var_name not in wanted:
            continue
        obs = _col(df_raw, mnemonic)
        if obs is None:
            continue
        mask = obs.notna() & times.notna() & lat.notna() & lon.notna()
        if not mask.any():
            continue
        n = int(mask.sum())
        frames.append(pd.DataFrame({
            "lat":         lat[mask].to_numpy(dtype=np.float32),
            "lon":         lon[mask].to_numpy(dtype=np.float32),
            "time":        times[mask].to_numpy(),
            "elev":        elev[mask].to_numpy(dtype=np.float32) if elev is not None else np.full(n, np.nan, dtype=np.float32),
            "pres":        pres[mask].to_numpy(dtype=np.float32) if pres is not None else np.full(n, np.nan, dtype=np.float32),
            "observation": obs[mask].to_numpy(dtype=np.float32),
            "type":        np.zeros(n, dtype=np.uint16),
            "variable":    var_name,
        }))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    t_min = pd.Timestamp(cycle_dt - timedelta(hours=window_h))
    t_max = pd.Timestamp(cycle_dt + timedelta(hours=window_h))
    return df[(df["time"] >= t_min) & (df["time"] < t_max)].reset_index(drop=True)


def fetch_conv_earth2bufr(
    wanted_conv_sources: list[str],
    analysis_time: np.ndarray,
    window: tuple[timedelta, timedelta],
    cache_dir: Path,
) -> pd.DataFrame:
    """Fetch NNJA conv/gpsro data using earth2bufr as the BUFR decoder.

    Parameters
    ----------
    wanted_conv_sources : list[str]
        Subset of ``["nnja", "gpsro"]`` to include.
    analysis_time : np.ndarray
        Shape-(1,) array with the analysis datetime64.
    window : tuple[timedelta, timedelta]
        (lower, upper) obs window bounds relative to analysis_time.
    cache_dir : Path
        Directory for caching downloaded BUFR files.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = pd.Timestamp(analysis_time[0]).to_pydatetime()
    anchors = _e2b_cycle_anchors(t0, window[0], window[1])
    logger.info(
        f"earth2bufr: {len(anchors)} cycle(s)  sources={wanted_conv_sources}"
    )

    prep_wanted = {"t", "q", "pres", "u", "v"} if "nnja" in wanted_conv_sources else set()
    gps_wanted  = {"gps", "gps_t", "gps_q"} if "gpsro" in wanted_conv_sources else set()

    all_frames: list[pd.DataFrame] = []
    for cycle in anchors:
        logger.info(f"  cycle {cycle:%Y-%m-%d %HZ}")
        if prep_wanted:
            uri = (
                f"s3://{_NNJA_BUCKET}/{_NNJA_PREFIX}/conv/prepbufr/"
                f"{cycle:%Y}/{cycle:%m}/prepbufr/"
                f"gdas.{cycle:%Y%m%d}.t{cycle:%H}z.prepbufr.nr"
            )
            path = _e2b_s3_fetch(uri, cache_dir)
            if path:
                df = _e2b_decode_prepbufr(path, cycle, prep_wanted)
                if not df.empty:
                    logger.info(f"    PrepBUFR: {len(df):,} rows")
                    all_frames.append(df)

        if gps_wanted:
            uri = (
                f"s3://{_NNJA_BUCKET}/{_NNJA_PREFIX}/gps/gpsro/"
                f"{cycle:%Y}/{cycle:%m}/bufr/"
                f"gdas.{cycle:%Y%m%d}.t{cycle:%H}z.gpsro.tm00.bufr_d"
            )
            path = _e2b_s3_fetch(uri, cache_dir)
            if path:
                df = _e2b_decode_gpsro(path, cycle, gps_wanted)
                if not df.empty:
                    logger.info(f"    GPSRO: {len(df):,} rows")
                    all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()
    out = pd.concat(all_frames, ignore_index=True, sort=False)
    logger.info(f"earth2bufr conv total: {len(out):,} rows")
    return out


# ===========================================================================
# Datetime parsing
# ===========================================================================
def parse_dt(s: str) -> np.datetime64:
    """Accept ``YYYY-MM-DDTHH`` (or ``YYYY-MM-DD``) and return a numpy datetime."""
    for fmt in ("%Y-%m-%dT%H", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H", "%Y-%m-%d"):
        try:
            return np.datetime64(datetime.strptime(s, fmt))
        except ValueError:
            continue
    raise ValueError(f"bad datetime: {s!r}")


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--analysis-time", default="2024-01-01T00",
        help="GDAS cycle to analyze (default: 2024-01-01T00)",
    )
    ap.add_argument(
        "--sources",
        nargs="+",
        choices=SOURCE_NAMES,
        default=list(SOURCE_NAMES),
        metavar="SOURCE",
        help=(
            f"Observation sources: {{{', '.join(SOURCE_NAMES)}}}. "
            "Default: all. 'nnja'/'gpsro' → conv_obs; 'atms'/'mhs'/'amsua' → sat_obs."
        ),
    )
    ap.add_argument(
        "--earth2bufr", action="store_true",
        help=(
            "Use the earth2bufr Rust decoder for conv sources (nnja, gpsro) "
            "instead of pybufrkit. Requires: pip install earth2bufr."
        ),
    )
    ap.add_argument(
        "--cache-dir", default=None,
        help=(
            "Local directory for caching downloaded BUFR files "
            "(earth2bufr path only). Default: system temp dir."
        ),
    )
    args = ap.parse_args()

    analysis_time = np.array([parse_dt(args.analysis_time)])
    window = (timedelta(hours=-21), timedelta(hours=3))
    selected = list(dict.fromkeys(args.sources))  # deduplicate, preserve order

    logger.info(f"Analysis time : {str(analysis_time[0])[:16]} UTC")
    logger.info(f"Obs window    : {window}")
    logger.info(f"Sources       : {selected}")
    logger.info(f"Conv decoder  : {'earth2bufr' if args.earth2bufr else 'pybufrkit (NNJAObsConv)'}")

    # ------------------------------------------------------------------
    # 1. Load HealDA on CPU.
    # ------------------------------------------------------------------
    logger.info("Loading HealDA model package...")
    package = HealDA.load_default_package()
    model = HealDA.load_model(package, lat_lon=True)
    conv_schema, sat_schema = model.input_coords()

    # Translate ``channel_index`` (HealDA name) → ``sensor_index`` (source name)
    # for the field request. We rename + filter after fetching (see below).
    sat_fields_raw = list(sat_schema.keys())
    sat_fields_for_source = np.array(
        ["sensor_index" if f == "channel_index" else f for f in sat_fields_raw]
    )
    sat_needs_rename = "channel_index" in sat_fields_raw

    # NNJAObsConv stores station elevation in "station_elev", not "elev".
    # We request it additionally so we can fill the HealDA-expected "elev"
    # column (which is always NaN for PrepBUFR obs otherwise).
    conv_fields_base = list(conv_schema.keys())
    conv_fields_with_stelev = np.array(conv_fields_base + ["station_elev"])

    fields_by_kind = {
        "conv": conv_fields_with_stelev,
        "sat":  sat_fields_for_source,
    }

    # ------------------------------------------------------------------
    # 2. Fetch observations.
    # ------------------------------------------------------------------
    registry = _build_sources(window)
    frames_by_kind: dict[str, list[pd.DataFrame]] = {"conv": [], "sat": []}

    # Conv sources: use earth2bufr or NNJAObsConv depending on flag.
    conv_sources = [s for s in selected if registry[s].kind == "conv"]
    if conv_sources:
        if args.earth2bufr:
            cache_dir = Path(args.cache_dir) if args.cache_dir else Path(tempfile.gettempdir()) / "nnja_e2b_cache"
            conv_df = fetch_conv_earth2bufr(conv_sources, analysis_time, window, cache_dir)
            if not conv_df.empty:
                conv_df.attrs = {"request_time": analysis_time}
                frames_by_kind["conv"].append(conv_df)
        else:
            for name in conv_sources:
                src = registry[name]
                logger.info(f"Fetching {src.label}...")
                try:
                    df = fetch_dataframe(
                        src.build(window),
                        time=analysis_time,
                        variable=np.array(src.variables),
                        fields=fields_by_kind["conv"],
                    )
                except Exception as exc:
                    logger.warning(f"  {src.label} failed ({type(exc).__name__}: {exc}); skipping")
                    continue
                # NNJAObsConv puts station elevation in "station_elev", not "elev".
                # HealDA's QC requires height (= elev) to be non-NaN, so fill
                # the NaN "elev" column from "station_elev" before dropping it.
                if "station_elev" in df.columns:
                    df = df.copy()
                    df["elev"] = df["elev"].fillna(df["station_elev"])
                    df = df.drop(columns=["station_elev"])
                logger.info(f"  {src.label}: {len(df):,} rows")
                frames_by_kind["conv"].append(df)

    # Sat sources always go through the earth2studio data sources.
    for name in [s for s in selected if registry[s].kind == "sat"]:
        src = registry[name]
        if src.requires_env and not os.environ.get(src.requires_env):
            logger.warning(f"Skipping {src.label}: env var {src.requires_env} not set")
            continue
        logger.info(f"Fetching {src.label}...")
        try:
            df = fetch_dataframe(
                src.build(window),
                time=analysis_time,
                variable=np.array(src.variables),
                fields=fields_by_kind["sat"],
            )
        except Exception as exc:
            logger.warning(f"  {src.label} failed ({type(exc).__name__}: {exc}); skipping")
            continue
        # Rename sensor_index → channel_index, then drop untrained channels.
        if sat_needs_rename and "sensor_index" in df.columns:
            df = df.rename(columns={"sensor_index": "channel_index"})
        if "channel_index" in df.columns:
            n_before = len(df)
            df = filter_trained_channels(df, model)
            logger.info(
                f"  {src.label}: {n_before:,} → {len(df):,} rows "
                f"(dropped {n_before - len(df):,} untrained channels)"
            )
        else:
            logger.info(f"  {src.label}: {len(df):,} rows")
        frames_by_kind["sat"].append(df)

    def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame | None:
        if not frames:
            return None
        out = pd.concat(frames, ignore_index=True)
        out.attrs = {"request_time": analysis_time}
        return out

    conv_df = _concat(frames_by_kind["conv"])
    sat_df  = _concat(frames_by_kind["sat"])

    if conv_df is None and sat_df is None:
        logger.error("No observations fetched (all selected sources empty).")
        sys.exit(1)

    logger.info(
        f"conv rows: {0 if conv_df is None else len(conv_df):,}   "
        f"sat rows:  {0 if sat_df is None else len(sat_df):,}"
    )

    # ------------------------------------------------------------------
    # 3. Run HealDA.
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        logger.info("Moving HealDA to CUDA...")
        model = model.to("cuda:0")
    else:
        logger.warning("CUDA not available; running on CPU (very slow).")

    torch.manual_seed(42)
    logger.info("Running HealDA inference...")
    result = model(conv_obs=conv_df, sat_obs=sat_df)
    logger.info(f"  result shape: {result.shape}")

    if hasattr(result.data, "get"):
        result = result.copy(data=result.data.get())

    # ------------------------------------------------------------------
    # 4. Save NetCDF.
    #
    # Filename encodes the sources used and the decoder, so outputs from
    # different runs don't overwrite each other:
    #   healda_ic_nnja-gpsro-atms_e2b.nc  (earth2bufr)
    #   healda_ic_nnja-gpsro-atms.nc      (pybufrkit)
    #   healda_ic_atms-mhs-amsua.nc       (sat-only)
    # ------------------------------------------------------------------
    sources_tag = "-".join(selected)
    decoder_tag = "_e2b" if args.earth2bufr else ""
    nc_name = f"healda_ic_{sources_tag}{decoder_tag}.nc"

    ts = pd.Timestamp(analysis_time[0]).to_pydatetime()
    out_dir = Path(f"outputs/{ts:%Y%m%d_%H}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nc = out_dir / nc_name
    result.to_netcdf(out_nc)
    logger.info(f"Wrote {out_nc}")
    logger.info("DONE")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
