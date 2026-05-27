"""Compare one or more HealDA NetCDF analyses against ERA5.

For each input file the script:
  1. Reads the HealDA analysis (produced by ``run_healda_nnja.py``).
  2. Fetches ERA5 from CDS for the same valid time and variables.
  3. Interpolates ERA5 to the HealDA lat/lon grid.
  4. Plots a 3-row comparison:  HealDA / ERA5 / (HealDA − ERA5).
  5. Writes per-variable metrics (MAE, RMSE, bias) to a CSV file.

Outputs (written next to each input NetCDF unless ``--output-dir`` is set):
  <stem>_vs_era5.jpg   -- 3-row comparison figure
  <stem>_vs_era5.csv   -- per-variable MAE / RMSE / bias table

Usage
-----
  # Single file
  python compare_healda_era5.py outputs/20240101_00/healda_ic_nnja-gpsro-atms.nc

  # Glob / multiple files
  python compare_healda_era5.py outputs/20240101_00/*.nc

  # Custom variables and output directory
  python compare_healda_era5.py outputs/**/*.nc \\
      --vars t2m z500 u10m v10m msl \\
      --output-dir plots/

Environment variables
---------------------
  CDSAPI_KEY   CDS API key (see https://cds.climate.copernicus.eu/how-to-api)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from tqdm import tqdm

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

sys.path.insert(0, str(Path(__file__).parent / "earth2studio"))

from earth2studio.data import CDS  # noqa: E402


# ---------------------------------------------------------------------------
# Plotting config
# ---------------------------------------------------------------------------
PROJECTION = ccrs.Robinson()

# Default colormaps and symmetric difference ranges per variable.
_CMAP: dict[str, str] = {
    "t2m":  "Spectral_r",
    "z500": "PRGn",
    "u10m": "RdBu_r",
    "v10m": "RdBu_r",
    "msl":  "viridis",
}
_DIFF_RANGE: dict[str, tuple[float, float]] = {
    "t2m":  (-10,   10),
    "z500": (-500, 500),
    "u10m": (-10,   10),
    "v10m": (-10,   10),
    "msl":  (-500, 500),
}
_DEFAULT_CMAP   = "viridis"
_DEFAULT_DIFF   = 0.05  # fraction of field range used as ±diff limit if not in table


def _cmap(var: str) -> str:
    return _CMAP.get(var, _DEFAULT_CMAP)


def _diff_range(var: str, healda: np.ndarray) -> tuple[float, float]:
    if var in _DIFF_RANGE:
        return _DIFF_RANGE[var]
    lim = float(np.abs(healda).max()) * _DEFAULT_DIFF
    return (-lim, lim)


def _to_numpy(arr) -> np.ndarray:
    """CuPy → NumPy if needed."""
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparison(
    healda_da: xr.DataArray,
    era5_da: xr.DataArray,
    variables: list[str],
    analysis_time: np.datetime64,
    src_label: str,
    out_path: Path,
) -> None:
    """3-row (HealDA / ERA5 / Difference) comparison figure."""
    healda_vars = list(healda_da.coords["variable"].values)
    era5_vars   = list(era5_da.coords["variable"].values)

    n_vars = len(variables)
    fig, axes = plt.subplots(
        3, n_vars,
        subplot_kw={"projection": PROJECTION},
        figsize=(7 * n_vars, 12),
        squeeze=False,
    )

    ts_str = pd.Timestamp(analysis_time).strftime("%Y-%m-%d %H UTC")

    for col, var in enumerate(variables):
        if var not in healda_vars:
            logger.warning(f"  '{var}' not in HealDA output, skipping column")
            for row in range(3):
                axes[row, col].set_visible(False)
            continue
        if var not in era5_vars:
            logger.warning(f"  '{var}' not in ERA5, skipping column")
            for row in range(3):
                axes[row, col].set_visible(False)
            continue

        lat = healda_da.coords["lat"].values
        lon = healda_da.coords["lon"].values

        h_field = _to_numpy(healda_da.sel(variable=var).values.squeeze())
        e_field = _to_numpy(era5_da.sel(variable=var).values.squeeze())
        diff    = h_field - e_field

        vmin = float(min(np.nanmin(h_field), np.nanmin(e_field)))
        vmax = float(max(np.nanmax(h_field), np.nanmax(e_field)))
        cmap = _cmap(var)
        dmin, dmax = _diff_range(var, h_field)

        mae  = float(np.nanmean(np.abs(diff)))
        rmse = float(np.sqrt(np.nanmean(diff ** 2)))
        bias = float(np.nanmean(diff))

        def _map(ax, field, cm, vlo, vhi, title):
            im = ax.pcolormesh(
                lon, lat, field,
                transform=ccrs.PlateCarree(),
                cmap=cm, vmin=vlo, vmax=vhi,
            )
            ax.coastlines(linewidth=0.4)
            ax.gridlines(linewidth=0.2, alpha=0.4)
            ax.set_global()
            fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
            ax.set_title(title, fontsize=11)

        _map(axes[0, col], h_field, cmap, vmin, vmax, f"HealDA ({src_label})  {var}")
        _map(axes[1, col], e_field, cmap, vmin, vmax, f"ERA5  {var}")
        _map(
            axes[2, col], diff, "RdBu_r", dmin, dmax,
            f"HealDA − ERA5  {var}\nMAE={mae:.3f}  RMSE={rmse:.3f}  bias={bias:.3f}",
        )

    fig.suptitle(
        f"HealDA ({src_label}) vs ERA5  —  {ts_str}",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(
    healda_da: xr.DataArray,
    era5_da: xr.DataArray,
    variables: list[str],
    nc_path: Path,
    analysis_time: np.datetime64,
) -> pd.DataFrame:
    """Compute MAE, RMSE, bias for each variable."""
    healda_vars = list(healda_da.coords["variable"].values)
    era5_vars   = list(era5_da.coords["variable"].values)
    rows = []
    for var in variables:
        if var not in healda_vars or var not in era5_vars:
            continue
        h = _to_numpy(healda_da.sel(variable=var).values.squeeze()).astype(np.float64)
        e = _to_numpy(era5_da.sel(variable=var).values.squeeze()).astype(np.float64)
        diff = h - e
        rows.append({
            "file":            nc_path.name,
            "analysis_time":   str(analysis_time)[:16],
            "variable":        var,
            "healda_mean":     float(np.nanmean(h)),
            "era5_mean":       float(np.nanmean(e)),
            "mae":             float(np.nanmean(np.abs(diff))),
            "rmse":            float(np.sqrt(np.nanmean(diff ** 2))),
            "bias":            float(np.nanmean(diff)),
            "max_abs_diff":    float(np.nanmax(np.abs(diff))),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Single-file processing
# ---------------------------------------------------------------------------
def process_file(
    nc_path: Path,
    variables: list[str],
    cds: CDS,
    out_dir: Path | None,
) -> pd.DataFrame:
    """Load HealDA NC, fetch ERA5, plot, return metrics DataFrame."""
    logger.info(f"\n{'='*60}")
    logger.info(f"File: {nc_path.name}")

    # -- Load HealDA result -------------------------------------------------
    # result.to_netcdf() saves an xr.DataArray with a ``variable`` coordinate
    # dimension (74 channels), not a Dataset with one variable per channel.
    # Open with open_dataarray() so we can select with .sel(variable=...).
    healda_da = xr.open_dataarray(nc_path)

    # Extract the analysis time from the file (first time coordinate).
    if "time" in healda_da.coords:
        analysis_time = healda_da.coords["time"].values.flat[0]
    else:
        logger.warning("No 'time' coordinate found; using NaT")
        analysis_time = np.datetime64("NaT")

    ts = pd.Timestamp(analysis_time)
    logger.info(f"Analysis time: {ts}")

    healda_vars = list(healda_da.coords["variable"].values)

    # Determine which requested variables are actually in the file.
    available = [v for v in variables if v in healda_vars]
    missing   = [v for v in variables if v not in healda_vars]
    if missing:
        logger.warning(f"Variables not in HealDA output: {missing}")
    if not available:
        logger.error("None of the requested variables found in the file — skipping")
        return pd.DataFrame()

    # -- Fetch ERA5 from CDS -----------------------------------------------
    time_np = np.array([np.datetime64(ts)])
    logger.info(f"Fetching ERA5 {available} from CDS...")
    try:
        era5_da = cds(time_np, available)
    except Exception as exc:
        logger.error(f"ERA5 fetch failed: {exc}")
        return pd.DataFrame()

    # Interpolate ERA5 to HealDA grid (ERA5 lon is 0–360, HealDA too).
    lat = healda_da.coords["lat"].values
    lon = healda_da.coords["lon"].values
    era5_interp = era5_da.interp(lat=lat, lon=lon, method="nearest")

    # -- Source label from filename (strip common prefix/suffix) -----------
    stem = nc_path.stem
    src_label = stem.replace("healda_ic_", "").replace("healda_nnja_ic_", "")

    # -- Output paths -------------------------------------------------------
    dest = out_dir if out_dir else nc_path.parent
    dest.mkdir(parents=True, exist_ok=True)
    fig_path = dest / f"{stem}_vs_era5.jpg"
    csv_path = dest / f"{stem}_vs_era5.csv"

    # -- Plot ---------------------------------------------------------------
    plot_comparison(healda_da, era5_interp, available, analysis_time, src_label, fig_path)

    # -- Metrics ------------------------------------------------------------
    metrics = compute_metrics(healda_da, era5_interp, available, nc_path, analysis_time)
    metrics.to_csv(csv_path, index=False)
    logger.info(f"  wrote {csv_path}")
    logger.info("  Metrics:")
    for _, row in metrics.iterrows():
        logger.info(
            f"    {row['variable']:8s}  MAE={row['mae']:.4f}  "
            f"RMSE={row['rmse']:.4f}  bias={row['bias']:+.4f}"
        )

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "nc_files",
        nargs="+",
        metavar="NC_FILE",
        help="HealDA NetCDF file(s) to compare against ERA5.",
    )
    ap.add_argument(
        "--vars",
        nargs="+",
        default=["t2m", "z500", "u10m", "v10m", "msl"],
        metavar="VAR",
        help=(
            "Variables to compare. Must be present in both the HealDA output "
            "and the ERA5 lexicon. Default: t2m z500 u10m v10m msl."
        ),
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory for output figures and CSVs. "
            "Default: same directory as each input file."
        ),
    )
    args = ap.parse_args()

    nc_paths = [Path(p) for p in args.nc_files]
    missing = [p for p in nc_paths if not p.exists()]
    if missing:
        logger.error(f"File(s) not found: {missing}")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else None

    logger.info("Initialising CDS data source...")
    cds = CDS(cache=True, verbose=False)

    all_metrics: list[pd.DataFrame] = []
    for nc_path in nc_paths:
        df = process_file(nc_path, list(args.vars), cds, out_dir)
        if not df.empty:
            all_metrics.append(df)

    if len(all_metrics) > 1:
        combined = pd.concat(all_metrics, ignore_index=True)
        dest = out_dir if out_dir else nc_paths[0].parent
        combined_csv = dest / "all_metrics_vs_era5.csv"
        combined.to_csv(combined_csv, index=False)
        logger.info(f"\nCombined metrics → {combined_csv}")

    logger.info("DONE")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
