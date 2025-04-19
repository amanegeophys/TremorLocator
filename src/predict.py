from __future__ import annotations

import argparse
import gc
import glob
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from my_module import SacHandler, SacTrace, SpectrogramGenerator
from my_module.utils import setup_logger

# ────────────────────────────────────────────────────────────
# CONSTANTS (hyper‑parameters that rarely change)
TREMORNET_PATH: Path = Path("../model/spec_to_proba/tremornet.keras")
STATION_FILE: Path = Path("../station/hinet129.txt")
SAC_ROOT: Path = Path("/net/ikkyu/mnt/sda/sac")
AMP_TO_EPI_DIR: Path = Path("../model/amp_to_epicenter")

DBSCAN_EPS: float = 0.5
DBSCAN_MIN_SAMPLES: int = 3
BUNGO_LOC: Tuple[float, float] = (132.1, 33.15)
ORIGIN_LOC: Tuple[float, float] = (135.0, 34.1)  # base point for offsets

AMP_NULL_VALUE: float = 1e-9
TREMOR_THRESHOLD: float = 0.9
# ────────────────────────────────────────────────────────────
logger = setup_logger(__name__)
T = TypeVar("T", bound="SacTrace")


# ╭──────────────────────────────────────────────────────────╮
# │  Utility functions                                      │
# ╰──────────────────────────────────────────────────────────╯
def load_stations(file_path: Path) -> pd.DataFrame:
    """Read station list

    Parameters
    ----------
    file_path
        Path to a whitespace‑separated station table.

    Returns
    -------
    pd.DataFrame
        DataFrame containing at least ``station``, ``lon``, and ``lat``.
    """
    return pd.read_table(file_path, sep=r"\s+")


def rms_amplitude(traces: dict[str, T]) -> np.ndarray:
    """Compute RMS amplitude for three‐component traces.

    Parameters
    ----------
    traces
        Dictionary keyed by ``"NS"``, ``"EW"``, ``"UD"``.

    Returns
    -------
    np.ndarray
        Array ``[NS, EW, UD]`` of RMS values.
    """
    return np.array(
        [np.sqrt(np.mean(traces[c].data ** 2)) for c in ("NS", "EW", "UD")]
    )


def tremor_proba(spec: np.ndarray, model) -> np.ndarray:
    """Predict tremor / noise / earthquake probabilities.

    Returns ``[noise, tremor, eq]``.  If prediction fails, returns
    ``[1, 0, 0]`` so that the trace is treated as noise.

    Notes
    -----
    Any exception is caught and logged (no interruption).
    """
    try:
        return model.predict(spec[np.newaxis, ...], verbose=0)[0]
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("TremorNet failed: %s", exc)
        return np.array([1.0, 0.0, 0.0])


def process_station(
    station: str,
    sac_handler: SacHandler,
    spec_gen: SpectrogramGenerator,
    tremor_model,
    start_time: str,
) -> list[float]:
    """Process one station for a given time window.

    Parameters
    ----------
    station
        Station ID string.
    sac_handler
        SAC file reader.
    spec_gen
        Spectrogram generator.
    tremor_model
        Loaded TremorNet model.
    start_time
        Window start in ``YYYY-mm-dd-HH:MM:SS.ffffff`` format.

    Returns
    -------
    list
        ``[NS_amp, EW_amp, UD_amp, noise_p, tremor_p, eq_p]``
    """
    traces = sac_handler.get_sac_traces(station, start_time)
    if traces is None:  # missing data ⇒ treat as noise
        return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

    spec = spec_gen.generate_spectrograms(traces)
    proba = tremor_proba(spec, tremor_model)
    amp = rms_amplitude(traces)
    return amp.tolist() + proba.tolist()


# ╭──────────────────────────────────────────────────────────╮
# │  Core pipeline                                          │
# ╰──────────────────────────────────────────────────────────╯
def estimate_once(
    stations: pd.DataFrame,
    sac_handler: SacHandler,
    spec_gen: SpectrogramGenerator,
    tremor_model,
    amp_models: list,
    start_time: str,
    max_workers: int,
) -> list[Tuple[str, float, float]]:
    """Run one inference step and return epicenter list.

    Parameters
    ----------
    stations
        Station master table.
    sac_handler, spec_gen, tremor_model
        Shared objects (loaded once).
    amp_models
        List of amplitude → epicenter regression models.
    start_time
        Window start time string.
    max_workers
        ThreadPool size for station‐level I/O.

    Returns
    -------
    list of tuple
        ``[(timestamp, lon, lat, lon_std, lat_std), ...]``.  Empty if no epicenter detected.
    """
    # --- parallel station processing ----------------------------------
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futs = [
            exe.submit(
                process_station,
                st,
                sac_handler,
                spec_gen,
                tremor_model,
                start_time,
            )
            for st in stations["station"].astype(str)
        ]

    results = [f.result() for f in futs]
    cols = ["NS", "EW", "UD", "noise", "tremor", "eq"]
    df = pd.concat([stations, pd.DataFrame(results, columns=cols)], axis=1)

    # --- tremor clustering -------------------------------------------
    tremor_df = df[df["tremor"] >= TREMOR_THRESHOLD].copy()
    if tremor_df.empty:
        return []  # nothing to write

    coords = np.vstack([tremor_df[["lon", "lat"]].values, BUNGO_LOC])
    labels = (
        DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
        .fit(coords)
        .labels_[:-1]
    )
    tremor_df["cluster"] = labels
    df = df.merge(
        tremor_df[["station", "cluster"]], on="station", how="left"
    ).fillna({"cluster": -1})
    df["cluster"] = df["cluster"].astype(int)

    # --- epicenter estimation per cluster ----------------------------
    epics: list[Tuple[str, float, float]] = []
    for lbl in sorted(df["cluster"].unique()):
        if lbl < 0:
            continue
        mask = df["cluster"] == lbl
        if mask.sum() < DBSCAN_MIN_SAMPLES:
            continue

        d = df.copy()
        target = mask | (d["cluster"] == -1)
        d.loc[~target, ["noise", "tremor", "eq"]] = (1.0, 0.0, 0.0)
        d.loc[
            (d["tremor"] >= TREMOR_THRESHOLD) & (d["cluster"] == -1),
            ["NS", "EW", "UD"],
        ] = AMP_NULL_VALUE

        # clip outliers and zero‑std columns
        mean = d[["NS", "EW", "UD"]].mean()
        std = d[["NS", "EW", "UD"]].std().replace(0.0, np.nan)
        cutoff = mean + 3 * std
        for ch in ["NS", "EW", "UD"]:
            d.loc[d[ch] >= cutoff[ch], ch] = AMP_NULL_VALUE
        d.loc[d["tremor"] < TREMOR_THRESHOLD, ["NS", "EW", "UD"]] = (
            AMP_NULL_VALUE
        )

        # z‑score
        amp_mean = d[["NS", "EW", "UD"]].mean()
        amp_std = d[["NS", "EW", "UD"]].std()

        for ch in ["NS", "EW", "UD"]:
            if amp_std[ch] == 0.0:
                d[ch] = 0.0
            else:
                d[ch] = (d[ch] - amp_mean[ch]) / amp_std[ch]

        # inference (batch to save GPU/CPU)
        amp_in = d[["NS", "EW", "UD"]].to_numpy().T[np.newaxis]
        preds = np.array([m.predict(amp_in, verbose=0)[0] for m in amp_models])
        lon, lat = preds.mean(axis=0) + ORIGIN_LOC[::-1]
        lon_std, lat_std = preds.std(axis=0)
        epics.append(
            (
                start_time,
                float(lon),
                float(lat),
                float(lon_std),
                float(lat_std),
            )
        )

    # free large objects explicitly
    del df, tremor_df
    gc.collect()
    return epics


# ╭──────────────────────────────────────────────────────────╮
# │  CLI & entry point                                      │
# ╰──────────────────────────────────────────────────────────╯
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command‑line arguments."""
    p = argparse.ArgumentParser(description="Tremor pipeline runner")
    p.add_argument(
        "--start",
        required=True,
        help="start time (YYYY-mm-dd-HH:MM:SS.ffffff)",
    )
    p.add_argument(
        "--end", required=True, help="end time   (YYYY-mm-dd-HH:MM:SS.ffffff)"
    )
    p.add_argument(
        "--step", type=int, default=1, help="time step in minutes (default 1)"
    )
    p.add_argument("--workers", type=int, default=8, help="thread pool size")
    p.add_argument(
        "--out",
        default="../reports/prediction_results.csv",
        help="output CSV path",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Script entry point."""
    args = parse_args(argv)
    stations = load_stations(STATION_FILE)
    sac_handler = SacHandler(str(SAC_ROOT))
    spec_gen = SpectrogramGenerator()
    tremor_model = load_model(str(TREMORNET_PATH))
    amp_models = [
        load_model(p) for p in glob.iglob(str(AMP_TO_EPI_DIR / "*.keras"))
    ]
    logger.info("Loaded %d amp →  epic models", len(amp_models))

    # CSV header handling
    out_path = Path(args.out)
    header_needed = not out_path.exists()

    # iterate over time range
    time_range = pd.date_range(args.start, args.end, freq=f"{args.step}min")
    for t in tqdm(time_range, desc="Estimating"):
        t_str = t.strftime("%Y-%m-%d-%H:%M:%S.%f")[:-3]
        epics = estimate_once(
            stations,
            sac_handler,
            spec_gen,
            tremor_model,
            amp_models,
            t_str,
            args.workers,
        )
        if epics:
            pd.DataFrame(
                epics,
                columns=["timestamp", "lon", "lat", "lon_std", "lat_std"],
            ).to_csv(out_path, mode="a", index=False, header=header_needed)
            header_needed = False

        # release small loop‑scoped objects
        del epics
        gc.collect()


if __name__ == "__main__":
    main()
