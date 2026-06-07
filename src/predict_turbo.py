"""
Tremor detection and epicenter estimation pipeline.

This module provides functions to process SAC waveform data from Hi-net stations,
compute RMS amplitude and tremor probabilities, and estimate epicenters using
pre-trained models and DBSCAN clustering.
"""

from __future__ import annotations

import argparse
import csv
import glob
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from obspy import UTCDateTime
from tqdm import tqdm

from my_module import SacHandler, SacTrace, SpectrogramGenerator
from my_module.utils import setup_logger

# tf.config.set_visible_devices([], device_type='GPU')
# ────────────────────────────────────────────────────────────
# CONSTANTS (hyper‑parameters that rarely change)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
TREMORDETECTOR_PATH: Path = PROJECT_ROOT / "model/tremor_detector/TremorDetector.keras"
STATION_FILE: Path = PROJECT_ROOT / "station/hinet129.txt"
AMP_TO_EPI_DIR: Path = PROJECT_ROOT / "model/epicenter_regressors"
RMS_PROBA_DIR: Path = PROJECT_ROOT / "reports/rms_proba"

DBSCAN_EPS: float = 0.5
DBSCAN_MIN_SAMPLES: int = 3
BUNGO_LOC: Tuple[float, float] = (132.1, 33.15)
ORIGIN_LOC: Tuple[float, float] = (
    34.1,
    135,
)  # Base point for epicenter offsets (lat, lon)

AMP_NULL_VALUE: float = 1e-9
TREMOR_THRESHOLD: float = 0.9
STATION_THRESHOLD: int = 3

logger = setup_logger(__name__)
SPEC_COMPONENTS = ["EW", "NS", "UD"]
AMP_COMPONENTS = ["NS", "EW", "UD"]
DEFAULT_COMPONENT_CHANNELS = {
    "EW": "E",
    "NS": "N",
    "UD": "U",
}
DEFAULT_YEAR_TO_PATH = {
    2007: "/net/pekora/mnt/sde/sac/2007",
    2008: "/net/remie/mnt/sde/sac/2008",
    2009: "/net/remie/mnt/sde/sac/2009",
    2010: "/net/synology/volume1/Main/sac/2010",
    2011: "/net/synology/volume1/Main/sac/2011",
    2012: "/net/synology/volume1/Main/sac/2012",
    2013: "/net/remie/mnt/sdd/sac/2013",
    2014: "/net/remie/mnt/sdd/sac/2014",
    2015: "/net/pekora/mnt/sda/sac/2015",
    2016: "/net/pekora/mnt/sda/sac/2016",
    2017: "/net/pekora/mnt/sde/sac/2017",
    2018: "/net/synology/volume1/Main/sac/2018",
    2019: "/net/synology/volume1/Main/sac/2019",
    2020: "/net/synology/volume1/Main/sac/2020",
    2021: "/net/synology/volume1/Main/sac/2021",
    2022: "/net/pekora/mnt/sdb/sac/2022",
    2023: "/net/mandel/mnt/sdb/sac/2023",
    2024: "/net/pekora/mnt/sdb/sac/2024",
    2025: "/net/mandel/mnt/sdb/sac/2025",
}


def build_year_to_path(sac_root: str | None) -> dict[int, str | Path]:
    if sac_root is None:
        return dict(DEFAULT_YEAR_TO_PATH)

    root = Path(sac_root)
    return {year: root / str(year) for year in DEFAULT_YEAR_TO_PATH}


# ╭──────────────────────────────────────────────────────────╮
# │  Utility functions                                      │
# ╰──────────────────────────────────────────────────────────╯
def load_stations(file_path: Path) -> pd.DataFrame:
    """
    Load station metadata from a whitespace-separated file.

    Parameters
    ----------
    file_path : Path
        Path to the station list file.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing station information with columns: ['station', 'lon', 'lat'].
    """
    return pd.read_table(file_path, sep=r"\s+")


def process_station_nopredict(
    station_code: str,
    sac_handler: SacHandler,
    spec_gen: SpectrogramGenerator,
    start_time: str,
) -> tuple[str, np.ndarray, list[int], list[np.ndarray], list[np.ndarray]]:
    """
    Process one station to extract spectrograms and RMS amplitudes without prediction.

    Parameters
    ----------
    station_code : str
        Station ID.
    sac_handler : SacHandler
        Handler for SAC data.
    spec_gen : SpectrogramGenerator
        Spectrogram generator.
    start_time : str
        Start time of the 1-hour window in the format 'YYYY-mm-dd-HH:MM:SS'.

    Returns
    -------
    tuple
        - station ID
        - array of 60 timestamps
        - list of valid minute indices
        - list of spectrograms (np.ndarray)
        - list of RMS amplitude vectors (np.ndarray)
    """
    # 1) 時刻準備
    start = UTCDateTime(start_time)
    times = np.array(
        [(start + 60 * m).strftime("%Y-%m-%d-%H:%M:%S.%f") for m in range(60)],
        dtype=object,
    )

    # 2) SACトレース取得とフィルタ
    sac_traces = sac_handler.get_sac_traces(station_code, start)
    if any(sac_traces.get(k) is None for k in SPEC_COMPONENTS):
        return station_code, times, [], [], []
    sac_traces = sac_handler.filter_sac_traces(
        sac_traces,
        freqmin=2,
        freqmax=8,
        filter_type="bandpass",
        corners=2,
        zerophase=True,
    )
    if any(sac_traces.get(k) is None for k in SPEC_COMPONENTS):
        return station_code, times, [], [], []

    # 3) 1分切り出し
    sac_traces_min, minute_times = sac_handler.split_sac_traces_by_minute(
        sac_traces,
        start,
        components=tuple(SPEC_COMPONENTS),
    )

    valid_idxs: list[int] = []
    specs: list[np.ndarray] = []
    rms_list: list[np.ndarray] = []

    for minute_time, seg in zip(minute_times, sac_traces_min):
        spec = spec_gen.generate_spectrograms(seg, normalize=True)
        if spec is None:
            continue
        minute_idx = int(round((UTCDateTime(minute_time) - start) / 60.0))
        valid_idxs.append(minute_idx)
        specs.append(spec.astype(np.float32))
        # RMS をベクトル化
        data = np.vstack([seg[c].data for c in AMP_COMPONENTS])
        rms = np.sqrt(np.mean(data * data, axis=1, dtype=np.float32))
        rms_list.append(rms)

    return station_code, times, valid_idxs, specs, rms_list


def estimate_once_fast(
    station_codes: list[str],
    sac_handler: SacHandler,
    spec_gen: SpectrogramGenerator,
    tremor_detector: object,
    start_time: str,
    max_workers: int,
    output_csv_path: Path,
) -> None:
    """
    High-speed processing of tremor detection using parallelized SAC reading and batch inference.

    Parameters
    ----------
    station_codes : list of str
        List of station IDs.
    sac_handler : SacHandler
        Handler for SAC data.
    spec_gen : SpectrogramGenerator
        Spectrogram generator.
    tremor_detector : keras.Model
        Trained tremor detection model.
    start_time : str
        Start time for the processing window ('YYYY-mm-dd-HH:MM:SS').
    max_workers : int
        Number of workers for parallel processing.
    output_csv_path : Path
        Output path for CSV file with probabilities and amplitudes.
    """
    logger.info(f"[{start_time}] Starting estimate_once_fast")
    t0 = time.perf_counter()

    # 1) ThreadPool: SAC読み込み＋spec/RMS収集
    records: list[dict] = []
    specs_all: list[np.ndarray] = []

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {
            exe.submit(
                process_station_nopredict, st, sac_handler, spec_gen, start_time
            ): st
            for st in station_codes
        }
        # 受付順不問なので as_completed
        with tqdm(
            total=len(futures), desc=f"[{start_time}] 📥 Load + RMS + Spec"
        ) as pbar:
            for fut in as_completed(futures):
                st, times, valid_idxs, specs, rms_list = fut.result()
                for idx, spec, rms in zip(valid_idxs, specs, rms_list):
                    specs_all.append(spec)
                    records.append(
                        {
                            "station": st,
                            "minute_idx": idx,
                            "rms": rms,
                            "prob": None,  # 後で埋める
                        }
                    )
                pbar.update(1)

    # 2) スペクトログラムがなければ空CSV出力して終了
    if not specs_all:
        pd.DataFrame(
            columns=["datetime", "NS", "EW", "UD", "noise", "tremor", "eq", "station"]
        ).to_csv(output_csv_path, index=False)
        logger.info(f"[{start_time}] No data found — skipping.")
        return

    # 3) バッチ予測（chunking with batch_size）
    specs_arr = np.stack(specs_all, axis=0)  # shape=(N,52,52,3)
    logger.info(f"[{start_time}] Starting tremor prediction...")
    preds = tremor_detector.predict(specs_arr, batch_size=specs_arr.shape[0], verbose=0)
    logger.info(f"[{start_time}] Prediction completed")

    # 4) records に確率を戻す
    for rec, prob in zip(records, preds):
        rec["prob"] = prob

    # 5) DataFrame 一括組立
    # 全ステーション×60分のテンプレートを準備
    template = pd.DataFrame(
        {
            "datetime": times,
            "NS": np.nan,
            "EW": np.nan,
            "UD": np.nan,
            "noise": np.nan,
            "tremor": np.nan,
            "eq": np.nan,
        }
    )

    dfs: list[pd.DataFrame] = []
    for st in station_codes:
        df_st = template.copy()
        df_st["station"] = st
        # このステーション分だけフィルタ
        for rec in filter(lambda r: r["station"] == st, records):
            i = rec["minute_idx"]
            df_st.loc[i, ["NS", "EW", "UD"]] = rec["rms"]
            df_st.loc[i, ["noise", "tremor", "eq"]] = rec["prob"]
        dfs.append(df_st)

    df_all = pd.concat(dfs, ignore_index=True)

    # 6) 一括 CSV 出力（フォーマットは float_format で）
    df_all.to_csv(output_csv_path, index=False, float_format="%.3e")
    logger.info(
        f"[{start_time}] estimate_once_fast finished in {time.perf_counter() - t0:.2f}s"
    )


def process_rms_inputdata(
    station_df: pd.DataFrame,
    time: str,
    gp: pd.DataFrame,
    station_codes: Sequence[str],
) -> tuple[list[str], list[np.ndarray], list[pd.DataFrame]]:
    """
    Preprocess per-station amplitude and probability data for epicenter estimation.

    Parameters
    ----------
    station_df : pd.DataFrame
        Metadata for stations.
    time : str
        Current datetime string.
    gp : pd.DataFrame
        Per-station results (RMS and probabilities).
    station_codes : sequence of str
        List of station IDs.

    Returns
    -------
    tuple
        - list of datetime strings
        - list of amplitude arrays (3 x station)
        - list of processed dataframes
    """
    from sklearn.cluster import DBSCAN

    df = gp.set_index("station").loc[station_codes].reset_index()
    df = pd.merge(df, station_df, on="station", how="left")
    mask_all_nan = df[["tremor", "noise", "eq"]].isna().all(axis=1)
    df.loc[mask_all_nan, ["tremor", "noise", "eq"]] = -1
    tremor_df = df[df["tremor"] >= TREMOR_THRESHOLD].copy()
    if tremor_df.empty:
        return [], [], []

    coords = np.vstack([tremor_df[["lon", "lat"]].values, BUNGO_LOC])
    labels = (
        DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(coords).labels_[:-1]
    )

    tremor_df["cluster"] = labels
    df = df.merge(tremor_df[["station", "cluster"]], on="station", how="left").fillna(
        {"cluster": -1}
    )
    df["cluster"] = df["cluster"].astype(int)

    time_records = []
    input_records = []
    d_records = []
    for lbl in sorted(df["cluster"].unique()):
        if lbl < 0 or (df["cluster"] == lbl).sum() < DBSCAN_MIN_SAMPLES:
            continue

        d = df.copy()
        d.loc[d["cluster"] != lbl, ["noise", "tremor", "eq"]] = (1.0, 0.0, 0.0)

        mean = d[AMP_COMPONENTS].mean()
        std = d[AMP_COMPONENTS].std()
        cutoff = mean + 3 * std
        for ch in AMP_COMPONENTS:
            d.loc[d[ch] >= cutoff[ch], ch] = AMP_NULL_VALUE

        d.loc[d["tremor"] < TREMOR_THRESHOLD, AMP_COMPONENTS] = AMP_NULL_VALUE

        for ch in AMP_COMPONENTS:
            amp_mean = np.nanmean(d[ch])
            amp_std = np.nanstd(d[ch])
            d[ch] = 0 if amp_std == 0 else (d[ch] - amp_mean) / amp_std

        amp_in = np.nan_to_num(d[AMP_COMPONENTS].to_numpy().T)
        time_records.append(time)
        input_records.append(amp_in)
        d_records.append(d)

    return time_records, input_records, d_records


def estimate_epicenter(
    writer: csv.writer,
    station_df: pd.DataFrame,
    station_codes: Sequence[str],
    max_workers: int,
    input_csv_path: Path,
    amp2epicenter: list[object],
    std_threshold: float | None = None,
    station_range: float = 0.5,
) -> None:
    """
    Estimate epicenters based on tremor predictions and amplitude data.

    Parameters
    ----------
    writer : csv.writer
        Writer to output epicenter CSV.
    station_df : pd.DataFrame
        Metadata for stations.
    station_codes : sequence of str
        List of station IDs.
    max_workers : int
        Number of threads for parallel processing.
    input_csv_path : Path
        Input CSV with station-wise prediction data.
    amp2epicenter : list of keras.Model
        List of trained epicenter regression models.
    std_threshold : float or None, optional
        Standard deviation threshold to filter uncertain estimates.
    station_range : float, optional
        Geographic range threshold for local tremor clustering.
    """
    logger.info("Starting estimate_epicenter...")
    df = pd.read_csv(input_csv_path)
    groups = df.groupby("datetime")

    time_list = []
    input_list = []
    d_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        future_to_time = [
            exe.submit(
                process_rms_inputdata,
                station_df,
                time,
                gp,
                station_codes,
            )
            for time, gp in groups
        ]

        with tqdm(total=len(future_to_time), desc="🔍 Epicenter clustering") as pbar:
            for fut in as_completed(future_to_time):
                times, inputs, d_records = fut.result()
                if len(times) != 0:
                    time_list.extend(times)
                    input_list.extend(inputs)
                    d_list.extend(d_records)
                pbar.update(1)

    if len(time_list) != 0:
        amp_in = np.stack(input_list, axis=0)  # (N, 3, S)
        logger.info("Prediction with amp2epicenter models...")
        preds = np.stack([m.predict(amp_in, verbose=0) for m in amp2epicenter], axis=0)
        logger.info("Epicenter estimation completed.")

        pred_mean = preds.mean(axis=0)  # (N, 2)
        pred_std = preds.std(axis=0)  # (N, 2)
        lats = (
            pred_mean[:, 0] + ORIGIN_LOC[0]
        )  # 注意: ORIGIN_LOC=(lat,lon)…順序混乱してないか確認!
        lons = pred_mean[:, 1] + ORIGIN_LOC[1]
        lat_stds = pred_std[:, 0]
        lon_stds = pred_std[:, 1]

        for time, lat, lon, lat_std, lon_std, d in zip(
            time_list, lats, lons, lat_stds, lon_stds, d_list
        ):
            if std_threshold is not None and (
                lat_std > std_threshold or lon_std > std_threshold
            ):
                continue

            bound_lon_min = lon - station_range
            bound_lon_max = lon + station_range
            bound_lat_min = lat - station_range
            bound_lat_max = lat + station_range
            around = d[
                (d["lon"] >= bound_lon_min)
                & (d["lon"] <= bound_lon_max)
                & (d["lat"] >= bound_lat_min)
                & (d["lat"] <= bound_lat_max)
                & (d["tremor"] >= TREMOR_THRESHOLD)
            ]

            if len(around) < STATION_THRESHOLD:
                continue

            stations = ";".join(around["station"].tolist())

            writer.writerow(
                [
                    time,
                    round(float(lat), 5),
                    round(float(lon), 5),
                    round(float(lat_std), 5),
                    round(float(lon_std), 5),
                    stations,
                ]
            )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """
    Parse command-line arguments for pipeline.

    Parameters
    ----------
    argv : sequence of str
        List of command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes: start, end, std_threshold,
        station_range, workers, out.
    """
    p = argparse.ArgumentParser(description="Tremor Locator runner")
    p.add_argument(
        "--start", required=True, help="Start time (YYYY-mm-dd-HH:MM:SS.ffffff)"
    )
    p.add_argument(
        "--end", required=True, help="End time   (YYYY-mm-dd-HH:MM:SS.ffffff)"
    )
    p.add_argument(
        "--std_threshold",
        type=float,
        default=None,
        help="Std dev threshold for epicenter filtering",
    )
    p.add_argument(
        "--station_range",
        type=float,
        default=0.5,
        help="Station range for epicenter refinement",
    )
    p.add_argument("--workers", type=int, default=8, help="Thread pool size")
    p.add_argument(
        "--sac-root",
        default=None,
        help=(
            "Optional SAC root directory with files arranged as "
            "SAC_ROOT/{year}/{YYYYMMDDHH}/{station}.{component}.SAC. "
            "If omitted, the built-in year-to-path mapping is used."
        ),
    )
    p.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "reports/prediction_results.csv"),
        help="Output CSV path",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str]) -> None:
    """
    Entry point for the tremor detection and epicenter estimation pipeline.

    This function orchestrates the processing of waveform data, tremor prediction,
    and epicenter localization over a specified time range.

    Parameters
    ----------
    argv : sequence of str
        Command-line arguments.
    """
    args = parse_args(argv)
    from keras.models import load_model

    station_df = load_stations(STATION_FILE)
    station_codes = station_df["station"].tolist()
    sac_handler = SacHandler(
        year_to_path=build_year_to_path(args.sac_root),
        component_channels=DEFAULT_COMPONENT_CHANNELS,
        duration_seconds=3599.99,
    )
    spec_gen = SpectrogramGenerator(
        normalize_type="mean_std",
        stft_padding="zeros",
        output_layout="channels_last",
    )
    tremor_detector = load_model(str(TREMORDETECTOR_PATH))
    amp2epicenter = [load_model(p) for p in glob.iglob(str(AMP_TO_EPI_DIR / "*.keras"))]
    logger.info("Loaded %d epicenter models", len(amp2epicenter))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    time_range = pd.date_range(args.start, args.end, freq="1h")

    with open(out_path, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(
            ["datetime", "lat", "lon", "lat_std", "lon_std", "used_station"]
        )
        for t in tqdm(time_range, desc="⏳ Processing each hour:"):
            t_str = t.strftime("%Y-%m-%d-%H:%M:%S")
            RMS_PROBA_DIR.mkdir(parents=True, exist_ok=True)
            rms_csv = RMS_PROBA_DIR / t.strftime("%Y%m%d%H.csv")
            estimate_once_fast(
                station_codes,
                sac_handler,
                spec_gen,
                tremor_detector,
                t_str,
                args.workers,
                rms_csv,
            )
            estimate_epicenter(
                writer,
                station_df,
                station_codes,
                args.workers,
                rms_csv,
                amp2epicenter,
                args.std_threshold,
                args.station_range,
            )


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
