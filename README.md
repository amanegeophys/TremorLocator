<p align="center">
  <img src="assets/logo.png" alt="TremorLocator Logo" width="400"/>
</p>

# TremorLocator

TremorLocator detects tectonic tremors from continuous Hi-net waveform data and
estimates their epicenters with pretrained deep learning models.

The pipeline has two stages:

- **Tremor detection**: three-component SAC waveforms are filtered, split into
  one-minute windows, converted to spectrograms, and classified by
  `TremorDetector.keras`.
- **Epicenter estimation**: station-wise tremor probabilities and RMS amplitudes
  are clustered and passed to an ensemble of epicenter regression models.

## Repository Layout

```text
TremorLocator/
├── assets/
│   └── logo.png
├── model/
│   ├── tremor_detector/
│   │   └── TremorDetector.keras
│   └── epicenter_regressors/
│       ├── 001.keras
│       ├── 002.keras
│       └── ...
├── reports/
│   ├── prediction_results.csv  # Final epicenter estimates
│   └── rms_proba/              # Per-hour station probabilities and RMS values
│       ├── 2018070100.csv
│       ├── 2018070101.csv
│       └── ...
├── station/
│   └── hinet129.txt            # Station metadata
├── src/
│   ├── predict_turbo.py        # Main inference pipeline
│   └── my_module/
│       ├── sac/                # SAC reading, trimming, filtering, and splitting
│       ├── spectrogram_generator.py
│       └── utils.py
├── pyproject.toml              # uv project metadata and dependencies
├── uv.lock                     # Locked dependency versions
└── README.md
```

## SAC Input Layout

The SAC loader reads hourly files named:

```text
{station}.{component}.SAC
```

where `{component}` is `E`, `N`, or `U`. When `--sac-root` is supplied, files
must be arranged like this:

```text
SAC_ROOT/
└── {year}/
    └── {YYYYMMDDHH}/
        ├── N.KWBH.E.SAC
        ├── N.KWBH.N.SAC
        └── N.KWBH.U.SAC
```

If `--sac-root` is omitted, the built-in year-to-path mapping in
`src/predict_turbo.py` is used.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for environment and
dependency management. Python 3.11 is recommended because it works well with the
pinned TensorFlow version.

```bash
git clone https://github.com/amanegeophys/TremorLocator.git
cd TremorLocator
uv sync
```

For notebook or plotting work, install the development dependency group:

```bash
uv sync --group dev
```

## Running Inference

Run the full detection and epicenter pipeline from the repository root:

```bash
uv run python src/predict_turbo.py \
  --start 2025-04-01-00:00:00.000000 \
  --end 2025-04-02-00:00:00.000000 \
  --sac-root /path/to/sac \
  --workers 16 \
  --out reports/prediction_results.csv \
  --std_threshold 0.065 \
  --station_range 0.5
```

Arguments:

- `--start`, `--end`: processing range in `YYYY-mm-dd-HH:MM:SS.ffffff` format.
- `--sac-root`: optional SAC root directory. Omit this to use the built-in
  year-specific network paths.
- `--workers`: number of parallel station workers. The default is `8`.
- `--out`: final epicenter CSV path. The default is
  `reports/prediction_results.csv`.
- `--std_threshold`: optional maximum latitude or longitude standard deviation
  allowed for an epicenter estimate.
- `--station_range`: geographic range, in degrees, used to check tremor stations
  around each estimated epicenter. The default is `0.5`.

The final epicenter catalog is written to the path passed with `--out`.
Intermediate per-station probability and amplitude files are written to
`reports/rms_proba/{YYYYMMDDHH}.csv`.

## Dependency Management

Runtime dependencies are declared in `pyproject.toml` and locked in `uv.lock`.

Use these commands when dependencies change:

```bash
uv add PACKAGE
uv remove PACKAGE
uv lock
uv sync
```

Avoid editing `uv.lock` by hand. Commit both `pyproject.toml` and `uv.lock`
after dependency changes.

## Output Files

The current `reports/` directory is organized as:

```text
reports/
├── prediction_results.csv
└── rms_proba/
    ├── 2018070100.csv
    ├── 2018070101.csv
    └── ...
```

`reports/prediction_results.csv` is the final tremor epicenter catalog. It
contains:

- `datetime`: one-minute tremor window time.
- `lat`, `lon`: estimated epicenter.
- `lat_std`, `lon_std`: ensemble uncertainty.
- `used_station`: semicolon-separated station IDs used after filtering.

Each `reports/rms_proba/{YYYYMMDDHH}.csv` file is the intermediate output for
one hour. It contains one row per station per minute:

- `datetime`: one-minute waveform window time.
- `NS`, `EW`, `UD`: RMS amplitudes for the three components.
- `noise`, `tremor`, `eq`: TremorDetector class probabilities.
- `station`: station code.

## Models

### TremorDetector

`model/tremor_detector/TremorDetector.keras` classifies normalized
three-component spectrograms into:

- `noise`
- `tremor`
- `eq`

### Epicenter Regressors

`model/epicenter_regressors/*.keras` is an ensemble of amplitude-to-location
models. The pipeline averages their predictions and uses the ensemble standard
deviation as an uncertainty estimate.

## Reference

Jinde, Y., Sugii, A., and Hiramatsu, Y. Enhanced deep learning approach for
detecting and locating tectonic tremors in the Nankai subduction zone. *Earth,
Planets and Space*, 77, 121 (2025). https://doi.org/10.1186/s40623-025-02257-y

```bibtex
@article{jinde2025enhanced,
  title={Enhanced deep learning approach for detecting and locating tectonic tremors in the Nankai subduction zone},
  author={Jinde, Yuya and Sugii, Amane and Hiramatsu, Yoshihiro},
  journal={Earth, Planets and Space},
  volume={77},
  pages={121},
  year={2025},
  doi={10.1186/s40623-025-02257-y},
}
```

## License

This project is licensed under the [MIT License](LICENSE).
