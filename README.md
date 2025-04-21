# TremorEpicenter

Version: 0.1.0

**TremorEpicenter** is a pipeline for detecting seismic tremors and estimating epicenters using deep learning models. It integrates a spectrogram-to-probability classification model and a set of amplitude-to-epicenter regression models to generate predictions based on the amplitudes of continuous SAC waveform data.

## 🚀 Features

- **Tremor Detection**: Classify waveform segments as noise, tremor, or earthquake using a pretrained TremorNet model.
- **Epicenter Estimation**: Cluster detected tremor signals and regress epicenter coordinates using multiple pretrained regression models.
- **Parallel Processing**: Leverage multithreading to process station data concurrently for high-throughput inference.
- **Configurable Time Windows**: Run predictions over arbitrary time ranges with customizable step sizes.

## 📁 Directory Structure

```
TremorEpicenter/
├── model/
│   ├── spec_to_proba/
│   │   └── tremornet.keras        # Pretrained classification model
│   └── amp_to_epicenter/
│       ├── 001.keras
│       ├── 002.keras
│       └── ...
├── reports/ 
│    └── prediction_results.csv
├── sac/
│   └── {year}/
│       └── {YYYYMMDDHH}/
│           ├── N.xxxx.N.SAC
│           ├── N.xxxx.E.SAC
│           └── N.xxxx.U.SAC      # Raw SAC files (hourly folders)
├── station/
│   └── hinet129.txt              # Station list (lat, lon, station)
├── src/
│   └── my_module/
│       ├── sac/
│       │   ├── sac_handler.py    # SAC file I/O and trace loading
│       │   └── sac_trace.py      # SAC trace data class
│       ├── spectrogram_generator.py  # Generate spectrograms
│       ├── utils.py              # Logging and utilities
│       └── predict.py            # Main pipeline script
├── requirements.txt              # Python dependencies
└── README.md                     # Project overview and usage
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/amanegeophys/TremorEpicenter.git
   cd TremorEpicenter
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data**
   - Place SAC waveform files under `sac/{year}/{YYYYMMDDHH}/` or modify the `SAC_ROOT` variable (line 23) in `predict.py` to specify your own path 
   - Ensure `station/hinet129.txt` is formatted with whitespace separator and headers `lon lat station`.

## 🖥️ Usage

Run the pipeline over a specified time range with a given step size (in minutes) and number of worker threads:

```bash
python predict.py \
  --start 2025-04-01-00:00:00.000000 \
  --end   2025-04-02-00:00:00.000000 \
  --step  10 \
  --workers 16 \
  --out prediction_results.csv
```

- `--start` and `--end`: Define start and end timestamps in `YYYY-mm-dd-HH:MM:SS.ffffff` format.
- `--step`: Time-step increment in minutes (default: 1).
- `--workers`: Number of threads for parallel station processing (default: 8).
- `--out`: (Optional) Output CSV file path (default: `reports/prediction_results.csv`).

## 📄 Paper

Include the publication URL here:

```
Paper URL: coming soon ?
```

## 📄 Requirements

- Python 3.9+
- numpy
- pandas
- tensorflow>=2.17
- scikit-learn
- scipy
- tqdm
- obspy

## 📑 Contributing

Contributions, issues, and feature requests are welcome:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/my-feature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/my-feature`).
5. Open a Pull Request.

## 📝 License

This project is licensed under the [MIT License](LICENSE).