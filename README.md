# TremorLocator

**Version**: 2.0.2

**TremorLocator** is a modular deep learning system for detecting tectonic tremors and estimating their epicenters from continuous waveform data. It is designed for both real-time and offline seismic monitoring workflows.

- **TremorDetector** is a convolutional neural network (CNN) that classifies spectrograms generated from seismic waveform segments, labeling them as noise, tremor, or earthquake.
- **EpicenterRegressor** is a set of regression models that estimate the geographic coordinates (epicenter) of detected tremors based on amplitude features from multiple seismic stations.

The system supports SAC (Seismic Analysis Code) format and is optimized for efficient processing of large-scale seismic datasets.

---

## 🚀 Features

- **Event Classification**: Classify waveform segments using spectrogram inputs with a pretrained CNN-based model (**TremorDetector**).
- **Epicenter Estimation**: Estimate epicenters of detected tremors using pretrained regression models (**EpicenterRegressor**).
- **Multithreaded Processing**: Handle high-throughput data with concurrent processing across multiple seismic stations.
- **Flexible Time Control**: Specify custom time ranges and processing intervals for tailored analysis.

---

## 📁 Directory Structure

```
TremorLocator/
├── model/
│   ├── tremor_detector/                # Spectrogram-to-class model (TremorDetector)
│   │   └── TremorDetector.keras
│   └── epicenter_regressors/          # Amplitude-to-location models (EpicenterRegressor)
│       ├── 001.keras
│       ├── 002.keras
│       └── ...
├── reports/ 
│   └── prediction_results.csv          # Output predictions
├── sac/
│   └── {year}/{YYYYMMDDHH}/            # Raw waveform files (.SAC format)
│       ├── N.xxxx.N.SAC
│       ├── N.xxxx.E.SAC
│       ├── N.xxxx.U.SAC
│       └── ...
├── station/
│   └── hinet129.txt                    # Station list (lon, lat, station)
├── src/
│   └── my_module/
│       ├── sac/
│       │   ├── sac_handler.py          # SAC I/O utilities
│       │   └── sac_trace.py            # Data container for traces
│       ├── spectrogram_generator.py    # STFT and normalization
│       ├── utils.py                    # Logging and support tools
│       └── predict.py                  # Main entry point
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation (this file)
```

---

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/amanegeophys/TremorLocator.git
   cd TremorLocator
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare input data**
   - Place SAC files under `sac/{year}/{YYYYMMDDHH}/`
   - Modify `SAC_ROOT` in `predict.py` (line 23) if needed
   - Ensure the station list file `station/hinet129.txt` has whitespace-separated columns: `lon lat station`

---

## 💻 Usage

Run inference over a specified time window using the command below:

```bash
python predict.py \
  --start 2025-04-01-00:00:00.000000 \
  --end   2025-04-02-00:00:00.000000 \
  --step  10 \
  --workers 16 \
  --out prediction_results.csv \
  --std_threshold 0.065
```

### Arguments

- `--start`, `--end`: Time range for processing (`YYYY-mm-dd-HH:MM:SS.ffffff`)
- `--step`: Step size in minutes between each analysis window (default: 1)
- `--workers`: Number of threads for parallel processing (default: 8)
- `--out`: (Optional) Output CSV file path (default: `reports/prediction_results.csv`)
- `--std_threshold`: (Optional) Standard deviation threshold for filtering low-confidence epicenter estimates.
If specified, estimates with either latitude or longitude standard deviation greater than this threshold will be excluded. No filtering is applied by default.

---

## 🧠 Models

### 🔹 TremorDetector (`tremor_detector/TremorDetector.keras`)
A CNN-based classification model that inputs spectrograms from 3-component waveform segments and outputs the probability of:

- **Noise**
- **Tremor**
- **Earthquake**

Spectrograms are generated using short-time Fourier transform (STFT) with configurable window length and overlap. These are normalized and fed into the model for classification.

### 🔸 EpicenterRegressor (`epicenter_regressors/*.keras`)
A set of regression models that input amplitude-normalized waveforms from multiple stations and output:

- **Latitude**
- **Longitude**  
of the estimated tremor source.

---

## 📄 Requirements

- Python 3.9+
- numpy==1.26.4
- pandas==2.2.2
- tensorflow==2.17.0
- scikit-learn==1.5.1
- tqdm==4.66.5
- scipy==1.13.1
- obspy==1.4.1

---

## 📚 Paper

> *Coming soon...?*

---

## 📁 Contributing

Feel free to contribute improvements or bug fixes!

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/my-feature`)
3. Commit changes (`git commit -m "Add feature"`)
4. Push (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).