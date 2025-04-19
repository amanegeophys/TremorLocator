from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import signal

from .sac.sac_trace import SacTrace
from .utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class SpectrogramGenerator:
    """
    Class for generating spectrograms from seismic waveform data.

    Parameters
    ----------
    sampling_rate : float, default=100
        Sampling rate of the waveform in Hz.
    fft_window_sec : float, default=5.0
        Length of the FFT window in seconds.
    overlap_rate : float, default=0.762
        Fraction of overlap between consecutive FFT windows.
    freqmin : int, default=0
        Minimum frequency to retain in the spectrogram.
    freqmax : int, default=10
        Maximum frequency to retain in the spectrogram.
    normalize_type : str, default="mean_std"
        Method used to normalize the spectrogram. Options: "min_max", "mean_std".
    """

    sampling_rate: float = 100
    fft_window_sec: float = 5.0
    overlap_rate: float = 0.762
    freqmin: int = 0
    freqmax: int = 10
    normalize_type: str = "mean_std"

    def __post_init__(self):
        self.nperseg, self.noverlap, self.nfft = (
            self._calculate_fft_parameters(
                self.sampling_rate, self.fft_window_sec, self.overlap_rate
            )
        )
        self.window = signal.windows.tukey(self.nperseg, alpha=0.1)

    def _calculate_fft_parameters(
        self, sampling_rate: float, fft_window_sec: float, overlap_rate: float
    ) -> Tuple[int, int, int]:
        """
        Calculate FFT parameters.

        Parameters
        ----------
        sampling_rate : float
            Sampling rate of the waveform.
        fft_window_sec : float
            Length of FFT window in seconds.
        overlap_rate : float
            Fraction of overlap.

        Returns
        -------
        nperseg : int
            Number of samples per FFT segment.
        noverlap : int
            Number of samples to overlap.
        nfft : int
            Number of FFT points.
        """
        nperseg = int(sampling_rate * fft_window_sec)
        noverlap = int(nperseg * overlap_rate)
        nfft = int(2 ** (len(bin(nperseg)) - 2))
        return nperseg, noverlap, nfft

    def generate_spectrograms(
        self,
        sac_traces: Dict[str, Optional[SacTrace]],
        normalize: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Generate spectrograms for 3-component traces.

        Parameters
        ----------
        sac_traces : dict of str to SacTrace
            Dictionary mapping component ('E', 'N', 'U') to SacTrace.
        normalize : bool, default=True
            Whether to normalize the spectrogram.

        Returns
        -------
        np.ndarray or None
            3D array of spectrograms (freq, time, component) or None if failed.
        """
        spectrograms = []
        for key, sac_trace in sac_traces.items():
            if sac_trace is None:
                logger.warning(f"No SacTrace found for key: {key}.")
                return None

            spectrogram = self.generate_spectrogram(sac_trace, normalize)

            if spectrogram is None:
                return None

            spectrograms.append(spectrogram)

        return np.stack(spectrograms, axis=2)

    def generate_spectrogram(
        self,
        sac_trace: SacTrace,
        normalize: bool,
    ) -> Optional[np.ndarray]:
        """
        Generate a spectrogram from a single SacTrace.

        Parameters
        ----------
        sac_trace : SacTrace
            The waveform trace to transform.
        normalize : bool
            Whether to normalize the spectrogram.

        Returns
        -------
        np.ndarray or None
            2D array of the spectrogram (freq x time) or None on error.
        """
        waveform = sac_trace.data

        freq, _, zxx = signal.stft(
            waveform,
            fs=self.sampling_rate,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            nfft=self.nfft,
            window=self.window,
        )
        sxx = np.abs(zxx)

        if self.normalize_type == "min_max":
            sxx2 = sxx**2 / (self.sampling_rate / self.nfft)

            if np.any(sxx2 == 0):
                return None

            sxx2 = np.log10(sxx2)
        else:
            sxx2 = sxx

        sxx2 = sxx2[(self.freqmin <= freq) & (freq <= self.freqmax), :]

        if normalize:
            if (sxx2 := self._normalize_spectrogram(sxx2)) is None:
                return None

        return sxx2

    def _normalize_spectrogram(self, sxx: np.ndarray) -> Optional[np.ndarray]:
        """
        Normalize spectrogram values.

        Parameters
        ----------
        sxx : np.ndarray
            Spectrogram data.

        Returns
        -------
        np.ndarray or None
            Normalized spectrogram, or None if normalization fails.
        """
        if self.normalize_type == "min_max":
            sxx_max = np.max(sxx)
            sxx_min = np.min(sxx)

            if sxx_max == sxx_min:
                logger.warning(
                    "Standard deviation is zero during normalization. Can't make spectrgram."
                )
                return None
            return (sxx - sxx_min) / (sxx_max - sxx_min)

        if self.normalize_type == "mean_std":
            sxx_mean = np.mean(sxx)
            sxx_std = np.std(sxx)

            if sxx_std == 0:
                logger.warning(
                    "Standard deviation is zero during normalization. Can't make spectrgram."
                )
                return None

            return (sxx - sxx_mean) / sxx_std
