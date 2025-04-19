from dataclasses import dataclass
from typing import Optional, Type, TypeVar

import numpy as np
import obspy
from obspy import Stream, Trace, UTCDateTime
from obspy.signal.filter import bandpass, bandstop

from ..utils import setup_logger

T = TypeVar("T", bound="SacTrace")
logger = setup_logger(__name__)


@dataclass(frozen=True)
class SacStats:
    """Metadata for a SAC waveform trace."""

    station_code: str
    channel_code: str
    sampling_rate: float
    start_time: UTCDateTime
    end_time: UTCDateTime
    npts: int

    def __str__(self) -> str:
        """Return a human-readable summary of the trace statistics."""
        return (
            f"Station: {self.station_code}\n"
            f"Channel: {self.channel_code}\n"
            f"Start Time: {self.start_time}\n"
            f"End Time: {self.end_time}\n"
            f"Sampling Rate: {self.sampling_rate} Hz\n"
            f"Number of Points: {self.npts}"
        )


@dataclass(frozen=True)
class SacTrace:
    """
    Container class for SAC waveform data and associated metadata.

    Attributes:
        data (np.ndarray): The waveform time-series data.
        stats (SacStats): Metadata associated with the waveform.
    """

    data: np.ndarray
    stats: SacStats

    @classmethod
    def from_file(cls: Type[T], sac_filepath: str) -> Optional[T]:
        """
        Load a SAC file and create a SacTrace object.

        Args:
            sac_filepath (str): Path to the SAC file.

        Returns:
            Optional[T]: A SacTrace instance if the file is read successfully; otherwise None.
        """
        try:
            sac_data_stream: Stream = obspy.read(sac_filepath)
        except (FileNotFoundError, IOError) as e:
            logger.warning(f"{sac_filepath} could not be read: {e}")
            return None

        trace: Trace = sac_data_stream[0]
        trace.data *= 1e-9  # Assume Hi-net data scaling

        stats: SacStats = SacStats(
            station_code=trace.stats.station,
            channel_code=trace.stats.channel,
            sampling_rate=trace.stats.sampling_rate,
            npts=trace.stats.npts,
            start_time=trace.stats.starttime,
            end_time=trace.stats.endtime,
        )

        return cls(trace.data, stats)

    def _can_concatenate(self, other: T) -> bool:
        """
        Check if another SacTrace is contiguous and compatible for concatenation.

        Args:
            other (SacTrace): Another SacTrace instance.

        Returns:
            bool: True if the traces can be concatenated, else False.
        """
        return (
            self.stats.station_code == other.stats.station_code
            and self.stats.channel_code == other.stats.channel_code
            and self.stats.sampling_rate == other.stats.sampling_rate
            and self.stats.end_time + (1 / self.stats.sampling_rate)
            == other.stats.start_time
        )

    def __add__(self, other_trace: T) -> T:
        """
        Concatenate this SacTrace with another one.

        Args:
            other_trace (SacTrace): Another SacTrace to concatenate.

        Returns:
            SacTrace: A new SacTrace with combined data and updated stats.

        Raises:
            ValueError: If the traces cannot be concatenated.
        """
        if not self._can_concatenate(other_trace):
            raise ValueError("Cannot concatenate the given SacTraces.")

        new_data = np.concatenate((self.data, other_trace.data))
        new_stats = SacStats(
            station_code=self.stats.station_code,
            channel_code=self.stats.channel_code,
            sampling_rate=self.stats.sampling_rate,
            npts=new_data.size,
            start_time=self.stats.start_time,
            end_time=other_trace.stats.end_time,
        )
        return SacTrace(new_data, new_stats)

    def __str__(self) -> str:
        """Return a compact summary of the SacTrace."""
        return (
            f"{self.stats.station_code}.{self.stats.channel_code} | "
            f"{self.stats.start_time} - {self.stats.end_time} | "
            f"{self.stats.sampling_rate:.1f} Hz | "
            f"{self.stats.npts} samples"
        )

    def filter(
        self,
        freq_min: int,
        freq_max: int,
        corners: int = 2,
        zerophase: bool = True,
        filter_type: str = "bandpass",
    ) -> T:
        """
        Apply a bandpass or bandstop filter to the waveform.

        Args:
            freq_min (int): Minimum frequency.
            freq_max (int): Maximum frequency.
            corners (int): Filter corners (order).
            zerophase (bool): Whether to apply a zero-phase filter.
            filter_type (str): Type of filter ('bandpass' or 'bandstop').

        Returns:
            SacTrace: Filtered SacTrace.

        Raises:
            ValueError: If an unsupported filter_type is provided.
        """
        if filter_type == "bandpass":
            filtered_data = bandpass(
                self.data,
                freq_min,
                freq_max,
                df=self.stats.sampling_rate,
                corners=corners,
                zerophase=zerophase,
            )
        elif filter_type == "bandstop":
            filtered_data = bandstop(
                self.data,
                freq_min,
                freq_max,
                df=self.stats.sampling_rate,
                corners=corners,
                zerophase=zerophase,
            )
        else:
            raise ValueError(f"Invalid filter_type: {filter_type}")
        return SacTrace(filtered_data, self.stats)

    def trim(
        self,
        start_time: UTCDateTime,
        end_time: UTCDateTime,
    ) -> Optional[T]:
        """
        Trim the waveform data to a specific time window.

        Args:
            start_time (UTCDateTime): Start time of the trim window.
            end_time (UTCDateTime): End time of the trim window.

        Returns:
            Optional[SacTrace]: A trimmed SacTrace or None if the window is invalid.
        """
        if not (
            self.stats.start_time
            <= start_time
            < end_time
            <= self.stats.end_time
        ):
            logger.warning(
                f"Provided start and end times are not within the SacTrace's time series: \n"
                f"{self.stats}\n"
                f"request date: {start_time} - {end_time}\n"
                f"now date: {self.stats.start_time} - {self.stats.end_time}"
            )
            return None

        start_idx = round(
            (start_time - self.stats.start_time) * self.stats.sampling_rate
        )
        end_idx = (
            round(
                (end_time - self.stats.start_time) * self.stats.sampling_rate
            )
            + 1
        )

        trimmed_data = self.data[start_idx:end_idx]
        new_stats = SacStats(
            station_code=self.stats.station_code,
            channel_code=self.stats.channel_code,
            sampling_rate=self.stats.sampling_rate,
            npts=trimmed_data.size,
            start_time=start_time,
            end_time=end_time,
        )
        return SacTrace(trimmed_data, new_stats)

    def remove_mean(self) -> T:
        """
        Remove the mean value from the waveform.

        Returns:
            SacTrace: A new SacTrace with demeaned data.
        """
        demeaned_data = self.data - np.mean(self.data)
        return SacTrace(demeaned_data, self.stats)

    def synthetic_waveform(self, waveform: np.ndarray) -> T:
        """
        Add a synthetic waveform to the trace.

        Args:
            waveform (np.ndarray): The synthetic signal to be added.

        Returns:
            SacTrace: A new SacTrace with the synthetic waveform added.

        Raises:
            ValueError: If the waveform length does not match.
        """
        if self.data.size != waveform.size:
            raise ValueError(
                "The length of the waveform does not match the trace data length."
            )

        synthesized_data = self.data + waveform
        return SacTrace(synthesized_data, self.stats)
