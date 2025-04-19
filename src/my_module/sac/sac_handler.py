import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

from obspy import UTCDateTime

from ..utils import setup_logger
from .sac_trace import SacTrace

logger = setup_logger(__name__)


@dataclass(frozen=True)
class SacHandler:
    """
    A high-level interface to manage and process SAC waveform files.

    This class provides functionality to read, concatenate, filter, and
    trim SAC files located in a structured directory. It is designed to simplify
    waveform extraction across time windows and channels for seismic analysis.

    Attributes:
        sac_directory (str): Root directory containing SAC files in a structured format.
        freqmin (int): Default minimum frequency for filtering.
        freqmax (int): Default maximum frequency for filtering.
        duration_seconds (float): Default waveform extraction duration.
        filter_type (str): Default filter type ("bandpass" or "bandstop").
    """

    sac_directory: str
    freqmin: int = 2
    freqmax: int = 8
    duration_seconds: float = 59.99
    filter_type: str = "bandpass"

    def _read_sac_trace(self, sac_filepath: str) -> Optional[SacTrace]:
        """
        Read a single SAC file.

        Args:
            sac_filepath (str): Path to the SAC file.

        Returns:
            Optional[SacTrace]: Parsed SacTrace object or None if reading fails.
        """
        if not os.path.exists(sac_filepath):
            logger.warning(f"File not found: {sac_filepath}")
            return None

        return SacTrace.from_file(sac_filepath)

    def _calculate_time_range(
        self, start_time: str, duration_seconds: float
    ) -> Tuple[UTCDateTime, UTCDateTime]:
        """
        Compute the UTCDateTime range from a string and duration.

        Args:
            start_time (str): Start time in '%Y-%m-%d-%H:%M:%S.%f' format.
            duration_seconds (float): Duration of the time window.

        Returns:
            Tuple[UTCDateTime, UTCDateTime]: Start and end times.
        """
        start_time = datetime.strptime(start_time, "%Y-%m-%d-%H:%M:%S.%f")
        start_time = UTCDateTime(start_time)
        end_time = start_time + duration_seconds
        return start_time, end_time

    def get_sac_trace(
        self,
        station_code: str,
        start_time: Union[str, UTCDateTime],
        channel_code: str,
        freqmin: Optional[int] = None,
        freqmax: Optional[int] = None,
        duration_seconds: Optional[float] = None,
        filter_type: Optional[str] = None,
    ) -> Optional[SacTrace]:
        """
        Retrieve a filtered and trimmed SAC trace for one component.

        Args:
            station_code (str): Station identifier.
            start_time (str | UTCDateTime): Start time.
            channel_code (str): Component ('EW', 'NS', 'UD').
            freqmin (int, optional): Minimum filter frequency.
            freqmax (int, optional): Maximum filter frequency.
            duration_seconds (float, optional): Duration in seconds.
            filter_type (str, optional): Type of filter.

        Returns:
            Optional[SacTrace]: Processed waveform or None if failure.
        """
        if isinstance(start_time, str):
            start_time, end_time = self._calculate_time_range(
                start_time, duration_seconds or self.duration_seconds
            )
        else:
            end_time = start_time + (duration_seconds or self.duration_seconds)

        freqmin = freqmin or self.freqmin
        freqmax = freqmax or self.freqmax
        filter_type = filter_type or self.filter_type

        trace = self._fetch_filtered_sac_trace(
            start_time.strftime("%Y%m%d%H"),
            end_time.strftime("%Y%m%d%H"),
            station_code,
            channel_code,
            freqmin,
            freqmax,
            filter_type,
        )

        if trace is not None:
            trace = trace.trim(start_time, end_time)

        return trace

    def get_sac_traces(
        self,
        station_code: str,
        start_time: str,
        freqmin: int = None,
        freqmax: int = None,
        duration_seconds: float = None,
        filter_type: str = "bandpass",
    ) -> Dict[str, Optional[SacTrace]]:
        """
        Retrieve filtered 3-component SAC traces (EW, NS, UD) for a given station and time window.

        Args:
            station_code (str): Station identifier.
            start_time (str): Start time string in '%Y-%m-%d-%H:%M:%S.%f' format.
            freqmin (int): Minimum frequency for filtering.
            freqmax (int): Maximum frequency for filtering.
            duration_seconds (float): Time window duration in seconds.
            filter_type (str): Filter type ("bandpass" or "bandstop").

        Returns:
            Dict[str, Optional[SacTrace]]: Dictionary of channel codes to SacTrace objects.
        """
        return {
            ch: self.get_sac_trace(
                station_code,
                start_time,
                ch,
                freqmin=freqmin,
                freqmax=freqmax,
                duration_seconds=duration_seconds,
                filter_type=filter_type,
            )
            for ch in ["EW", "NS", "UD"]
        }

    def _fetch_filtered_sac_trace(
        self,
        start_hour: str,
        end_hour: str,
        station_code: str,
        channel_code: str,
        freqmin: int,
        freqmax: int,
        filter_type: str,
    ) -> Optional[SacTrace]:
        """
        Load, concatenate, and filter SAC traces from filepaths based on time and metadata.

        Args:
            start_time (str): Start hour string (e.g., '2024010112').
            end_time (str): End hour string (e.g., '2024010113').
            station_code (str): Station identifier.
            channel_code (str): Channel code.
            freqmin (int): Minimum frequency for filtering.
            freqmax (int): Maximum frequency for filtering.
            filter_type (str): Type of filter.

        Returns:
            Optional[SacTrace]: Filtered concatenated trace or None.
        """
        paths = self._generate_sac_filepaths(
            start_hour, end_hour, station_code, channel_code
        )
        trace = self._read_and_concatenate_sac_traces(paths)

        if trace is None:
            return None
        if freqmax == -1:
            return trace

        return trace.filter(freqmin, freqmax, filter_type=filter_type)

    def _generate_sac_filepaths(
        self, start: str, end: str, station: str, ch: str
    ) -> List[str]:
        """
        Generate SAC file paths between two times assuming hourly subdirectories.

        Args:
            start_time (str): Start hour string (format: YYYYMMDDHH).
            end_time (str): End hour string (format: YYYYMMDDHH).
            station_code (str): Station identifier.
            channel_code (str): Channel code.

        Returns:
            List[str]: List of full SAC file paths.
        """
        paths = []
        current = datetime.strptime(start, "%Y%m%d%H")
        end_dt = datetime.strptime(end, "%Y%m%d%H")

        while current <= end_dt:
            base = current.strftime("%Y%m%d%H")
            path = os.path.join(
                self.sac_directory, base[:4], base, f"{station}.{ch[0]}.SAC"
            )
            paths.append(path)
            current += timedelta(hours=1)

        return paths

    def _read_and_concatenate_sac_traces(
        self, paths: List[str]
    ) -> Optional[SacTrace]:
        """
        Read and concatenate multiple SAC traces.

        Args:
            sac_filepaths (List[str]): List of SAC file paths.

        Returns:
            Optional[SacTrace]: Concatenated SacTrace, or None if any part fails.
        """
        traces = []
        for path in paths:
            trace = self._read_sac_trace(path)
            if trace is None:
                logger.warning(f"Skipping unreadable file: {path}")
                return None
            traces.append(trace)

        base = traces[0]
        for t in traces[1:]:
            try:
                base += t
            except ValueError as e:
                logger.warning(f"Concatenation failed: {e} | {t}")
                return None

        return base
