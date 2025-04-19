import logging


class CustomFormatter(logging.Formatter):
    """
    Custom log formatter that adds an icon based on the log level.

    This formatter prepends a log level-specific icon to the log message along with
    the log level, timestamp, filename, line number, and the actual message.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record with an appropriate icon.

        Parameters
        ----------
        record : logging.LogRecord
            The log record containing information about the logging event.

        Returns
        -------
        str
            The formatted log message with an icon corresponding to its log level.
        """
        level_icons = {
            "DEBUG": "🔍",
            "INFO": "✨",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "CRITICAL": "💀",
        }
        log_fmt = (
            f"{level_icons.get(record.levelname, '❓')} {record.levelname.ljust(9)}"
            f" {self.formatTime(record, '%Y-%m-%d %H:%M:%S')} "
            f"[{record.filename}:{record.lineno}] "
            f"🚀 {record.getMessage()} ✨"
        )
        return log_fmt


def setup_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Set up and return a logger with the custom formatter.

    Parameters
    ----------
    name : str
        The name of the logger.
    level : int, optional
        The logging level (default is logging.DEBUG).

    Returns
    -------
    logging.Logger
        The configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = CustomFormatter()

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
