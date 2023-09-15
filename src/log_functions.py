"""Log functions"""
import time
from datetime import datetime, timedelta
from pytz import timezone
import logging


def build_logger(log_file_path: str) -> logging.RootLogger:
    """
    Creates logger file.

    Parameters
    ----------
    log_file_path: str
        Path to save log file

    Returns
    ----------
    logger: logging.RootLogger
        Logger with file path.
    """
    # Reference
    # https://qiita.com/r1wtn/items/d615f19e338cbfbfd4d6
    # https://www.sejuku.net/blog/23149

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file_path)
    formatter = logging.Formatter(
        fmt="%(asctime)s: %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    def japanTime(*args):
        return datetime.now(timezone("Asia/Tokyo")).timetuple()

    formatter.converter = japanTime
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def get_log_file_name(name: str = "") -> str:
    """
    Creates log file name based on executed time in Japan time.

    Parameters
    ----------
    name: str, optional
        Name of log file

    Returns
    ----------
    filename: str
        Generated log file name.
    """
    now = datetime.now(timezone("Asia/Tokyo"))
    date_time_str = now.strftime("%Y-%m-%d_%H:%M")
    filename = name + date_time_str + ".log"
    return filename


def log_execution_time(logger: logging.RootLogger) -> None:
    """
    Decorator to log function execution time to logger.

    Parameters
    ----------
    logger: logging.RootLogger
        The logger to write execution time for the function.

    Returns
    ----------
    None
    """

    def decorator(function):
        def wrapper(*args, **kwargs):
            start = time.time()
            function(*args, **kwargs)
            elapsed = time.time() - start
            elapsed_time = str(timedelta(seconds=elapsed))
            logger.info(f"Execution time: {elapsed_time}")

        return wrapper

    return decorator
