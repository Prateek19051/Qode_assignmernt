"""
Central logging setup.

Why a separate module: every part of the pipeline (scraper, cleaner,
storage, analysis) needs logging, and we want them all to log in the
same format, to the same rotating file, instead of each file setting
up its own ad-hoc logger.
"""
import logging
import logging.handlers
import os


def get_logger(name: str, log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    """Return a logger that writes to both the console and a rotating file.

    Args:
        name: usually __name__ of the calling module, so log lines show
              where they came from.
        log_dir: folder where log files are written.
        level: logging level as a string, e.g. "INFO", "DEBUG".
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        # Logger already configured (e.g. re-imported module) - don't add
        # duplicate handlers, which would print every line twice.
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotate at 5MB, keep 3 old copies, so logs from a long scraping run
    # don't grow forever.
    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "pipeline.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
