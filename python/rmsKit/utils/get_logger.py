"""Get logger."""
import logging
import sys


FORMAT = "%(asctime)s " + "-" * 2 + " %(levelname)-5s " + " %(message)s"


def get_logger(
        log_filename: str,
        level: int = logging.INFO,
        stdout: bool = False) -> logging.Logger:
    """Get logger."""
    if not stdout:
        logging.basicConfig(
            level=level,
            filename=log_filename,
            format=FORMAT,
            datefmt="%m.%d %H:%M"
        )

        logger = logging.getLogger()
        print("Log file will be saved to " + log_filename)
    else:
        logging.basicConfig(
            level=level,
            stream=sys.stdout,
            format=FORMAT,
            datefmt="%m.%d %H:%M"
        )

        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(FORMAT)
        file_handler.setFormatter(file_formatter)

        logger = logging.getLogger()
        logger.addHandler(file_handler)

    logging.info("Log file will be saved to " + log_filename)
    return logger
