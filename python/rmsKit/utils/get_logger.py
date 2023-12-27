import logging
import sys


FORMAT = "%(asctime)s " + "-" * 15 + " %(levelname)s " + "-" * 15 + "\n%(message)s"


def get_logger(
        log_filename: str,
        level: int = logging.INFO,
        stdout: bool = False) -> logging.Logger:
    if not stdout:
        logging.basicConfig(
            level=level,
            filename=log_filename,
            format=FORMAT
        )

        logger = logging.getLogger()
        print("Log file will be saved to " + log_filename)
    else:
        logging.basicConfig(
            level=level,
            stream=sys.stdout,
            format=FORMAT
        )

        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)

        logger = logging.getLogger()
        logger.addHandler(file_handler)

    logging.info("Log file will be saved to " + log_filename)
    logging.info("filename : torch_optimize_loc.py")
    return logger
