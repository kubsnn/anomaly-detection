import logging
import colorlog


def setup_logging(name) -> logging.Logger:
    """
    Configure colored logging:
    - Log level is colored.
    - Date is gray.
    - Message is white.
    """
    logging.addLevelName(logging.WARNING, "WARN")
    logging.addLevelName(logging.ERROR, "FAIL")

    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            fmt="%(asctime)s [%(log_color)s%(levelname)s%(reset)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARN": "yellow",
                "FAIL": "red",
                "CRITICAL": "bold_red",
            },
            style="%",
        )
    )
    logger = colorlog.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger