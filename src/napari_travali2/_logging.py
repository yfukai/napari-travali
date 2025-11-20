import logging
import traceback
from functools import wraps
from ._consts import LOGGING_PATH
import os
from os import path

logging_path = path.join(path.expanduser("~"), LOGGING_PATH)
os.makedirs(path.dirname(logging_path), exist_ok=True)

logger = logging.getLogger(__name__)
handler = logging.FileHandler(logging_path)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logger.propagate = False


def log_error(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BaseException:
            logger.error(traceback.format_exc())

    return wrapped
