import logging
import traceback
from functools import wraps
from ._consts import LOGGING_PATH
import os
from os import path

logging_path = path.join(path.expanduser("~"),LOGGING_PATH)
os.makedirs(path.dirname(logging_path), exist_ok=True)

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler(logging_path))
logger.propagate = True


def log_error(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BaseException:
            logger.error(traceback.format_exc())

    return wrapped
