import numpy as np

NOSEL_VALUE = np.iinfo(np.uint32).max
NEW_LABEL_VALUE = NOSEL_VALUE - 1

LOGGING_PATH = ".napari-travali/log.txt"

SELECTED_COLOR = (1, 0, 0, 1)
DAUGHTER_COLORS = [
    (0, 1, 0, 1),
    (0, 0, 1, 1)
]