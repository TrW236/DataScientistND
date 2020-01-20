import platform
import logging
import pandas as pd

save_log = True
log_file_name = "tmp.log"

if platform.system() == "Windows":
    PATH_SEP = "\\"
if platform.system() == "Linux":
    PATH_SEP = "/"
LOGGING_FORMAT = "%(asctime)s [%(module)s] %(levelname)s - %(message)s"
LOGGING_LEVEL = logging.INFO  # or DEBUG or INFO for log

logger = logging.getLogger()
logger.setLevel(LOGGING_LEVEL)

if save_log:
    file_handler = logging.FileHandler(log_file_name, mode='a')
    file_handler.setLevel(LOGGING_LEVEL)
    file_handler.setFormatter(logging.Formatter(LOGGING_FORMAT, datefmt="%Y.%m.%d-%H:%M:%S"))
    logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(LOGGING_LEVEL)
console_handler.setFormatter(logging.Formatter(LOGGING_FORMAT, datefmt="%Y.%m.%d-%H:%M:%S"))
logger.addHandler(console_handler)

pd.set_option('expand_frame_repr', False)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 1000)
