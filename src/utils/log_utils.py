import datetime
import logging

from src import LIBRARY_ROOT


def create_logger_with_fh(params_str=""):
    running_timestamp = str(datetime.datetime.now().time().strftime("%H%M%S%m%d%Y"))
    log_file = str(LIBRARY_ROOT) + "/logging/" + running_timestamp + "_" + params_str + ".log"

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ])
