import datetime
import logging

from src.utils.io_utils import create_and_get_path


def create_logger_with_fh(params_str=""):
    running_timestamp = str(datetime.datetime.now().time().strftime("%H%M%S%m%d%Y"))
    log_path = create_and_get_path("logging")
    log_file = log_path + "/" + running_timestamp + "_" + params_str + ".log"

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ])
