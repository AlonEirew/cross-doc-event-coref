import logging


def create_logger(class_name, log_file):
    logger = logging.getLogger(class_name)
    logging.basicConfig(level=logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger
