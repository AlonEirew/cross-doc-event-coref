import logging


def create_logger_with_fh(class_name, log_file):
    logger = logging.getLogger(class_name)
    logging.basicConfig(level=logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger


def create_logger(class_name):
    logger = logging.getLogger(class_name)
    logging.basicConfig(level=logging.DEBUG)
    return logger
