import os
import logging
from datetime import datetime



def build_logger(log_path, set_level=0):
    os.makedirs(log_path, exist_ok=True)
    time_created = datetime.today().strftime('%Y-%m-%d_%H-%M')

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%Y-%m-%d %H:%M:%S')
    fileHandler = logging.FileHandler(log_path / f'{time_created}.txt')
    fileHandler.setFormatter(formatter)

    logger_levels = [
        logging.DEBUG, # set_level = 0
        logging.INFO, # set_level = 1
        logging.WARNING, # set_level = 2
        logging.ERROR, # set_level = 3
        logging.CRITICAL # set_level = 4
    ]

    logger = logging.getLogger(name=f'{log_path}')
    logger.setLevel(level=logger_levels[set_level])
    logger.addHandler(fileHandler)
    return logger



if __name__ == '__main__':
    from pathlib import Path

    log_path = 'log-test'
    level = 0
    logger = Basic_Logger(log_path=Path(log_path), set_level=level)
    message = 'good-good'
    logger.debug(message)