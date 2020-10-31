import logging
import datetime

LOG_DIR = '/media/divyesh/WorkSpace/ML_Works/Projects/idea_projects/RecommenderSystem/logs/'
format = "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"


def set_log_config(level, is_file_log=False):
    """
    Log Configuration
    :param level: log level
    :param is_file_log: logs in file
    :return:
    """
    file_name = datetime.datetime.now().strftime("%Y-%m-%d")
    handlers = [logging.StreamHandler()]
    if is_file_log:
        handlers.append(logging.FileHandler("{0}/{1}.log".format(LOG_DIR, file_name)))
    logging.basicConfig(format=format, handlers=handlers, level=level)
    logging.critical('LOG LEVEL: {0}, IS_FILE_LEVEL: {1}'.format(level, is_file_log))
