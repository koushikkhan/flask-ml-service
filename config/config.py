import os
import logging

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(ROOT, 'data')
MODEL_PATH = os.path.join(ROOT, 'model')
LOG_PATH = os.path.join(ROOT, 'logs')
MODEL_FILE_NAME = "classifier.pickle"

# setup log file time stamp format
logging.basicConfig(
    filename=os.path.join(LOG_PATH, 'application.log'),
    filemode="a+",
    level=logging.DEBUG,
    format='%(asctime)s %(message)s', 
    datefmt='%m/%d/%Y %I:%M:%S %p'
)

# change permission
# os.chmod(DATA_PATH,  0o777)
# os.chmod(LOG_PATH,  0o777)
# os.chmod(os.path.join(LOG_PATH, 'application.log'), 0o777)