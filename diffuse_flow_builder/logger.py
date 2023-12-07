"""Logger."""

import logging
import sys
from datetime import datetime

from diffuse_flow_builder.utils.constant import PATH_LOGS


PATH_LOGS.mkdir(exist_ok=True)
_filename = datetime.utcnow().strftime("%Y_%m_%d") + ".log"
_path_file = PATH_LOGS / _filename

_fmt = "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"

log_conf = {
    'filename': _path_file,
    'level': logging.INFO,
    'datefmt': '%Y-%m-%d %H:%M:%S',
    'format': _fmt
}
logging.basicConfig(**log_conf, force=True)

logFormatter = logging.Formatter(_fmt)
logger = logging.getLogger()

# fileHandler = logging.FileHandler(_path_file)
# fileHandler.setFormatter(logFormatter)
# logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)
