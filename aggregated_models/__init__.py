import sys
import logging
from logging import StreamHandler

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

stdout_handler = StreamHandler(stream=sys.stdout)
stdout_handler.setLevel(logging.INFO)

logger_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

stdout_handler.setFormatter(logger_formatter)

root_logger.addHandler(stdout_handler)
