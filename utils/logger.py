from setuptools import logging
import logging
from pathlib import Path

def setup_logger(
    log_dir: str = "logs",
    log_file: str = "training.log"
):
   Path(log_dir).mkdir(parents=True, exist_ok=True)

   logger = logging.getLogger("GPT")

   logger.setLevel(logging.INFO)

   if logger.hasHandlers():
      logger.handlers.clear()
   formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S"
   )
   console_handler = logging.StreamHandler()
   console_handler.setFormatter(formatter)

   file_handler = logging.FileHandler(
        Path(log_dir) / log_file
    )
   file_handler.setFormatter(formatter)

   logger.addHandler(console_handler)
   logger.addHandler(file_handler)

   return logger