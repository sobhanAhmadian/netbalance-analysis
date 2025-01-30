import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

from netbalance.configs import LOG_DIR

os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, "project.log")

# Create a rotating file handler
file_handler = TimedRotatingFileHandler(log_file_path, when="H", encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[file_handler],
)


def get_header_format(message, size="l"):
    if size == "s":
        message = "{:#^40}".format(f" {message} ")
    elif size == "m":
        message = "{:#^50}".format(f" {message} ")
    elif size == "l":
        message = "{:#^60}".format(f" {message} ")
    elif size == "xl":
        message = "{:#^70}".format(f" {message} ")
    elif size == "xxl":
        message = "{:#^80}".format(f" {message} ")
    else:
        raise ValueError("size parameter is not valid!")
    return message
