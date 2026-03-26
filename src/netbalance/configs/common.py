import os

from dotenv import load_dotenv

from netbalance import ROOT_DIR

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", ROOT_DIR + "/data_repository")

RAW_DATA_DIR = DATA_DIR + "/raw"
PROCESSED_DATA_DIR = DATA_DIR + "/processed"
LOG_DIR = DATA_DIR + "/log"
MODEL_SAVED_DIR = DATA_DIR + "/model"

RESULTS_DIR = DATA_DIR + "/results"
