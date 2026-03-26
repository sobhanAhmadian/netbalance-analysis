import os

from .common import PROCESSED_DATA_DIR, RAW_DATA_DIR

PICARD_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "picard")
PICARD_PROCESSED_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, "picard")

PICARD_DATASET_FILE = os.path.join(
    PICARD_PROCESSED_DATA_DIR, "picard_bacteria_phage_interactions.txt"
)
PICARD_BACTERIA_NAMES_FILE = os.path.join(PICARD_PROCESSED_DATA_DIR, "bacteria_names.csv")
PICARD_PHAGE_NAMES_FILE = os.path.join(PICARD_PROCESSED_DATA_DIR, "phage_names.csv")
