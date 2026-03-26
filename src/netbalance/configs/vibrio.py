import os

from .common import PROCESSED_DATA_DIR, RAW_DATA_DIR

VIBRIO_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "vibrio")
VIBRIO_PROCESSED_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, "vibrio")

VIBRIO_DATASET_FILE = os.path.join(VIBRIO_PROCESSED_DATA_DIR, "vibrio.txt")
VIBRIO_BACTERIA_NAMES_FILE = os.path.join(VIBRIO_PROCESSED_DATA_DIR, "strains.csv")
VIBRIO_PHAGE_NAMES_FILE = os.path.join(VIBRIO_PROCESSED_DATA_DIR, "phages.csv")
