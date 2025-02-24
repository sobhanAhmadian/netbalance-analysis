import os

from .common import PROCESSED_DATA_DIR

SANGER_PROCESSED_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, "sanger")

SANGER_DATASET_FILE = os.path.join(
    SANGER_PROCESSED_DATA_DIR, "sanger_drug_drug_cell_associations.txt"
)
SANGER_CELL_LINE_NAMES_FILE = os.path.join(
    SANGER_PROCESSED_DATA_DIR, "cell_line_names.csv"
)
SANGER_DRUG_NAMES_FILE = os.path.join(SANGER_PROCESSED_DATA_DIR, "drug_names.csv")
SANGER_DRUG_INCH_KEYS_FILE = os.path.join(SANGER_PROCESSED_DATA_DIR, "drug_inch_keys.csv")