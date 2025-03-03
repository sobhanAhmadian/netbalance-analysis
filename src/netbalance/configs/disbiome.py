import os

from .common import PROCESSED_DATA_DIR, RAW_DATA_DIR

DISBIOME_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "disbiome")
DISBIOME_PROCESSED_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, "disbiome")

DISBIOME_DATASET_FILE = os.path.join(
    DISBIOME_PROCESSED_DATA_DIR, "disbiome_microbe_disease_associations.txt"
)
DISBIOME_DISEASE_NAMES_FILE = os.path.join(
    DISBIOME_PROCESSED_DATA_DIR, "disease_names.csv"
)
DISBIOME_MICROBE_NAMES_FILE = os.path.join(
    DISBIOME_PROCESSED_DATA_DIR, "microbe_names.csv"
)
