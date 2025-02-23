import os

from .common import PROCESSED_DATA_DIR, RAW_DATA_DIR

LUODTI_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "luodti")
LUODTI_PROCESSED_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, "luodti")

LUODTI_RAW_DATASET_FILE = os.path.join(LUODTI_RAW_DATA_DIR, "mat_drug_protein.txt")
LUODTI_RAW_PROTEIN_NAMES_FILE = os.path.join(LUODTI_RAW_DATA_DIR, "protein.txt")
LUODTI_RAW_DRUG_NAMES_FILE = os.path.join(LUODTI_RAW_DATA_DIR, "drug.txt")

LUODTI_DATASET_FILE = os.path.join(
    LUODTI_PROCESSED_DATA_DIR, "luodti_drug_protein_associations.txt"
)
LUODTI_PROTEIN_NAMES_FILE = os.path.join(LUODTI_PROCESSED_DATA_DIR, "protein_names.csv")
LUODTI_DRUG_NAMES_FILE = os.path.join(LUODTI_PROCESSED_DATA_DIR, "drug_names.csv")
