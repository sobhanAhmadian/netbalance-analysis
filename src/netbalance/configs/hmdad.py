import os

from .common import PROCESSED_DATA_DIR, RAW_DATA_DIR

HMDAD_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "hmdad")
HMDAD_PROCESSED_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, "hmdad")

HMDAD_DATASET_FILE = os.path.join(
    HMDAD_PROCESSED_DATA_DIR, "hmdad_microbe_disease_associations.npy"
)
HMDAD_DISEASE_NAMES_FILE = os.path.join(HMDAD_PROCESSED_DATA_DIR, "disease_names.csv")
HMDAD_MICROBE_NAMES_FILE = os.path.join(HMDAD_PROCESSED_DATA_DIR, "microbe_names.csv")
HMDAD_MESH_DISEASE_SEMANTIC_SIMILARITY_FILE = os.path.join(
    HMDAD_PROCESSED_DATA_DIR, "mesh_disease_semantic_similarity.csv"
)
HMDAD_HSDN_DISEASE_SYMPTOM_SIMILARITY_FILE = os.path.join(
    HMDAD_PROCESSED_DATA_DIR, "hsdn_disease_symptom_similarity.csv"
)
HMDAD_MDAD_BIOFILM_DRUG_BASED_SIMILARITY_FILE = os.path.join(
    HMDAD_PROCESSED_DATA_DIR, "mdad_biofilm_microbe_drug_based_similarity.csv"
)
