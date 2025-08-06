import os

from .common import PROCESSED_DATA_DIR

BERNETT_PROCESSED_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, "bernett")

BERNETT_PROTEIN_NAMES_FILE = os.path.join(
    BERNETT_PROCESSED_DATA_DIR, "protein_names.csv"
)
BERNETT_PROTEIN_SEQUENCE_FILE = os.path.join(
    BERNETT_PROCESSED_DATA_DIR, "protein_sequences.csv"
)
BERNETT_NUCLEOTIDE_FEATURES_FILE = os.path.join(
    BERNETT_PROCESSED_DATA_DIR, "nucleotide_features.txt"
)

BERNETT_INTRA_0_FILE = os.path.join(BERNETT_PROCESSED_DATA_DIR, "intra0.txt")
BERNETT_INTRA_1_FILE = os.path.join(BERNETT_PROCESSED_DATA_DIR, "intra1.txt")
BERNETT_INTRA_2_FILE = os.path.join(BERNETT_PROCESSED_DATA_DIR, "intra2.txt")
