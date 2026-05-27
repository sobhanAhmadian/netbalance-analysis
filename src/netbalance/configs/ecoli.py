import os

from .common import PROCESSED_DATA_DIR, RAW_DATA_DIR

ECOLI_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "ecoli")
ECOLI_PROCESSED_DATA_DIR = os.path.join(PROCESSED_DATA_DIR, "ecoli")

ECOLI_DATASET_FILE = os.path.join(ECOLI_PROCESSED_DATA_DIR, "ecoli.txt")
ECOLI_BACTERIA_NAMES_FILE = os.path.join(ECOLI_PROCESSED_DATA_DIR, "strains.csv")
ECOLI_PHAGE_NAMES_FILE = os.path.join(ECOLI_PROCESSED_DATA_DIR, "phages.csv")
ECOLI_STRAIN_FEATURE_NAMES_FILE = os.path.join(
    ECOLI_PROCESSED_DATA_DIR, "strains-features-names.csv"
)
ECOLI_PHAGE_FEATURE_NAMES_FILE = os.path.join(
    ECOLI_PROCESSED_DATA_DIR, "phages-features-names.csv"
)
ECOLI_PHAGE_FEATURES_ANNOTATIONS_FILE = os.path.join(
    ECOLI_PROCESSED_DATA_DIR, "phages-features-annotations.csv"
)
ECOLI_STRAIN_FEATURES_ANNOTATIONS_FILE = os.path.join(
    ECOLI_PROCESSED_DATA_DIR, "strains-features-annotations.csv"
)

ANNOTATION_CATEGORIES_DIR = os.path.join(
    ECOLI_PROCESSED_DATA_DIR, "annotation-categories"
)

PHAGE_ANNOTATION_CATEGORIES_FILE = os.path.join(
    ANNOTATION_CATEGORIES_DIR, "phages-categories.csv"
)
STRAIN_ANNOTATION_CATEGORIES_FILE = os.path.join(
    ANNOTATION_CATEGORIES_DIR, "strains-categories.csv"
)
