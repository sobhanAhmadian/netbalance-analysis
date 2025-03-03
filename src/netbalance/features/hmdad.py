import pandas as pd

from netbalance.configs.hmdad import (
    HMDAD_DATASET_FILE,
    HMDAD_DISEASE_NAMES_FILE,
    HMDAD_HSDN_DISEASE_SYMPTOM_SIMILARITY_FILE,
    HMDAD_MDAD_BIOFILM_DRUG_BASED_SIMILARITY_FILE,
    HMDAD_MESH_DISEASE_SEMANTIC_SIMILARITY_FILE,
    HMDAD_MICROBE_NAMES_FILE,
    HMDAD_PROCESSED_DATA_DIR,
)
from netbalance.utils import data as data_utils
from netbalance.utils import prj_logger

logger = prj_logger.getLogger(__name__)

from netbalance.features.bipartite_graph_dataset import ADataset


def get_hsdn_disease_symptom_similarity():
    """Retrieves the similarity matrix for diseases based on symptoms.

    This similarity matrix is calculated based on the Human Symptom-Disease Network (HSDN).
    Ref: MVGAEW

    Returns:
        numpy.ndarray: The similarity matrix for diseases based on symptoms.
    """
    disease_sim_symptom = pd.read_csv(
        HMDAD_HSDN_DISEASE_SYMPTOM_SIMILARITY_FILE, index_col=[0]
    ).to_numpy()

    logger.info(
        "HSDN symptom similarity for diseases in HMDAD dataset has been loaded."
    )
    return disease_sim_symptom


def get_mdad_biofilm_microbe_drug_based_similarity():
    """Get the drug-based similarity for microbes.

    This similarity matrix is calculated based on the MDAD dataset and a biofilm.
    Ref: MVGAEW

    Returns:
        numpy.ndarray: The drug-based similarity matrix for microbes.
    """
    microbe_sim_func = pd.read_csv(
        HMDAD_MDAD_BIOFILM_DRUG_BASED_SIMILARITY_FILE, index_col=0
    ).to_numpy()

    logger.info(
        "MDAD_BIOFILM drug based similarites for microbes in HMDAD dataset has been loaded."
    )
    return microbe_sim_func


def get_mesh_disease_semantic_simialrity():
    """Retrieves the semantic similarity for diseases.

    The semantic similarity matrix is calculated based on the MeSH ids.
    Ref: MVGAEW

    Returns:
        numpy.ndarray: The semantic similarity matrix for diseases.
    """
    disease_sim_sematic = pd.read_csv(
        HMDAD_MESH_DISEASE_SEMANTIC_SIMILARITY_FILE, index_col=[0]
    ).to_numpy()

    logger.info(
        "MeSH semantic similarity for diseases in HMDAD dataset has been loaded."
    )
    return disease_sim_sematic


class HMDADDataset(ADataset):

    def __init__(self) -> None:
        super().__init__(["microbe", "disease"])

    def get_node_names(self):
        microbe_names = self.get_cluster_a_node_names()
        disease_names = self.get_cluster_b_node_names()
        return [microbe_names, disease_names]

    def get_cluster_a_node_names(self):
        names = pd.read_csv(HMDAD_MICROBE_NAMES_FILE)
        return list(names.iloc[:, 1])

    def get_cluster_b_node_names(self):
        names = pd.read_csv(HMDAD_DISEASE_NAMES_FILE)
        return list(names.iloc[:, 1])

    def get_dataset_file_path(self):
        return HMDAD_DATASET_FILE
