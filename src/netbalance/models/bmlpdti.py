import csv
import os
from typing import Union

import numpy as np
import scipy.sparse as sp
import torch

from netbalance.configs.bmlpdti import (BMLPDTI_PROCESSED_DATA_DIR,
                                        BMLPDTIModelConfig)
from netbalance.configs.midti import MIDTI_PROCESSED_DATA_DIR
from netbalance.methods import FeatureExtractor
from netbalance.models.modules.simple_mlp import SimpleMLP
from netbalance.utils import prj_logger
from torch import sigmoid

from .interface import AModelHandler, HandlerFactory

logger = prj_logger.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BMLPDTIFeatureExtractor(FeatureExtractor):

    def __init__(self, model_config: BMLPDTIModelConfig) -> None:
        super().__init__()
        self.model_config = model_config
        self.feature_dataset = None

    def build(self, train_associations: Union[None, np.ndarray] = None):
        data_root = MIDTI_PROCESSED_DATA_DIR

        drug_protein_matrix = self.read_txt(
            data_root + "/input/mat_drug_protein.txt", " "
        )
        logger.info(f"shape of drug_protein_matrix: {drug_protein_matrix.shape}")
        drug_sim_matrix = self.read_csv(
            data_root + "/input/dim/drug_f512_matrix708.csv"
        )
        drug_feature = self.read_csv(data_root + "/input/dim/drug_f512_feature708.csv")
        protein_sim_matrix = self.read_csv(
            data_root + "/input/dim/protein_f512_matrix1512.csv"
        )
        protein_feature = self.read_csv(
            data_root + "/input/dim/protein_f512_feature1512.csv"
        )

        # Mask Train Associations
        if train_associations is not None:
            drug_protein_matrix = np.zeros_like(drug_protein_matrix)
            for i, j, k in train_associations:
                drug_protein_matrix[i][j] = k

        drug_num = len(drug_protein_matrix)
        protein_num = len(drug_protein_matrix[0])

        drug_sim_matrix_proc = self.preprocess_adj(drug_sim_matrix)
        drug_sim_edge_index = self.get_edge_index(drug_sim_matrix_proc)
        drug_feature = self.normalize_features(drug_feature)

        protein_sim_matrix_proc = self.preprocess_adj(protein_sim_matrix)
        protein_sim_edge_index = self.get_edge_index(protein_sim_matrix_proc)
        protein_feature = self.normalize_features(protein_feature)

        self.feature_dataset = self.build_feature_dataset(
            drug_protein_matrix,
            drug_sim_matrix,
            drug_feature,
            protein_sim_matrix,
            protein_feature,
            drug_num,
            protein_num,
            drug_sim_matrix_proc,
            drug_sim_edge_index,
            protein_sim_matrix_proc,
            protein_sim_edge_index,
        )

    def build_feature_dataset(
        self,
        drug_protein_matrix,
        drug_sim_matrix,
        drug_feature,
        protein_sim_matrix,
        protein_feature,
        drug_num,
        protein_num,
        drug_sim_matrix_proc,
        drug_sim_edge_index,
        protein_sim_matrix_proc,
        protein_sim_edge_index,
    ):
        feature_dataset = dict()

        feature_dataset["dd"] = {
            "matrix": drug_sim_matrix_proc,
            "edges": drug_sim_edge_index,
            "feature": drug_feature,
        }

        feature_dataset["pp"] = {
            "matrix": protein_sim_matrix_proc,
            "edges": protein_sim_edge_index,
            "feature": protein_feature,
        }

        feature_dataset["dp"] = self.build_drug_protein_features(
            drug_protein_matrix,
            drug_feature,
            protein_feature,
            drug_num,
            protein_num,
        )

        feature_dataset["ddpp"] = self.build_ddpp_features(
            drug_protein_matrix,
            drug_sim_matrix,
            protein_sim_matrix,
            feature_dataset["dp"]["feature"],
        )

        return feature_dataset

    def build_ddpp_features(
        self, drug_protein_matrix, drug_sim_matrix, protein_sim_matrix, dp_feature
    ):
        ddpp_matrix = np.vstack(
            (
                np.hstack((drug_sim_matrix, drug_protein_matrix)),
                np.hstack((drug_protein_matrix.T, protein_sim_matrix)),
            )
        )
        ddpp_matrix = self.preprocess_adj(ddpp_matrix)
        ddpp_edge_index = self.get_edge_index(ddpp_matrix)
        x = {
            "matrix": ddpp_matrix,
            "edges": ddpp_edge_index,
            "feature": dp_feature,
        }

        return x

    def build_drug_protein_features(
        self, drug_protein_matrix, drug_feature, protein_feature, drug_num, protein_num
    ):
        dp_matrix = np.vstack(
            (
                np.hstack((np.zeros((drug_num, drug_num)), drug_protein_matrix)),
                np.hstack(
                    (drug_protein_matrix.T, np.zeros((protein_num, protein_num)))
                ),
            )
        )
        logger.info("dp_matrix shape: {}".format(dp_matrix.shape))
        dp_matrix = self.preprocess_adj(dp_matrix)
        dp_edge_index = self.get_edge_index(dp_matrix)
        dp_feature = np.vstack(
            (
                np.hstack((drug_feature, drug_protein_matrix)),
                np.hstack((drug_protein_matrix.T, protein_feature)),
            )
        )
        dp_feature = self.normalize_features(dp_feature)

        return {
            "matrix": dp_matrix,
            "edges": dp_edge_index,
            "feature": dp_feature,
        }

    def read_txt(self, path, delim):
        reader = np.loadtxt(path, dtype=int, delimiter=delim)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return np.array(md_data)

    def read_csv(self, path):
        with open(path, "r", newline="") as csv_file:
            reader = csv.reader(csv_file)
            md_data = []
            md_data += [[float(i) for i in row] for row in reader]
            return np.array(md_data)
            # return torch.Tensor(md_data)

    def preprocess_adj(self, adj):
        """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
        adj_normalized = self.normalize_adj(adj + np.eye(adj.shape[0]))
        return torch.Tensor(adj_normalized)

    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).toarray()

    def normalize_features(self, feat):
        degree = np.asarray(feat.sum(1)).flatten()
        degree[degree == 0.0] = np.inf
        degree_inv = 1.0 / degree
        degree_inv_mat = sp.diags([degree_inv], [0])
        feat_norm = degree_inv_mat.dot(feat)
        return torch.Tensor(feat_norm)

    def get_edge_index(self, matrix):
        edge_index = [[], []]
        for i in range(matrix.size(0)):
            for j in range(matrix.size(1)):
                if matrix[i][j] != 0:
                    edge_index[0].append(i)
                    edge_index[1].append(j)
        return torch.LongTensor(edge_index)

    def extract_features(self, a_nodes, b_nodes):
        a_features = self.feature_dataset["dd"]["feature"][a_nodes]
        b_features = self.feature_dataset["pp"]["feature"][b_nodes]
        features = torch.cat((a_features, b_features), 1).detach()
        return features


class BMLPDTIModelHandler(AModelHandler):

    def __init__(self, model_config: BMLPDTIModelConfig) -> None:
        super().__init__(model_config)

    def destroy(self):
        del self.model
        del self.fe

    def predict_impl(self, node_lists: list[np.ndarray]):
        a_nodes, b_nodes = node_lists
        dp_embedd = self.fe.extract_features(a_nodes, b_nodes).numpy()
        dp_embedd = torch.tensor(dp_embedd).to(self.model_config.device)
        preds = sigmoid(self.model(dp_embedd).flatten()).cpu().detach().numpy()
        return preds

    def summary(self):
        pass

    def _build_model(self):
        return SimpleMLP(**self.model_config.get_model_kwargs())

    def _build_feature_extractor(self):
        return BMLPDTIFeatureExtractor(self.model_config)


class BMLPDTIHandlerFactory(HandlerFactory):

    def __init__(self, model_config: BMLPDTIModelConfig) -> None:
        super().__init__()
        self.model_config = model_config

    def create_handler(self) -> BMLPDTIModelHandler:
        return BMLPDTIModelHandler(self.model_config)
