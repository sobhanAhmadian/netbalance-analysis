import csv
import math
import os
from typing import Union

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import softmax
from torch.nn.parameter import Parameter

from netbalance.configs.midti import MIDTI_PROCESSED_DATA_DIR, MIDTIModelConfig
from netbalance.methods import FeatureExtractor
from netbalance.utils import get_header_format, prj_logger

from .interface import AModelHandler, HandlerFactory

logger = prj_logger.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj, input):
        try:
            input = input.float()
        except:
            pass

        support = torch.matmul(input, self.weight)  # .clone()

        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN_hete(nn.Module):

    def __init__(self, hidden_c, output_hete):
        super(GCN_hete, self).__init__()

        self.hete = GraphConvolution(hidden_c, output_hete)
        self.hete2 = GraphConvolution(output_hete, output_hete)
        self.hete3 = GraphConvolution(output_hete, output_hete)

    def forward(self, adj_c, features_c):

        # gcn_layer=3
        out1 = torch.relu(self.hete(adj_c, features_c))
        out2 = torch.relu(self.hete2(adj_c, out1))
        out3 = torch.relu(self.hete3(adj_c, out2))
        return out1, out2, out3


class GCN_homo(nn.Module):

    def __init__(self, hidden_homo, output_homo):
        super(GCN_homo, self).__init__()

        self.gcn_homo = GraphConvolution(hidden_homo, output_homo)
        self.gcn_homo2 = GraphConvolution(output_homo, output_homo)
        self.gcn_homo3 = GraphConvolution(output_homo, output_homo)

    def forward(self, adj, features):

        # gcn_layer=3
        out_d1 = torch.relu(self.gcn_homo(adj, features))
        out_d2 = torch.relu(self.gcn_homo2(adj, out_d1))
        out_d3 = torch.relu(self.gcn_homo3(adj, out_d2))
        return out_d1, out_d2, out_d3


class GCN_bi(nn.Module):

    def __init__(self, hidden_bi, output_bi):
        super(GCN_bi, self).__init__()

        # inter drug-protein graph
        self.gcn_bi_dp = GraphConvolution(hidden_bi, output_bi)
        self.gcn_bi_dp2 = GraphConvolution(output_bi, output_bi)
        self.gcn_bi_dp3 = GraphConvolution(output_bi, output_bi)

    def forward(self, adj_dp, features_dp):

        # gcn_layer=3
        out_dp1 = torch.relu(self.gcn_bi_dp(adj_dp, features_dp))
        out_dp2 = torch.relu(self.gcn_bi_dp2(adj_dp, out_dp1))
        out_dp3 = torch.relu(self.gcn_bi_dp3(adj_dp, out_dp2))
        return out_dp1, out_dp2, out_dp3


class MHAtt(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super(MHAtt, self).__init__()

        self.linear_v = nn.Linear(hid_dim, hid_dim)
        self.linear_k = nn.Linear(hid_dim, hid_dim)
        self.linear_q = nn.Linear(hid_dim, hid_dim)
        self.linear_merge = nn.Linear(hid_dim, hid_dim)
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.nhead = n_heads

        self.dropout = nn.Dropout(dropout)
        self.hidden_size_head = int(self.hid_dim / self.nhead)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = (
            self.linear_v(v)
            .view(n_batches, -1, self.nhead, self.hidden_size_head)
            .transpose(1, 2)
        )

        k = (
            self.linear_k(k)
            .view(n_batches, -1, self.nhead, self.hidden_size_head)
            .transpose(1, 2)
        )

        q = (
            self.linear_q(q)
            .view(n_batches, -1, self.nhead, self.hidden_size_head)
            .transpose(1, 2)
        )

        atted = self.att(v, k, q, mask)  # 1,8,1,64
        atted = (
            atted.transpose(1, 2).contiguous().view(n_batches, -1, self.hid_dim)
        )  # 1,1,512

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)  # 64

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # 1,8,1,1

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


class DTA_TDA(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super(DTA_TDA, self).__init__()

        self.mhatt = MHAtt(hid_dim, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, x, y, y_mask=None):

        # x as V while y as Q and K
        x = self.norm(x + self.dropout(self.mhatt(y, y, x, y_mask)))

        return x


class SA(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super(SA, self).__init__()

        self.mhatt = MHAtt(hid_dim, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, x, mask=None):

        x = self.norm(x + self.dropout(self.mhatt(x, x, x, mask)))

        return x


class Deep_inter_att(nn.Module):
    def __init__(self, dim, nhead, dropout):
        super(Deep_inter_att, self).__init__()
        self.sda = SA(dim, nhead, dropout)
        self.sta = SA(dim, nhead, dropout)
        self.dta = DTA_TDA(dim, nhead, dropout)
        self.tda = DTA_TDA(dim, nhead, dropout)

    def forward(
        self,
        drug_vector,
        protein_vector,
    ):
        drug_vector = self.sda(drug_vector, None)  # self-attention
        protein_vector = self.sta(protein_vector, None)  # self-attention
        drug_covector = self.dta(
            drug_vector, protein_vector, None
        )  # drug-target-attention
        protein_covector = self.tda(
            protein_vector, drug_vector, None
        )  # target-drug-attention

        return drug_covector, protein_covector


class GCNLayer(nn.Module):
    def __init__(self, input_d, input_p, dim):
        super(GCNLayer, self).__init__()
        self.input_d = input_d
        self.input_p = input_p
        # self.lin_d = torch.nn.Linear(dim, dim)
        # self.lin_p = torch.nn.Linear(dim, dim)

        self.gcn_homo_d = GCN_homo(input_d, dim)
        self.gcn_homo_p = GCN_homo(input_p, dim)
        self.gcn_bi = GCN_bi(input_d + input_p, dim)
        self.gcn_hete = GCN_hete(input_d + input_p, dim)

    def forward(self, datasetF):
        # l=3
        x_d_dr1, x_d_dr2, x_d_dr3 = self.gcn_homo_d(
            datasetF["dd"]["matrix"].to(device), datasetF["dd"]["feature"].to(device)
        )
        y_p_pro1, y_p_pro2, y_p_pro3 = self.gcn_homo_p(
            datasetF["pp"]["matrix"].to(device), datasetF["pp"]["feature"].to(device)
        )
        dp1, dp2, dp3 = self.gcn_bi(
            datasetF["dp"]["matrix"].to(device), datasetF["dp"]["feature"].to(device)
        )
        ddpp1, ddpp2, ddpp3 = self.gcn_hete(
            datasetF["ddpp"]["matrix"].to(device),
            datasetF["ddpp"]["feature"].to(device),
        )

        x_d_dr4 = dp1[: self.input_d, :]
        y_p_pro4 = dp1[self.input_d :, :]
        x_d_dr5 = dp2[: self.input_d, :]
        y_p_pro5 = dp2[self.input_d :, :]
        x_d_dr6 = dp3[: self.input_d, :]
        y_p_pro6 = dp3[self.input_d :, :]

        x_d_dr7 = ddpp1[: self.input_d, :]
        y_p_pro7 = ddpp1[self.input_d :, :]
        x_d_dr8 = ddpp2[: self.input_d, :]
        y_p_pro8 = ddpp2[self.input_d :, :]
        x_d_dr9 = ddpp3[: self.input_d, :]
        y_p_pro9 = ddpp3[self.input_d :, :]

        x_d_dr = torch.stack(
            (
                x_d_dr1,
                x_d_dr2,
                x_d_dr3,
                x_d_dr4,
                x_d_dr5,
                x_d_dr6,
                x_d_dr7,
                x_d_dr8,
                x_d_dr9,
            ),
            0,
        )  # torch.stack()的作用是在新的维度0上把n个同size的矩阵进行拼接.shape为9，n,512
        y_p_pro = torch.stack(
            (
                y_p_pro1,
                y_p_pro2,
                y_p_pro3,
                y_p_pro4,
                y_p_pro5,
                y_p_pro6,
                y_p_pro7,
                y_p_pro8,
                y_p_pro9,
            ),
            0,
        )
        # print('x_d_dr.shape,y_p_pro.shape',x_d_dr.shape,y_p_pro.shape)

        x_d_dr = torch.transpose(x_d_dr, 0, 1)  # shape=(708,9,512)
        y_p_pro = torch.transpose(y_p_pro, 0, 1)  # shape=(1512,9,512)

        # print('x_d_dr, y_p_pro:', x_d_dr, y_p_pro)
        return x_d_dr, y_p_pro


class DTI_pre(nn.Module):
    def __init__(
        self,
        input_d,
        input_p,
        dim,
        layer_output,
        layer_IA,
        nhead,
        dropout,
        attention="DIA",
    ):
        super(DTI_pre, self).__init__()

        self.gcnlayer = GCNLayer(input_d, input_p, dim)
        self.attention = attention

        # the number of attention layers
        self.layer_IA = layer_IA

        self.DIA_ModuleList = nn.ModuleList(
            [Deep_inter_att(dim, nhead, dropout) for _ in range(layer_IA)]
        )
        self.dr_lin = nn.Linear((layer_IA + 1) * dim, dim)
        self.pro_lin = nn.Linear((layer_IA + 1) * dim, dim)

        # the number of output layers
        self.layer_output = layer_output

        # self.W_out = nn.ModuleList([nn.Linear(2*dim, dim),nn.Linear(dim, 128),nn.Linear(128, 64)]) #mlp=4
        self.W_out = nn.ModuleList(
            [nn.Linear(2 * dim, dim), nn.Linear(dim, 128)]
        )  # mlp=3
        self.W_interaction = nn.Linear(128, 2)

        # self._init_weight()

    def forward(self, id0, id1, datasetF):

        global drug_vector_co, protein_vector_co, drug_vector, protein_vector
        x_d_dr, y_p_pro = self.gcnlayer(datasetF)  # GCN Module

        drugs = x_d_dr[id0, :, :]  # (1,9,512)
        proteins = y_p_pro[id1, :, :]  # (1,9,512)

        # DIA(Deep inteactive attention)
        for i in range(self.layer_IA):
            drug_vector, protein_vector = self.DIA_ModuleList[i](drugs, proteins)
            if i == 0:
                drug_vector_co, protein_vector_co = torch.cat(
                    [drugs, drug_vector], dim=-1
                ), torch.cat([proteins, protein_vector], dim=-1)
            else:
                drug_vector_co, protein_vector_co = torch.cat(
                    [drug_vector_co, drug_vector], dim=-1
                ), torch.cat([protein_vector_co, protein_vector], dim=-1)
                # print(drug_vector_co.shape, protein_vector_co.shape) #(1,9,512*4)
        drug_vector, protein_vector = self.dr_lin(drug_vector_co), self.pro_lin(
            protein_vector_co
        )  # (1,9,512)

        drug_covector = drug_vector.mean(dim=1)
        protein_covector = protein_vector.mean(dim=1)

        # # without DIA
        # drug_covector = drugs.mean(dim=1)
        # protein_covector = proteins.mean(dim=1)

        cat_vector = torch.cat((drug_covector, protein_covector), 1)
        cat_vector0 = cat_vector

        # cat_vector = torch.cat((drug_covector, protein_covector), 1)
        for j in range(self.layer_output - 1):
            cat_vector = torch.tanh(self.W_out[j](cat_vector))
        predicted = self.W_interaction(cat_vector)
        # print('predicted_interaction_shape', predicted_interaction.shape)
        # print('predicted_interaction',predicted_interaction)

        return cat_vector0, predicted


class MIDTIFeatureExtractor(FeatureExtractor):

    def __init__(self, model_config: MIDTIModelConfig) -> None:
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
        # print(reader)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return np.array(md_data)
        # return torch.Tensor(md_data)

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
        pass


class MIDTIModelHandler(AModelHandler):

    def __init__(self, model_config: MIDTIModelConfig) -> None:
        super().__init__(model_config)

    def destroy(self):
        del self.model
        del self.fe

    def predict_impl(self, node_lists: list[np.ndarray]):
        a_nodes, b_nodes = node_lists

        if isinstance(a_nodes, np.ndarray):
            a_nodes = torch.tensor(a_nodes)
        if isinstance(b_nodes, np.ndarray):
            b_nodes = torch.tensor(b_nodes)

        feature_dataset = self.fe.feature_dataset

        _, predicted = self.model(
            a_nodes.type(torch.long).to(self.model_config.device),
            b_nodes.type(torch.long).to(self.model_config.device),
            feature_dataset,
        )
        predicted = softmax(predicted, 1).cpu().detach().numpy()
        predicted = np.array(list(map(lambda x: x[1], predicted))).flatten()

        logger.info(f"predicted: {predicted.shape}")
        return predicted

    def summary(self):
        pass

    def _build_model(self):
        return DTI_pre(
            self.model_config.drug_num,
            self.model_config.protein_num,
            self.model_config.dim,
            self.model_config.layer_output,
            self.model_config.layer_IA,
            self.model_config.nhead,
            self.model_config.dropout,
        ).to(device)

    def _build_feature_extractor(self):
        return MIDTIFeatureExtractor(self.model_config)

    def save_model(self):
        os.makedirs(MIDTI_PROCESSED_DATA_DIR, exist_ok=True)
        path = os.path.join(MIDTI_PROCESSED_DATA_DIR, "model.pth")
        torch.save(self.model.state_dict(), path)
        logger.info("Model saved to {}".format(path))

    def load_model(self):
        path = os.path.join(MIDTI_PROCESSED_DATA_DIR, "model.pth")
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(path, weights_only=True))
        logger.info("Model loaded from {}".format(path))


class MIDTIHandlerFactory(HandlerFactory):

    def __init__(self, model_config: MIDTIModelConfig) -> None:
        super().__init__()
        self.model_config = model_config

    def create_handler(self) -> MIDTIModelHandler:
        return MIDTIModelHandler(self.model_config)
