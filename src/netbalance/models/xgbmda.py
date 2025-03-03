import os

import numpy as np
import torch
import torch_geometric as tg
from torch_geometric.nn import Node2Vec
from xgboost import XGBClassifier

from netbalance.configs.xgbmda import XGBMDA_PROCESSED_DATA_DIR, XGBMDAModelConfig
from netbalance.features.kgnmda import get_entities, get_relations
from netbalance.methods import FeatureExtractor
from netbalance.models.modules import SimpleMLP
from netbalance.utils import get_header_format, prj_logger

from .interface import AModelHandler, HandlerFactory

logger = prj_logger.getLogger(__name__)


class XGBMDAFeatureExtractor(FeatureExtractor):

    def __init__(self, model_config: XGBMDAModelConfig) -> None:
        super().__init__()
        self.model_config = model_config
        self.device = model_config.device
        self.homo = self.get_homogeneous_graph()
        self.node2vec_model = Node2Vec(
            self.homo.edge_index, **self.model_config.get_feature_extractor_kwargs()
        )

        if not os.path.exists(XGBMDA_PROCESSED_DATA_DIR):
            os.makedirs(XGBMDA_PROCESSED_DATA_DIR, exist_ok=True)

        model_name = ""
        for key in self.model_config.get_feature_extractor_kwargs():
            model_name += (
                f"{key}_{self.model_config.get_feature_extractor_kwargs()[key]}_"
            )
        self.model_save_path = os.path.join(
            XGBMDA_PROCESSED_DATA_DIR, f"node2vec_{model_name}.pth"
        )
        logger.info(f"Path for Saving Node2Vec Model : {self.model_save_path}")

    def build(self, op_config):
        if os.path.exists(self.model_save_path):
            self.load_node2vec_fe()
        else:
            logger.info("Model was not saved!")
            self.train_node2vec_fe(op_config)
            self.save_node2vec_fe()

        node_list = self.get_homogeneous_graph().x.squeeze().detach().tolist()
        self.homo.x = self._predict(node_list=node_list)

    def train_node2vec_fe(self, op_config):
        logger.info(get_header_format("Training Feature Extractor"))
        logger.info(f"Creating {op_config.fe_optimizer} with lr : {op_config.fe_lr}")
        optimizer = op_config.fe_optimizer(
            self.node2vec_model.parameters(), lr=op_config.fe_lr
        )
        logger.info(f"moving model to {op_config.device}")
        self.node2vec_model.to(op_config.device)
        self.node2vec_model.train()

        loader = self.node2vec_model.loader(**op_config.get_fe_loader_configuration())

        total_loss = 0
        running_loss = 0
        logger.info("Start batch optimizing")
        for epoch in range(op_config.fe_num_epochs):
            for j, (pos_rw, neg_rw) in enumerate(loader, 0):
                loss = self.node2vec_model.loss(
                    pos_rw.to(op_config.device), neg_rw.to(op_config.device)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                total_loss += loss.item()

                if j % op_config.fe_report_size == op_config.fe_report_size - 1:
                    loss = running_loss / op_config.fe_report_size
                    logger.info(f"loss: {loss:.4f}    [{j + 1:5d}]")
                    running_loss = 0

            total_loss = total_loss / len(loader)

            logger.info(f"Epoch {epoch + 1} : Loss {total_loss:.4f}")

    def load_node2vec_fe(self):
        logger.info(get_header_format("Loading Feature Extractor"))
        self.node2vec_model.load_state_dict(
            torch.load(self.model_save_path, map_location=torch.device(self.device))
        )
        self.node2vec_model.eval()
        logger.info("Model was loaded!")

    def save_node2vec_fe(self):
        torch.save(self.node2vec_model.state_dict(), self.model_save_path)
        logger.info("Model was saved!")

    def extract_features(self, mic_array, dis_array):
        m_embedd = self.homo.x[mic_array]
        d_embedd = self.homo.x[dis_array]
        return torch.cat((d_embedd, m_embedd), 1).detach()

    def get_homogeneous_graph(self):
        homo = tg.data.Data()
        homo.x = (
            torch.tensor(get_entities()["id"].tolist()).reshape(-1, 1).to(self.device)
        )
        homo.edge_index = torch.tensor(
            [get_relations()["head"].tolist(), get_relations()["tail"].tolist()]
        ).to(self.device)
        logger.info("Graph was calculated!")
        return homo

    def _predict(self, node_list):
        return self.node2vec_model()[node_list].detach()


class XGBMDAModelHandler(AModelHandler):

    def __init__(self, model_config: XGBMDAModelConfig) -> None:
        super().__init__(model_config)

    def destroy(self):
        del self.model
        del self.fe

    def predict_impl(self, node_lists: list[np.ndarray]):
        a_nodes, b_nodes = node_lists
        md_embedd = self.fe.extract_features(a_nodes, b_nodes).detach().numpy()
        return self.model.predict(md_embedd)

    def summary(self):
        raise NotImplementedError

    def _build_model(self):
        return XGBClassifier(
            n_estimators=1000,
            max_depth=2,
            learning_rate=0.01,
            objective="binary:logistic",
        )

    def _build_feature_extractor(self):
        return XGBMDAFeatureExtractor(self.model_config)


class XGBMDAHandlerFactory(HandlerFactory):

    def __init__(self, model_config: XGBMDAModelConfig) -> None:
        super().__init__()
        self.model_config = model_config

    def create_handler(self) -> XGBMDAModelHandler:
        return XGBMDAModelHandler(self.model_config)
