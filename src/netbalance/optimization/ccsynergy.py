import numpy as np
import tensorflow as tf
from keras import callbacks

from netbalance.configs.ccsynergy import (
    CCSYNERGY_PROCESSED_DATA_DIR as PROCESSED_DATA_DIR,
)
from netbalance.data.association_data import TGData
from netbalance.evaluation import Result
from netbalance.evaluation.utils import evaluate_binary_classification_simple
from netbalance.models.ccsynergy import CCSynergyModelHandler
from netbalance.utils import get_header_format, prj_logger
from netbalance.utils.learning_utils import statified_train_test_sampler

from .interface import Trainer

logger = prj_logger.getLogger(__name__)


class CCSynergyTrainer(Trainer):

    def train(
        self,
        model_handler: CCSynergyModelHandler,
        data: TGData,
        config,
    ) -> Result:
        logger.info(get_header_format("Training the model"))
        model_handler.fe.build()

        train_data, val_data = self.split_train_dev(data)
        X_train = model_handler.fe.extract_features(
            train_data.associations[:, 0],
            train_data.associations[:, 1],
            train_data.associations[:, 2],
        )
        X_val = model_handler.fe.extract_features(
            val_data.associations[:, 0],
            val_data.associations[:, 1],
            val_data.associations[:, 2],
        )
        Y_train = train_data.associations[:, -1]
        Y_val = val_data.associations[:, -1]

        cb_check = callbacks.ModelCheckpoint(
            (f"{PROCESSED_DATA_DIR}/cache_model.keras"),
            verbose=1,
            monitor="val_loss",
            save_best_only=True,
            mode="auto",
        )
        model_handler.model.fit(
            x=X_train,
            y=Y_train,
            batch_size=config.batch_size,
            epochs=config.n_epoch,
            shuffle=True,
            validation_data=(X_val, Y_val),
            callbacks=[
                callbacks.EarlyStopping(monitor="val_loss", mode="auto", patience=10),
                cb_check,
            ],
        )
        model_handler.model = tf.keras.models.load_model(
            (f"{PROCESSED_DATA_DIR}/cache_model.keras")
        )

        preds = model_handler.predict(
            [data.associations[:, i] for i in range(data.associations.shape[1] - 1)]
        )
        result = evaluate_binary_classification_simple(
            data.associations[:, -1], preds.reshape(-1), config.threshold
        )
        return result

    def split_train_dev(self, data: TGData):
        positive_associations = data.associations[data.associations[:, -1] == 1]
        negative_associations = data.associations[data.associations[:, -1] == 0]
        (
            train_positive_indices,
            train_negative_indices,
            dev_positive_indices,
            dev_negative_indices,
        ) = statified_train_test_sampler(
            positive_associations.shape[0],
            negative_associations.shape[0],
            train_ratio=0.8,
            seed=0,
        )
        train_positive_associations = positive_associations[train_positive_indices]
        train_negative_associations = negative_associations[train_negative_indices]
        dev_positive_associations = positive_associations[dev_positive_indices]
        dev_negative_associations = negative_associations[dev_negative_indices]
        train_associations = np.concatenate(
            [train_positive_associations, train_negative_associations], axis=0
        )
        dev_associations = np.concatenate(
            [dev_positive_associations, dev_negative_associations], axis=0
        )

        train_data = TGData(
            associations=train_associations,
            cluster_a_node_names=data.cluster_a_node_names,
            cluster_b_node_names=data.cluster_b_node_names,
            cluster_c_node_names=data.cluster_c_node_names,
        )
        val_data = TGData(
            associations=dev_associations,
            cluster_a_node_names=data.cluster_a_node_names,
            cluster_b_node_names=data.cluster_b_node_names,
            cluster_c_node_names=data.cluster_c_node_names,
        )
        return train_data, val_data
