import numpy as np
from sklearn.impute import SimpleImputer

from netbalance.utils import get_header_format, get_tqdm_desc, prj_logger

from .general import FeatureExtractor

logger = prj_logger.getLogger(__name__)

import tqdm


def calc_gaussian_sim(profile):
    """
    Calculates the Gaussian similarity matrix based on the given profile.
    The Gaussian similarity matrix is calculated for rows of the profile matrix.

    Parameters:
    profile (numpy.ndarray): The profile matrix. profile shape should be (n, m)

    Returns:
    numpy.ndarray: The similarity matrix (For rows).
    """
    logger.info(get_header_format("Calculating Gaussian Similarity Matrix"))
    n = profile.shape[0]
    similarity_matrix = np.zeros((n, n))
    sigma = (pow(np.linalg.norm(profile), 2)) / n

    temp = profile @ profile.T
    for i in range(n):
        for j in range(i, n):
            similarity = np.exp(
                -(temp[i, i] + temp[j, j] - 2 * temp[i, j]) / (2 * (sigma**2))
            )
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    return similarity_matrix


class GIPFeatureExtractor(FeatureExtractor):

    def __init__(
        self,
        microbe_ids,
        disease_ids,
        imputer_strategy="constant",
        inputer_keep_empty_features=True,
        imputer_fill_value=0,
    ):
        super(GIPFeatureExtractor, self).__init__()

        self.microbe_ids = microbe_ids
        self.disease_ids = disease_ids

        self.imp = SimpleImputer(
            missing_values=np.nan,
            strategy=imputer_strategy,
            keep_empty_features=inputer_keep_empty_features,
            fill_value=imputer_fill_value,
        )

        self.profile = np.full(
            (len(self.microbe_ids), len(self.disease_ids)), np.nan
        )  # Shape (n_microbes, n_diseases)
        self.mask = None
        self.dis_gip_sim_matrix = None
        self.mic_gip_sim_matrix = None

    def build(self, associations):
        """Builds the KGNMDA Feature Extractor.

        Args:
            associations (np.ndarray): The associations. shape: (n_associations, 3). example: [disease, microbe, increased]
        """
        logger.info(get_header_format("Building GIP Feature Extractor"))

        for i in range(associations.shape[0]):
            m = self._mic_id_to_index(associations[i, 0])
            d = self._dis_id_to_index(associations[i, 1])
            a = associations[i, 2]
            self.profile[m][d] = a
        logger.info("The profile has been filled with associations.")

        self.mask = ~np.isnan(self.profile)
        logger.info("The nan elements stored in the mask.")

        self.imp.fit(self.profile)
        self.profile = self.imp.transform(self.profile)
        logger.info("The profile has been imputed to delete nans.")

        self.mic_gip_sim_matrix = calc_gaussian_sim(profile=self.profile)
        logger.info("The microbe gaussian similarity matrix has been calculated.")

        self.dis_gip_sim_matrix = calc_gaussian_sim(profile=self.profile.T)
        logger.info("The disease gaussian similarity matrix has been calculated.")

    def extract_features(self, mic_array, dis_array):
        mic_features = np.zeros((mic_array.shape[0], len(self.microbe_ids)))
        dis_features = np.zeros((dis_array.shape[0], len(self.disease_ids)))

        for i in range(len(mic_array)):
            mic_features[i] = self._get_mic_gip_features(mic_array[i])
            dis_features[i] = self._get_dis_gip_features(dis_array[i])

        return mic_features, dis_features

    def _mic_id_to_index(self, m):
        """Converts a microbe identifier to its corresponding index in the microbe_ids list.

        Parameters:
            m (str): The microbe identifier to be converted.

        Returns:
            int: The index of the microbe in the microbe_ids list.
        """
        return self.microbe_ids.index(m)

    def _dis_id_to_index(self, d):
        """Converts a disease idintifer to its corresponding index in the disease_ids list.

        Parameters:
            d (str): The disease identifier to be converted.

        Returns:
            int: The index of the disease in the disease_ids list.
        """
        return self.disease_ids.index(d)

    def _get_mic_gip_features(self, m):
        """Retrieves the microbe GIP features.

        Parameters:
            m (str): The microbe identifier.

        Returns:
            numpy.ndarray: The microbe similarity vector.
        """
        i = self._mic_id_to_index(m)
        return self.mic_gip_sim_matrix[i]

    def _get_dis_gip_features(self, d):
        """Retrieves the disease GIP features.

        Parameters:
            d (str): The disease identifier.

        Returns:
            numpy.ndarray: The disease similarity vector.
        """
        i = self._dis_id_to_index(d)
        return self.dis_gip_sim_matrix[i]

    def get_pair_gip_features(self, m, d):
        """Retrives GIP features for a pair of microbe and disease and combines them.

        Args:
            m (int): The microbe identifier.
            d (int): The disease identifier.

        Returns:
            np.ndarray: The combined GIP features. size: (n_microbes + n_diseases,)
        """
        return np.concatenate(
            (self._get_mic_gip_features(m), self._get_dis_gip_features(d)), axis=0
        )

    def get_all_pair_gip_features(self, mic_list: list, dis_list: list):
        """Retrieves GIP features for all pairs of microbe and disease.

        Args:
            mic_list (list): The list of microbe identifiers.
            dis_list (list): The list of disease identifiers.

        Returns:
            np.ndarray: The combined GIP features. shape: (n_associations, n_microbes + n_diseases)
        """
        if len(mic_list) != len(dis_list):
            raise ValueError("The length of mic_list and dis_list should be equal.")

        X = []
        for i in range(len(mic_list)):
            X.append(
                self.get_pair_gip_features(m=mic_list[i], d=dis_list[i]).reshape(1, -1)
            )
        return np.concatenate(X, axis=0)

    def get_microbe_kernel(self):
        """Returns the microbe kernel matrix.

        This method retrieves the microbe kernel matrix, which represents the similarity between different microbes.

        Returns:
            numpy.ndarray: The microbe kernel matrix. shape: (n_microbes, n_microbes)
        """
        return self.mic_gip_sim_matrix

    def get_disease_kernel(self):
        """
        Returns the disease kernel matrix.

        This method returns the disease kernel matrix, which represents the similarity between diseases based on the GIP (Gene Interaction Profile) similarity.

        Returns:
            numpy.ndarray: The disease kernel matrix. shape: (n_diseases, n_diseases)
        """
        return self.dis_gip_sim_matrix
