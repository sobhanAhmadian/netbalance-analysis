import abc


class FeatureExtractor(abc.ABC):
    """
    Abstract base class for feature extraction.
    """

    @abc.abstractmethod
    def build(self, **kwargs):
        """
        Build the feature extractor.

        Args:
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def extract_features(self):
        """
        Extract features from the data.

        Returns:
            The extracted features.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError
