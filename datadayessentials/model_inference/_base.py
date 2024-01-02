# add imports
from abc import ABC, abstractmethod
from typing import Union, Tuple, List
import pandas as pd
import numpy as np


class IInferenceModel(ABC):
    """Interface for inference model"""

    feature_names_: List[str] = None
    feature_importances_: List[float] = None
    model = None

    @abstractmethod
    def __init__(self, model):
        if not model.model:
            raise ValueError(
                "Model has not been fit.  Only a trained model should be passed to this class"
            )

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.array:
        """Predicts the class for the given input data

        Args:
            X (pd.DataFrame): model input data.  Should be processed and in the correct format for the model to accept

        Returns:
            np.array: array of predictions for the given input data
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.array:
        """Predicts the class probabilities for the given input data

        Args:
            X (pd.DataFrame): model input data.  Should be processed and in the correct format for the model to accept

        Returns:
            np.array: array of class probabilities for the given input data
        """
        pass
