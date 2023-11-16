from ._base import IInferenceModel

import pandas as pd
import numpy as np


class lightgbmInferenceModel(IInferenceModel):
    """
    Class for inference model
    """

    def __init__(self, model):
        self.model = model.model
        self.feature_names_ = model.feature_names_
        self.feature_importances_ = model.feature_importances_

    def predict(self, X: pd.DataFrame) -> np.array:
        """Predicts the class for the given input data

        Args:
            X (pd.DataFrame): model input data.  Should be processed and in the correct format for the model to accept

        Returns:
            np.array: array of predictions for the given input data
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.array:
        """Predicts the class probabilities for the given input data

        Args:
            X (pd.DataFrame): model input data.  Should be processed and in the correct format for the model to accept

        Returns:
            np.array: array of class probabilities for the given input data
        """
        return self.model.predict_proba(X)




class InferenceModel(IInferenceModel):
    """
    Class for inference model
    """

    def __init__(self, model):
        
        if not model.model:
            raise ValueError("Model has not been fit.  Only a trained model should be passed to this class")
        
        self.model = model.model
        