from typing import List, Union
import pandas as pd
from datadayessentials.modelling.models._base import IModel, IModelSavingLoadingAttribute
import mlflow


class SklearnModel(IModel, IModelSavingLoadingAttribute):
    categorical_features: List[str] = None
    def __init__(self, model: IModel, categorical_features: List[str] = None):
        self.model = model
        self.categorical_features = categorical_features
    
    def fit(self, X, y, *args, **kwargs):
        self.model.fit(X, y, *args, **kwargs)
        return self
    
    def predict(self, X, *args, **kwargs):
        return self.model.predict(X, *args, **kwargs)
    
    def predict_proba(self, X, *args, **kwargs):
        return self.model.predict_proba(X, *args, **kwargs)

    def _save_model_to_folder(
        self, model_save_path: str, input_example: object = None
    ):
        """Logs the model to the MLFlow server, if an input example is passed then infer the model signature.

        Args:
            mlflow (ModuleType): MLFlow module
            model_save_path (str): path in the mlflow file sytem to save the model files
            input_example (object, optional): model input example, used to infer the model signature for saving to mlflow
        Returns:
            ModelInfo: info object that contains the model URI and other details about the model
        """
        if input_example is not None:
            prediction = self.predict(input_example)
            signature = mlflow.models.signature.infer_signature(
                input_example, prediction
            )
            meta_data = mlflow.sklearn.save_model(
                self.model, model_save_path, signature=signature
            )
        else:
            meta_data = mlflow.sklearn.save_model(
                self.model, model_save_path, 
            )
        self.save_categorical_features(model_save_path)
        return meta_data
    
    def _save_categorical_features(self, save_path: str):
        with open(f"{save_path}/categorical_features.txt", "w") as f:
            f.write("\n".join(self.categorical_features))

    @staticmethod
    def _load_categorical_features(load_path: str):
        with open(f"{load_path}/categorical_features.txt", "r") as f:
            return f.read().split("\n")

    @classmethod
    def load_model_from_folder(cls, model_folder):
        model = mlflow.sklearn.load_model(model_folder)
        categorical_features = cls.load_categorical_features(model_folder)
        return cls(model=model, categorical_features=categorical_features)