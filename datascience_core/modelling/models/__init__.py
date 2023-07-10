"""
This module contains two things:

1. Model wrappers, these are wrapped versions of the models that have additional functionality as specified by the IModelSavingLoadingAttribute interface. This interface is used to ensure that all models have the same functionality and can be used interchangeably. One example model wrapper is the CatBoostClassifierPipeline class.
2. ModelFactory, this is for loading models stored in the model registry or the experiment registry. Please see the ModelFactory class for an example use case.  
"""
from .catboost import CatBoostClassifierPipeline
from .xgboost import XGBoostClassifierPipeline
from .model_factory import ModelFactory
from .sklearn import SklearnModel

__all__ = [
    SklearnModel,
    XGBoostClassifierPipeline,
    CatBoostClassifierPipeline,
    ModelFactory
]
