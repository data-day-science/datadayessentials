"""The models package provides a few submodules and classes:

1. ExperimentManager class - Responsible for saving experiment runs to the experiment registry (Azure ml workspace)
2. ModelManager class - Responsible for downloading models from the model registry (Azure ml workspace), and for registering models to the model registry (from either experiments or local model files)
3. models package - Contains model wrappers and a ModelFactory for loading models from either the model registry or the experiment registry
"""
from .models import CatBoostClassifierPipeline, XGBoostClassifierPipeline, SklearnModel,ModelFactory
from .experiment_manager import ExperimentManager
from .utils import DataSplitter, ModelPipeline
from .model_manager import ModelManager
from .model_evaluator import ModelEvaluator

__all__ = [
    'ModelManager',
    'CatBoostClassifierPipeline',
    'DataSplitter',
    'ModelPipeline',
    'ModelEvaluator',
    'XGBoostClassifierPipeline',
    'ExperimentManager',
    'SklearnModel',
    'ModelFactory'
]
