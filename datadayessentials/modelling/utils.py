from typing import Union, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


from pathlib import Path
import os

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics

from datadayessentials.data_transformation._base import IPreProcessor
from datadayessentials.modelling._base import IExperimentManager

# from datadayessentials.modelling.experiment_manager import ExperimentManager

from datadayessentials.config import Config
from azureml.core.authentication import (
    ServicePrincipalAuthentication,
    InteractiveLoginAuthentication,
)
from azureml.core import Workspace
import sys



def get_workspace() -> Workspace:
    """
    Initialises the Azure ML workspace
    """

    try:
        auth = ServicePrincipalAuthentication(
            tenant_id=Config.get_environment_variable("AZURE_TENANT_ID"),
            service_principal_id=Config.get_environment_variable("AZURE_CLIENT_ID"),
            service_principal_password=Config.get_environment_variable("AZURE_CLIENT_SECRET"),
        )
    except KeyError:
        # Only try this locally
        if sys.platform == "win32":
            auth = InteractiveLoginAuthentication()
        else:
            raise KeyError(
                "You must set the following environment variables: AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET"
            )

    
    return Workspace.get(
        name=Config().get_environment_variable("machine_learning_workspace"),
        subscription_id=Config().get_environment_variable("subscription_id"),
        resource_group=Config().get_environment_variable("resource_group"),
        auth=auth,
    )

            
 


class DataSplitter:
    """Class to train test split data including a train, validation and test set.  The
    class will also handle rebalancing a dataset (upsampling minority class)
    """

    def __init__(
        self,
    ):
        pass

    def _subtract_df_from_df(
        self, remove_these_list: list, from_this_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Function to subtract all rows of one df from another.  Used for train test split

        Args:
            remove_these_list (list): list of dataframes to subtract
            from_this_df (pd.DataFrame): dataframe from which to subtract rows

        Returns:
        pd.DataFrame: returns the reduced dataframe
        """

        ls_sub = []
        for each_df in remove_these_list:
            ls_sub += each_df.index.to_list()

        ls_main = from_this_df.index.to_list()
        ls_out = list(set(ls_main) - set(ls_sub))

        return from_this_df.iloc[ls_out]

    def train_val_test_split(
        self,
        data: pd.DataFrame,
        target_list: list,
        validation_fraction: float,
        num_test_rows: int,
        validation_is_holdout: bool = False,
        verbose=True,
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        """Classic train test split (uses sklearn under the hood), but pops a hold out test set first

        Args:
            data (pd.DataFrame): features and target
            target (list): name of the target columns in a list
            validation_fraction (float): percentage of data for validation set (after removing test set) or number of rows to use for validation
            num_test_rows (int): number of hold out test rows
            validation_is_holdout (bool): use a holdout in time set for validation
            verbose (bool, optional): print statements for rows,cols in each set. Defaults to True.

        Returns:
            Tuple[ pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series ]: X_train,X_val,X_test, y_train, y_val,y_test
        """

        # separate a hold out test set

        if data.shape[1] < 2:
            raise ValueError(
                "Data contains only 1 feature. Splitter needs features and a target column"
            )
        for target in target_list:
            if target not in data.columns:
                raise ValueError(f"Target column '{target}' not found in the data")

        if not validation_fraction or validation_fraction == 0:
            validation_fraction = 0

        if not num_test_rows or num_test_rows == 0:
            num_test_rows = 0
            X_test = pd.DataFrame()
            y_test = pd.Series()
        else:
            X_test = data.tail(num_test_rows)
            y_test = X_test[target_list]
            X_test.drop(target_list, axis=1, inplace=True)

        if validation_fraction < 1:
            validation_rows = int((data.shape[0] - num_test_rows) * validation_fraction)
        else:
            validation_rows = int(validation_fraction)

        if validation_is_holdout:
            if validation_rows > 0:
                X_val = (
                    data.iloc[:-num_test_rows].tail(validation_rows)
                    if X_test.shape[0] != 0
                    else data.tail(validation_rows)
                )
                y_val = X_val.pop(target)
            else:
                X_val = pd.DataFrame()
                y_val = pd.Series()

            X_train = self._subtract_df_from_df(
                remove_these_list=[X_val, X_test], from_this_df=data
            )
            y_train = X_train[target_list]
            X_train.drop(target_list, inplace=True, axis=1)

        else:
            X = self._subtract_df_from_df(remove_these_list=[X_test], from_this_df=data)
            y = X[target_list]
            X.drop(target_list, inplace=True, axis=1)

            if validation_rows > 0:
                # conventional train test split

                X_train, X_val, y_train, y_val = train_test_split(
                    X,
                    y,
                    test_size=validation_fraction,
                    random_state=42,
                    stratify=y,
                )
            else:
                X_train = X
                y_train = y
                X_val = pd.DataFrame()
                y_val = pd.Series()

        if verbose:
            print(f"X_train set is: {X_train.shape}")
            print(f"target is in X_train set: {target in X_train.columns}")
            print(f"y_train set is {y_train.shape}\n")

            print(f"X_val set is: {X_val.shape}")
            print(f"target is in X_val set: {target in X_val.columns}")
            print(f"y_val set is {y_val.shape}\n")

            print(f"X_test set is: {X_test.shape}")
            print(f"target is in X_test set: {target in X_test.columns}")
            print(f"y_test set is {y_test.shape}\n")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def balance_dataset(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        total_samples: float = None,
        majority_class_fraction: float = 0.5,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Resamples from dataset passed.  The function will calculate how many majority samples are needed based on total
        samples and majority class fraction, then calculate remaining samples for minority class

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Truth label
            total_samples (float, optional): number of rows required. Defaults to None.
            majority_class_fraction (float, optional): percentage of data for majority class. Defaults to 0.5.

        Raises:
            ValueError: If sample number isnt passed, will raise an error

        Returns:
            Tuple(pd.DataFrame, pd_series]: Features and Truth label rebalanced
        """
        if not total_samples:
            raise ValueError("Total sample number is required to rebalance data")

        X[y.name] = y
        target = y.name

        majority_samples = int(np.round(total_samples * majority_class_fraction))
        minority_samples = int(total_samples - majority_samples)

        df_resampled = self._resample_data(
            data=X,
            minority_samples=minority_samples,
            majority_samples=majority_samples,
            target=target,
        )
        X = df_resampled
        y = X.pop(target)
        return X, y

    def _resample_data(
        self,
        data: pd.DataFrame,
        minority_samples: int = 100000,
        majority_samples: int = 100000,
        target: str = None,
    ) -> pd.DataFrame:
        if target not in data.columns:
            raise ValueError("Target column not found in the data")

        if minority_samples == 0 or majority_samples == 0:
            raise ValueError(
                "Both minority and majority samples have to be greater than 0"
            )

        majority_class = int(data[target].value_counts().index[0])
        minority_class = int(1 - majority_class)

        majority_df_source = data[data["Target"] == majority_class]
        minority_df_source = data[data["Target"] == minority_class]

        if minority_samples < minority_df_source.shape[0]:
            df_minority = minority_df_source.sample(
                n=int(minority_samples), replace=False
            )
        else:
            df_minority = minority_df_source.sample(
                n=int(minority_samples), replace=True
            )
        if majority_samples < majority_df_source.shape[0]:
            df_majority = majority_df_source.sample(
                n=int(majority_samples), replace=False
            )
        else:
            df_majority = majority_df_source.sample(
                n=int(majority_samples), replace=True
            )

        return pd.concat([df_minority, df_majority], axis=0).sort_index()

    def split_from_split_dict(
        self, data: pd.DataFrame, split_dict: dict, target=None
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        if not target:
            raise ValueError("No target column provided")

        if "train" in split_dict.keys():
            X_train = data.iloc[split_dict["train"]]
            y_train = X_train.pop(target)
        else:
            X_train = pd.DataFrame()
            y_train = pd.Series()

        if "validate" in split_dict.keys():
            X_val = data.iloc[split_dict["validate"]]
            y_val = X_train.pop(target)
        else:
            X_val = pd.DataFrame()
            y_val = pd.Series()

        if "test" in split_dict.keys():
            X_test = data.iloc[split_dict["test"]]
            y_test = X_train.pop(target)
        else:
            X_test = pd.DataFrame()
            y_test = pd.Series()

        return X_train, X_val, X_test, y_train, y_val, y_test


class ModelPipeline:
    def __init__(
        self,
        experiment_manager: IExperimentManager,
        data_splitter: DataSplitter,
        pre_processor: IPreProcessor,
        model,
    ):
        """Orchestration class to run a modelling training run

        Args:
            experiment_manager (ExperimentManager): Experiment Manager to handle submitting experiments and models to azure
            data_splitter (DataSplitter): DS data splitter for train test splits and resampling imbalanced data
            pre_processor (IPreProcessor): preprocessor object holding all preprocessing steps
            model (Any): Machine learning model object
        """
        self.experiment_manager = experiment_manager
        self.data_splitter = data_splitter
        self.pre_processor = pre_processor
        self.model = model

    def train_val_test_split_data(
        self,
        data,
        target_list,
        validation_fraction,
        num_test_rows,
        balance_data=False,
        total_samples=None,
        majority_class_fraction=None,
        verbose=True,
    ):
        """Classic train test split (uses sklearn under the hood), but pops a hold out test set first

        Args:
            data (pd.DataFrame): features and target
            target (str): name of the target column
            validation_fraction (float): percentage of data for validation set (after removing test set)
            num_test_rows (int): number of hold out test rows
            verbose (bool, optional): print statements for rows,cols in each set. Defaults to True.

        Returns:
            Tuple[ pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series ]: X_train,X_val,X_test, y_train, y_val,y_test
        """
        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        ) = self.data_splitter.train_val_test_split(
            data=data,
            target_list=target_list,
            validation_fraction=validation_fraction,
            num_test_rows=num_test_rows,
            verbose=verbose,
        )
        if balance_data:
            X_train, y_train = self.data_splitter.balance_dataset(
                X_train,
                y_train,
                majority_class_fraction=majority_class_fraction,
                total_samples=total_samples - X_val.shape[0],
            )
            X_val, y_val = self.data_splitter.balance_dataset(
                X_val,
                y_val,
                majority_class_fraction=majority_class_fraction,
                total_samples=X_val.shape[0],
            )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def evaluate_model(
        self,
        X_train: pd.DataFrame,
        y_train: Union[pd.Series, list],
        X_val: pd.DataFrame,
        y_val: Union[pd.Series, list],
        X_test: pd.DataFrame,
        y_test: Union[pd.Series, list],
        verbose: bool = True,
        artifacts_path="artifacts",
        **kwargs,
    ):
        """Runs the model evaluation class.  This will produce a dictionary of model
        evaluation metrics

        Args:
            X_train (pd.DataFrame): Features
            y_train (Union[pd.Series,list]): Truth labels
            X_val (pd.DataFrame): Features
            y_val (Union[pd.Series,list]): Truth labels
            X_test (pd.DataFrame): Features
            y_test (Union[pd.Series,list]): Truth labels
            cv (int, optional): number of folds for cross validation. Defaults to None.
            verbose (bool, optional): print outputs. Defaults to True.

        Returns:
            dict: model evaluation metrics
        """

        outputs_dict = {"tags": None, "params": None, "metrics": {}, "artifacts": {}}

        if hasattr(self.model, "get_all_params"):
            outputs_dict["params"] = self.model.get_all_params()

        Path(artifacts_path).mkdir(parents=True, exist_ok=True)

        training_outputs = self.model.evaluate_model(X_train, y_train, verbose=verbose)
        print("Training Metrics ^^^^^")

        validation_outputs = self.model.evaluate_model(X_val, y_val, verbose=verbose)
        print("Validation Metrics ^^^^^")

        test_outputs = self.model.evaluate_model(X_test, y_test, verbose=verbose)
        print("Test Metrics ^^^^^")

        outputs_dict["metrics"]["train"] = training_outputs["model_metrics"]
        outputs_dict["metrics"]["validate"] = validation_outputs["model_metrics"]
        outputs_dict["metrics"]["test"] = test_outputs["model_metrics"]

        training_outputs["model_performance_figure"].savefig(
            os.path.join(artifacts_path, "train_performance.png")
        )
        validation_outputs["model_performance_figure"].savefig(
            os.path.join(artifacts_path, "validation_performance.png")
        )
        test_outputs["model_performance_figure"].savefig(
            os.path.join(artifacts_path, "test_performance.png")
        )
        outputs_dict["artifacts"]["train_model_performance"] = os.path.join(
            artifacts_path, "train_performance.png"
        )
        outputs_dict["artifacts"]["validate_model_performance"] = os.path.join(
            artifacts_path, "validation_performance.png"
        )
        outputs_dict["artifacts"]["test_model_performance"] = os.path.join(
            artifacts_path, "test_performance.png"
        )

        if hasattr(self.model, "plot_feature_importance"):
            feature_importance = self.model.plot_feature_importance(X_train, y_train)
            feature_importance.savefig(
                os.path.join(artifacts_path, "feature_importance.png")
            )
            outputs_dict["artifacts"]["feature_importance"] = os.path.join(
                artifacts_path, "feature_importance.png"
            )

        permutation_importance = self.model.plot_permutation_importance(
            X_val, y_val, verbose=verbose
        )
        if hasattr(self.model, "plot_and_save_history"):
            metric_figures = self.model.plot_and_save_history()
            print(metric_figures)
            outputs_dict["artifacts"] = outputs_dict["artifacts"].update(metric_figures)

        if permutation_importance:
            permutation_importance.savefig(
                os.path.join(artifacts_path, "permutation_importance.png")
            )
            outputs_dict["artifacts"]["permutation_importance"] = os.path.join(
                artifacts_path, "permutation_importance.png"
            )

        if "fold_count" in kwargs:
            check_folds = kwargs["fold_count"]
            if check_folds > 0:
                fig = self.model.run_cross_validation(
                    X_train,
                    y_train,
                    **kwargs,
                )
            else:
                raise ValueError("fold count needs to be greater than 0 when used")
            fig.savefig(os.path.join(artifacts_path, "CrossValidationPerformance.png"))
            outputs_dict["artifacts"][
                "cross_validation"
            ] = "CrossValidationPerformance.png"

        return outputs_dict

    def initialise_ml_flow_run(self, run_name: str, description: str):
        """Set the run name and description for the training run

        Args:
            run_name (str): name
            description (str): description
        """
        self.experiment_manager.set_run(run_name, description)

    def submit_run_to_mlflow(self, data, versioning_meta_data: dict, model=None):
        return self.experiment_manager.submit_run(data, versioning_meta_data, model)

    def get_summary(self):
        pass
