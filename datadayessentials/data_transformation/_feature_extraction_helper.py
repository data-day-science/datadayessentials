from typing import List, Union, Tuple
from ._base import IFeatureExtractor


class FeatureExtractor(IFeatureExtractor):
    def __init__(self):
        pass

    def _get_available_categoricals(
        self, requested: list, available_features: list
    ) -> list:
        """checks if the categoricals requested are in the dataset

        Args:
            requested (list): requested list of categorical features
            available_features (list): list of columns in the dataset

        Returns:
            list: returns the categoricals available
        """
        return list(set(available_features).intersection(requested))

    def _get_available_features(
        self, requested: list, available_features: list
    ) -> list:
        """Checks if the requested features are in the dataset

        Args:
            requested (list): requested list of features
            available_features (list): available features in the dataset

        Returns:
            list: returns the intersection between requested and available features
        """
        if requested:
            return list(set(available_features).intersection(requested))
        else:
            return available_features

    def _remove_unwanted_features(
        self, requested: list, available_features: list
    ) -> list:
        return list(set(available_features) - set(requested))

    def _validate_request(
        self,
        dataset_features: list,
        unwanted_features: list,
        required_features: list,
        target: str,
    ):
        """Validates that run function will be successful.  Checks that the required and unwanted fields don't contradict each other
        and that the target field is in the dataset

        Args:
            dataset_features (list): list of features to be assessed
            unwanted_features (list): list of features to remove
            required_features (list): list of features to keep (exclusive list)
            target (str): dataset target column

        Raises:
            ValueError: raises if required and unwanted features are the same
            ValueError: raises if none of the required features are available in the dataset
            ValueError: raises if the target is not in the datset
            ValueError: raises if the target feature is not in the required feature list when used
            ValueError: raises if the target is in the unwanted features list
        """
        if list(set(unwanted_features).intersection(required_features)):
            raise ValueError(
                f"unwanted_columns and required_columns must be mutually exclusive. {set(unwanted_features).intersection(required_features)} found in both"
            )

        if required_features:
            if len(list(set(dataset_features).intersection(required_features))) == 0:
                raise ValueError(
                    f"None of the required columns are in the available columns passed in dataset_columns"
                )
        if target:
            if target not in dataset_features:
                raise ValueError(
                    "Target feature was not found in input dataset features"
                )

        if required_features and (target is not None):
            print(f"required features found {required_features}")
            if target not in required_features:
                raise ValueError(
                    "Target feature must be in 'required_features' when 'required_features' is used"
                )
        if unwanted_features and (target is not None):
            if target in unwanted_features:
                raise ValueError("Target can not be in 'unwanted_features'")

    def run(
        self,
        dataset_features: list,
        categorical_features: list,
        unwanted_features: list,
        required_features: list,
        target: str = None,
        verbose: bool = True,
    ) -> Tuple[list, list]:
        """Function to run the stages of column extraction prior to preprocessing

        Args:
            dataset_features (list): list of features to run the extractor on
            target (str): target column for the dataset
            categorical_features (list): list of known categorical features
            unwanted_features (list): list of features to remove in the final dataset
            required_features (list): list of features to use (use if list is smaller than list of features to remove)
            verbose (bool, optional): print statements for final dataset. Defaults to True.

        Returns:
            Tuple[list, list]: returns a list of all features in the output dataset and a list of the subset of all features that are categoricals
        """

        self._validate_request(
            dataset_features=dataset_features,
            unwanted_features=unwanted_features,
            required_features=required_features,
            target=target,
        )

        if verbose:
            print(
                f"Started with:\n{len(dataset_features)} total features\n{len(required_features)} required features\n{len(categorical_features)} categorical features\n{len(unwanted_features)} unwanted features\n{len([target])} target features\n----------"
            )

        # logic priority
        # 1. remove unwanted columns
        # 2. identify available features
        # 3. identify available categoricals

        dataset_features = self._remove_unwanted_features(
            requested=unwanted_features, available_features=dataset_features
        )
        dataset_features = self._get_available_features(
            requested=required_features, available_features=dataset_features
        )
        categorical_features = self._get_available_categoricals(
            requested=categorical_features, available_features=dataset_features
        )

        if verbose:
            app_features = [x for x in dataset_features if "App" in x]
            BSB_features = [x for x in dataset_features if "BSB" in x]
            QCB_features = [x for x in dataset_features if "QCB" in x]

            print(
                f"Finished with:\n{len(dataset_features)} total features\n{len(categorical_features)} categorical features\n{len(app_features)} App features\n{len(BSB_features)} BSB features\n{len(QCB_features)} QCB features\n{len([target])} target feature\n----------"
            )

        return dataset_features, categorical_features
