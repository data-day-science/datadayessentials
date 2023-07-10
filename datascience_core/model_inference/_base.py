from abc import ABC, abstractmethod
from typing import Optional, List
import pandas as pd
from ..data_retrieval import DataFrameValidator
from datascience_core.authentications import IAuthentication
import enum
from azure.ai.ml import MLClient, Input
import logging
from pathlib import Path
from datascience_core.data_retrieval import (
    BlobLocation,
    DataLakeJsonSaver,
    DataLakeCSVSaver,
)
from datascience_core.data_retrieval import DataLakeDirectoryDeleter
from ..config import LocalConfig, GlobalConfig
from pandas.util import hash_pandas_object
import os
import hashlib
import numpy as np


logger = logging.getLogger(__name__)
FORMAT = "[(%(asctime)s):%(filename)s:%(lineno)s:%(funcName)s()] %(message)s"
logging.basicConfig(
    filename="example.log",
    level=logging.DEBUG,
    format=FORMAT,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger.setLevel(logging.DEBUG)


class Models(enum.Enum):
    SCORECARD = "scorecard"
    AUTO_ALLOCATION = "auto_allocation"
    PRIME_PREDICTIONS = "prime_predictions"


class ServiceHitterCacher:
    """
    Class for Caching the results from hitting the batch endpoint. They are cached by hashing the payloads that are sent to the batch endpoint. There is a seperate cache for each model name.

    For an example on how to use this class, please see the IServiceHitter below.
    """

    def __init__(self, model_versions: List[str]):
        """Instantiate the cacher, creating the cache directory if it doesnt exist

        Args:
            model_versions (List[str]): Model versions to save/retrieve from the cache
        """
        self.model_versions = model_versions
        home = str(Path.home())
        self.cache_dir = os.path.join(
            home, LocalConfig.get_local_cache_dir(), "service_hitter_cache"
        )
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def get_cached_results(self, payload: pd.DataFrame) -> pd.DataFrame:
        """If all the rows of the input payload have been processed and have results stored in the cache, then these are retrieved by the hash of the dataframe row

        Args:
            payload (pd.DataFrame): Payload that would be sent to the endpoint for scoring

        Raises:
            ValueError: Raised if any of the scores have not been cached previously

        Returns:
            pd.DataFrame: _description_
        """
        if not self.check_all_scores_cached(payload):
            raise ValueError(
                "All payloads to this function must be cached. Plase check that they have all been cached using the check_all_scores_cached function in this class"
            )

        cached_results = pd.DataFrame(index=payload["ApplicationId"])
        for model_version in self.model_versions:
            cache = self._load_model_cache(model_version)
            hash_values = self.hash_df(payload)
            cached_part = cache[cache["hash"].isin(hash_values["hash"])]
            cached_part.drop("hash", axis=1, inplace=True)
            cached_results = cached_results.merge(
                cached_part,
                how="outer",
                left_on="ApplicationId",
                right_on="ApplicationId",
            )
        # for col in cached_results.columns:
        #    cached_results[col] = pd.to_numeric(cached_results[col], errors='ignore')
        cached_results.drop_duplicates(inplace=True, ignore_index=True)
        return cached_results

    def check_all_scores_cached(self, payload: pd.DataFrame) -> bool:
        """
        Checks if all of the rows in the payload have cached scores for each of the models being requested.

        Args:
            payload (pd.DataFrame): Payload to check for scores in cache

        Returns:
            bool: True if all scores are available in the cache. False otherwise.
        """
        for model_version in self.model_versions:
            if self._check_model_cache_exists(model_version):
                if self._check_model_scores_cached(model_version, payload):
                    pass
                else:
                    logger.debug(f"Not all scores are cached for this payload")
                    return False
            else:
                logger.debug(f"Not all scores are cached for this payload")
                return False
        logger.debug(f"All scores are cached for this payload")
        return True

    def save_to_cache(self, payload: pd.DataFrame, model_predictions: pd.DataFrame):
        """Save scores (model_predictions) from multiple models to the cache using payload to generate the hash for the cache

        Args:
            payload (pd.DataFrame): Model payload
            model_predictions (pd.DataFrame): Model predictions for the payload
        """
        logger.debug(
            f"The columns of the model predictions are {model_predictions.columns}"
        )
        for model in self.model_versions:
            hashes = self.hash_df(payload)
            logger.debug(f"Some example hashes are {hashes.sample(5, replace=True)}")
            relevant_columns = ["ApplicationId"] + [
                col for col in model_predictions if model in col
            ]
            model_specific_df = model_predictions[relevant_columns]
            model_specific_df = model_specific_df.merge(
                hashes, how="left", left_on="ApplicationId", right_on="ApplicationId"
            )
            self._update_model_cache(model, model_specific_df)

    def _update_model_cache(self, model_version: str, model_specific_df: pd.DataFrame):
        """
        Updates a single models cache.

        Args:
            model_version (str): THe models version to cache (v1, v2 etc.)
            model_specific_df (pd.DataFrame): Dataframe containing a column 'hash' with the unique payload hash, and the columns with the scores to save
        """
        filepath = os.path.join(self.cache_dir, model_version + ".csv")
        if not os.path.exists(filepath):
            logger.debug(f"Creating new cache for model {model_version}")
            model_specific_df.to_csv(filepath, index_label="index")
        else:
            logger.debug(f"Updating existing cache for model {model_version}")
            previous_model_cache = pd.read_csv(filepath, index_col="index")
            previous_model_cache = previous_model_cache[
                ~previous_model_cache["hash"].isin(model_specific_df["hash"])
            ]
            new_model_cache = pd.concat([previous_model_cache, model_specific_df])
            new_model_cache.to_csv(filepath, index_label="index")

    def _load_model_cache(self, model_version: str) -> pd.DataFrame:
        """
        Loads in the entire dataframe of cached results for a specific model. The cached model file is a csv and has two rows, the first column is the hash of the payload (pandas row), and the second column is the model score.

        Args:
            model_version (str): Model version (v1, v2 etc.)

        Returns:
            pd.DataFrame: All model scores stored in the cache.
        """
        cache_path = os.path.join(self.cache_dir, model_version + ".csv")
        return pd.read_csv(cache_path, index_col="index")

    def _check_model_cache_exists(self, model_version: str) -> bool:
        """
        Checks if a specific model cache exists.

        Args:
            model_version (str): The model version to check for in the cache

        Returns:
            bool: True if there is a cache for the model_version, False otherwise
        """
        return os.path.exists(os.path.join(self.cache_dir, model_version + ".csv"))

    def _check_model_scores_cached(
        self, model_version: str, payload: pd.DataFrame
    ) -> bool:
        """
        Checks to see if all the scores in the requested payload are in the cache.

        Args:
            model_version (str): Model version (v1, v2 etc.)
            payload (pd.DataFrame): Payload to use for hashing

        Returns:
            bool: True if all requested scores are cached for the model_version
        """
        model_cache = self._load_model_cache(model_version)
        payload_hashes = self.hash_df(payload)
        return payload_hashes["hash"].isin(model_cache["hash"]).all()

    def hash_df(self, df: pd.DataFrame) -> pd.DataFrame:
        def row_hash(row):
            this_hash = hashlib.md5(
                "-".join([str(val) for val in row]).encode()
            ).hexdigest()
            return this_hash

        if "App.ApplicationId" in df.columns:
            df = df.rename({"App.ApplicationId": "ApplicationId"}, axis=1)
        print(df.columns)
        df = df.reindex(sorted(df.columns), axis=1)
        print(df.columns)
        hash_df = pd.DataFrame()
        if "App.ApplicationId" in df.columns:
            hash_df["ApplicationId"] = df["App.ApplicationId"]
        else:
            hash_df["ApplicationId"] = df["ApplicationId"]
        hash_df["hash"] = df.apply(row_hash, axis=1)
        return hash_df


class IServiceHitter(ABC):
    """
    Abstract base class for all batch endpoint service hitters. This class creates the connection to Azure and hits the requested endpoint
    """

    def __init__(
        self,
        model_versions: List[str],
        data_lake_authentication: IAuthentication,
    ):
        """Instantiate the service hitter, initialising connection to azure

        Args:
            model_versions (List[str]): Model versions to use for scoring
            data_lake_authentication (IAuthentication): Authentication for azure (see authentications module)
        """
        self.endpoint_reference = ""
        self.model_versions = model_versions
        self._load_config()

        self.ml_client = MLClient(
            data_lake_authentication.get_credentials(),
            self.subscription_id,
            self.resource_group,
            self.workspace,
        )

        self.data_lake_authentication = data_lake_authentication

    def hit(
        self, payload: pd.DataFrame, verbose: bool = False, use_cache: bool = True
    ) -> pd.DataFrame:
        """Hit the batch endpoint with the payload. Caches the results and uses the cache by default.

        Args:
            payload (pd.DataFrame): Payload to score
            verbose (bool, optional): Enables additional logging. Defaults to False.
            use_cache (bool, optional): Enables the cache. Defaults to True.

        Returns:
            pd.DataFrame: Scores for the payload for each model version specified in the __init__
        """
        # Retrieve cached results
        cacher = ServiceHitterCacher(self.model_versions)
        if use_cache:
            if cacher.check_all_scores_cached(payload):
                return cacher.get_cached_results(payload)
        # Save to local
        saved_file_location = self._save_files_to_location(
            payload,
            self.model_versions,
            self.account,
            self.container,
            self.filepath,
        )
        # Trigger Batch Job
        results = self._trigger_batch_job(
            self.ml_client,
            self.batch_endpoint,
            self.deployment_name,
            uri_folder=saved_file_location,
        )
        cacher.save_to_cache(payload, results)
        return results

    def _load_config(self):
        """Loads in config needed for the batch endpoint invocation"""
        logger.debug("loading configs for batch endpoint inference")
        batch_endpoint_settings = LocalConfig.get_batch_endpoint(self.endpoint_reference)
        env_settings = LocalConfig.get_environment()
        payload_storage = LocalConfig.get_data_lake_folder(batch_endpoint_settings["payload_storage"])
        self.subscription_id = env_settings[
            "subscription_id"
        ]
        self.resource_group = env_settings[
            "resource_group"
        ]
        self.workspace = env_settings["machine_learning_workspace"]
        self.account = payload_storage["data_lake"]
        self.container = payload_storage["container"]
        self.filepath = payload_storage["path"]
        self.inference_results_local_save_path = os.path.join(Path.home(), GlobalConfig().read()['local_cache_dir'], batch_endpoint_settings["inference_results_local_save"]["full_path"])
        self.batch_endpoint = batch_endpoint_settings[
            "endpoint_name"
        ]
        self.deployment_name = batch_endpoint_settings[
            "deployment_name"
        ]

    def _save_files_to_location(
        self,
        df: pd.DataFrame,
        models_to_invoke: List[str],
        account: str,
        container: str,
        filepath: str,
    ) -> Path:
        """Saves dataframe to Azure blob storage, along with meta information of the models that require storing.
        THe models to score are passed to the batch endpoint inside the dataframe with the column name 'models_to_use'

        Args:
            df (pd.DataFrame): Dataframe to save
            models_to_invoke (List[str]): Models to use for scoring using batch endpoint
            account (str): Azure storage account
            container (str): Azure storage account container
            filepath (str): Path inside container

        Returns:
            Path: Complete path of Azure Blob Folder containind payload and meta information for batch scoring
        """
        # Firstly delete the directory to ensure nothing is left from the last invocation
        blob_location_df_for_inference = BlobLocation(
            account, container, filepath, ""
        )
        directory_deleter = DataLakeDirectoryDeleter(self.data_lake_authentication)
        directory_deleter.delete_directory(blob_location_df_for_inference) 
        logger.debug("saving data csv")
        # Split the dataframe into chunks so that the batch endpoint can work on the data in parallel. Maximum of 1000 rows per file
        df['models_to_score'] = ','.join(models_to_invoke)
        df_chunks = np.array_split(df, (len(df) - 1) // 1000 + 1)
        for i, df_chunk in enumerate(df_chunks):
            csv_saver = DataLakeCSVSaver(self.data_lake_authentication)
            blob_location_df_for_inference = BlobLocation(
                account, container, filepath, f"data_for_scorecard_inference_chunk_{i}.csv"
            )
            csv_saver.save(blob_location_df_for_inference, df_chunk)

        last_slash_position = str(blob_location_df_for_inference).rfind("/")

        return str(blob_location_df_for_inference)[0:last_slash_position]

    @abstractmethod
    def _trigger_batch_job(
        self,
        ml_client: MLClient,
        endpoint_name: str,
        deployment_name: str,
        uri_folder: str,
    ):
        pass
