# this should be scalable to include other models
# pass endpoint
# from _base import MLModel
import pandas as pd
from typing import Optional, List, Union
from azure.ai.ml import MLClient, Input
import pandas as pd
from pandas.io.parsers import TextFileReader
from ._base import IServiceHitter, Models
from azure import identity
from azure.identity import InteractiveBrowserCredential
import logging
import os
from datascience_core.authentications import IAuthentication
from datascience_core.data_retrieval import (
    BlobLocation,
    DataLakeJsonSaver,
    DataLakeCSVSaver,
    SchemaFetcher,
    DataLakeCSVLoader
)
from datascience_core.data_transformation import (
    DataFrameCaster,
)
from datascience_core.data_retrieval import ICSVSaver, IBlobLocation
import tempfile
import shutil
import json
from pathlib import Path, PurePosixPath
import time
from ..config import LocalConfig
from azure.ai.ml.entities import AzureDataLakeGen2Datastore, AzureBlobDatastore, BatchJob 


logger = logging.getLogger(__name__)
FORMAT = "[(%(asctime)s):%(filename)s:%(lineno)s:%(funcName)s()] %(message)s"
logging.basicConfig(
    filename="example.log",
    level=logging.DEBUG,
    format=FORMAT,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger.setLevel(logging.DEBUG)

ENDPOINTS = {"scorecard": ""}


class GenericServiceHitter(IServiceHitter):
    """
    Generic Service hitter for the prime predictions and scorecard endpoints as they have shared functionality
    """
    def __init__(
        self,
        model_versions: List[str],
        data_lake_authentication: IAuthentication,
    ):
        self.model_versions = model_versions
        self._load_config()

        self.ml_client = MLClient(
            data_lake_authentication.get_credentials(),
            self.subscription_id,
            self.resource_group,
            self.workspace,
        )

        self.data_lake_authentication = data_lake_authentication

    def _trigger_batch_job(
        self,
        ml_client: MLClient,
        endpoint_name: str,
        deployment_name: str,
        uri_folder: str,
    ) -> Union[pd.DataFrame, TextFileReader]:
        """Start the batch endpoint execution after the file has been uploaded to blob storage.

        Args:
            ml_client (MLClient): MLClient instance for managing azure connection
            endpoint_name (str): Name of endpoint to invoke
            deployment_name (str): Name of specific endpoint deployment
            uri_folder (str): Folder containing the paylaods for scoring

        Returns:
            Union[pd.DataFrame, TextFileReader]: _description_
        """
        store = AzureBlobDatastore(
                name="batch_deployment_datastore",
                description="Datastore for all batch deployment outputs",
                account_name=self.account,
                container_name=self.container,
            )
        
        # Create or retrieve a datastore to access the output of the batch endpoint
        # There is limited documentation available in azure for this, but after attempting to access
        # the outputs of the batch endpoint directly, we settled on this approach
        try:
            created_store = ml_client.datastores.get(store.name)
        except Exception as e:
            created_store = ml_client.datastores.create_or_update(store)   

        job = ml_client.batch_endpoints.invoke(
            endpoint_name=endpoint_name,
            deployment_name=deployment_name,
            inputs={"input": Input(path=uri_folder, type="uri_folder")},
                params_override=[
                    { "output_dataset.datastore_id": f"azureml:{created_store.id}" },
                    { "output_dataset.path": f"/batch-endpoint-output/{self.batch_endpoint}" },
                    { "output_file_name": "predictions.csv" },
                ]
            )

        self.wait_for_batch_completion(ml_client, job)

        model_names_for_results = []
        for version in self.model_versions:
            model_names_for_results.append(f"{version}.raw_score")
            model_names_for_results.append(f"{version}.mapped_score")

        csv_loader = DataLakeCSVLoader(authentication=self.data_lake_authentication)
        df_results = csv_loader.load(
                        BlobLocation(
                            account=self.account,
                            container=self.container,
                            filepath=f"batch-endpoint-output/{self.batch_endpoint}",
                            filename="predictions.csv"
                        ),
                        delimiter=" ",
                        names=["ApplicationId", *model_names_for_results],
                        index_col=False,
                    )
        return df_results

    def wait_for_batch_completion(self, ml_client: MLClient, job: BatchJob):
        """Wait for the batch endpoint to complete
        """
        current = time.time()
        interval = time.time() - current
        while interval <= 6000:
            if ml_client.jobs.get(job.name).status == "Completed":
                return
            if ml_client.jobs.get(job.name).status == "Failed":
                raise (Exception("batch inference failed"))
            time.sleep(10)
            interval = time.time() - current
        raise Exception("batch inference timed out")

    def hit(self, payload: pd.DataFrame, verbose=False, use_cache=True) -> pd.DataFrame:
        """Hit the scorecard batch endpoint with the payload given
        Args:
            payload (pd.DataFrame): Payload following the scorecard schema
            verbose (bool, optional): Enables additional logging. Defaults to False.
            use_cache (bool, optional): Uses cache by default, payloads are hashed to access the caching. Defaults to True.

        Returns:
            pd.DataFrame: Scorecard scores for the models specified in the __init__.
        """
        # So that we guarentee the column heading in the payload
        if (
            "App.ApplicationId" in payload.columns
            and "ApplicationId" not in payload.columns
        ):
            logger.debug("Renaming the App.ApplicationId column to ApplicationId")
            payload = payload.rename({"App.ApplicationId": "ApplicationId"}, axis=1)
        return super().hit(payload, verbose, use_cache)

class ScorecardServiceHitter(GenericServiceHitter):
    """Used for using the scorecard service batch endpoint. Can send a dataframe containing payloads and return scores from multiple models.
    Example Use Case:
    ```python
    from datascience_core.model_inference import ScorecardBatchEndpoint
    from datascience_core.data_retrieval import CreditDataLoader
    from datascience_core.authentications import DataLakeAuthentication

    # Range of credit payloads to retrieve and the models to score with the batch endpoint
    start_date = datetime.strptime('2022-01-01', '%Y-%m-%d')
    end_date = datetime.strptime('2022-02-01', '%Y-%m-%d')
    models = ['AD3_v1', 'AD3_v2']

    authentication = DataLakeAuthentication()
    loader = CreditDataLoader(authentication, start_date, end_date)

    payload = loader.load()

    service_hitter = ScorecardServiceHitter(models, authentication)

    scores = service_hitter(payload)
    ```

    """

    def __init__(
        self,
        model_versions: List[str],
        data_lake_authentication: IAuthentication,
    ):
        """Instantiate a Scorecard Service hitter

        Args:
            model_versions (List[str]): The models to use for scoring. Each model will return an additional results column with both the raw ans mapped score.
            data_lake_authentication (IAuthentication): Authentication for azure (see authentications module)
        """
        self.endpoint_reference = "scorecard_batch_endpoint_hit"
        super().__init__(model_versions, data_lake_authentication)
        
class PrimePredictionsServiceHitter(GenericServiceHitter):
    """Used for using the prime predictions batch endpoint service. Can send a dataframe containing payloads and return scores from multiple models.
    Example Use Case:
    ```python
    from datascience_core.model_inference import PrimePredictionsServiceHitter
    from datascience_core.data_retrieval import CreditDataLoader
    from datascience_core.authentications import DataLakeAuthentication

    # Range of credit payloads to retrieve and the models to score with the batch endpoint
    start_date = datetime.strptime('2022-01-01', '%Y-%m-%d')
    end_date = datetime.strptime('2022-02-01', '%Y-%m-%d')
    models = ['PP4_v1', 'PP4_v2']

    authentication = DataLakeAuthentication()
    loader = CreditDataLoader(authentication, start_date, end_date)

    payload = loader.load()

    service_hitter = PrimePredictionsServiceHitter(models, authentication)

    scores = service_hitter(payload)
    ```

    """

    def __init__(
        self,
        model_versions: List[str],
        data_lake_authentication: IAuthentication,
    ):
        """Instantiate a Scorecard Service hitter

        Args:
            model_versions (List[str]): The models to use for scoring. Each model will return an additional results column with both the raw ans mapped score.
            data_lake_authentication (IAuthentication): Authentication for azure (see authentications module)
        """
        self.endpoint_reference = "prime_predictions_batch_endpoint"
        super().__init__(model_versions, data_lake_authentication)

class AffordabilityServiceHitter(IServiceHitter):
    """
    For hitting the Affordability service batch endpoint.
    The functionality for this service is the same as the ScorecardServiceHitter above.
    """

    def __init__(
        self,
        data_lake_authentication: IAuthentication,
    ):
        self.endpoint_config_name = "affordability_batch_endpoint_hit"
        self.model_versions = "no_specified_version"
        self._load_config()

        self.ml_client = MLClient(
            data_lake_authentication.get_credentials(),
            self.subscription_id,
            self.resource_group,
            self.workspace,
        )

        self.data_lake_authentication = data_lake_authentication

    def _trigger_batch_job(
        self,
        ml_client: MLClient,
        endpoint_name: str,
        deployment_name: str,
        uri_folder: str,
    ) -> pd.DataFrame:
        """Start the batch endpoint execution after the payload file has been uploaded to blob storage.

        Args:
            ml_client (MLClient): MLClient instance for managing azure connection
            endpoint_name (str): Name of endpoint to invoke
            deployment_name (str): Name of specific endpoint deployment
            uri_folder (str): Folder containing the paylaods for scoring

        Returns:
            Union[pd.DataFrame, TextFileReader]: _description_
        """
        job = ml_client.batch_endpoints.invoke(
            endpoint_name=endpoint_name,
            deployment_name=deployment_name,
            inputs={"input": Input(path=uri_folder, type="uri_folder")},
        )

        current = time.time()
        interval = time.time() - current

        while interval <= 600:
            if ml_client.jobs.get(job.name).status == "Completed":
                ml_client.jobs.download(
                    name=job.name,
                    output_name="score",
                    download_path=self.inference_results_local_save_path,
                )
                break
            if ml_client.jobs.get(job.name).status == "Failed":
                raise (Exception("batch inference failed"))
            time.sleep(10)
            interval = time.time() - current

        column_names = [
            "service",
            "applicationId",
            "MaxMonthlyPayment",
            "MaxMonthlyPaymentIncludingHP",
            "status",
            "WARNINGS",
            "ERROR",
            "PostCodeUsed",
            "Age",
            "Marital Status",
            "Residential Status",
            "IncomeMultiplier",
            "RentDetermined",
            "RentDeterminedUnadjusted",
            "NetMonthlyIncome",
            "CostOfLivingTotalWeighted",
            "CostOfLivingTotalUnadjusted",
            "Surplus",
            "TotalCreditPayments",
        ]

        pick_columns = ["applicationId", "MaxMonthlyPaymentIncludingHP"]

        df_results = pd.read_csv(
            self.inference_results_local_save_path + "/predictions.csv",
            delimiter=" ",
            names=column_names,
            index_col=False,
        )[pick_columns]

        df_results["AFY_check"] = float(df_results["MaxMonthlyPaymentIncludingHP"]) > 0
        df_results.rename(columns={"applicationId": "ApplicationId"}, inplace=True)
        return df_results
