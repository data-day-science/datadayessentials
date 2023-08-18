from datadayessentials.config._execution_environment_manager import ExecutionEnvironmentManager, ExecutionEnvironment
from datadayessentials.config._config import AzureConfigManager
import os

class ConfigSetup:
    @staticmethod
    def initialise_core_config(self, tenant_id=None,use_local_config=False):
        if tenant_id:
            os.environ["AZURE_TENANT_ID"] = tenant_id
        AzureConfigManager(use_local_config=use_local_config).set_default_config_variables()
        
