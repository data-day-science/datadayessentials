from datadayessentials.config._execution_environment_manager import ExecutionEnvironmentManager, ExecutionEnvironment
from datadayessentials.config._config import Config
from datadayessentials.utils import ConfigCacheWriter
import os

class ConfigSetup:
    @staticmethod
    def initialise_core_config(tenant_id=None,use_local_config=False):
        if tenant_id:
            ConfigCacheWriter().add_key_value_to_config(key = 'tenant_id', value = tenant_id)
        Config(use_local_config=use_local_config).set_default_variables()
        
