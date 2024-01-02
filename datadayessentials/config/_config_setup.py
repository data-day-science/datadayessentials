from datadayessentials.config._execution_environment_manager import (
    ExecutionEnvironmentManager,
    ExecutionEnvironment,
)
from datadayessentials.config._config import Config
from datadayessentials.utils import ConfigCacheWriter, ConfigCacheReader
import os


class ConfigSetup:
    @staticmethod
    def initialise_core_config(tenant_id: str, base_url: str):
        ConfigCacheWriter().add_key_value_to_config(key="tenant_id", value=tenant_id)
        ConfigCacheWriter().add_key_value_to_config(key="base_url", value=base_url)
