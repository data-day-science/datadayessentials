from ._base import IConfigManager
from ._config import LocalConfig
from datascience_core.data_retrieval import ProjectDatasetManager
from typing import Optional


class ConfigManager(IConfigManager):
    """Manages the synchronisation of the local config with the latest version in azure"""

    local_config = None
    dataset_manager = None

    def __init__(self):
        self.dataset_manager = ProjectDatasetManager("datascience_core")
        """manager class used to pull and push the local config file to azure as a registered dataset.
        """

    def pull_config(self, version: Optional[int] = None) -> None:
        """Pulls the datascience specific config file saved in azure as a registered dataset, and store it as the local config.

        Args:
            version (Optional[int]): Config version to pull down
        """

        if version:
            configs = self.dataset_manager.load_datasets(versions={'local_config': version})
        else:
            configs = self.dataset_manager.load_datasets()

        LocalConfig().write(configs["local_config"])

    def register_new_config(self):
        """Pushes a modified version of the local config to the azure registered dataset, and increments the version."""
        new_config = LocalConfig().read()
        self.dataset_manager.register_dataset(
            registered_dataset_name="local_config", data=new_config
        )
