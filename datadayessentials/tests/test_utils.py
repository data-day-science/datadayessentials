import pathlib
import unittest
from pathlib import Path
from unittest.mock import patch
import yaml

import datadayessentials


class TestCoreCacheManager(unittest.TestCase):

    @patch("pathlib.Path.home", return_value=Path("C:/Users/test_user"))
    def test_cache_directory(self, mock_pathlib):
        assert mock_pathlib() == Path("C:/Users/test_user")
        assert pathlib.Path.home() == Path("C:/Users/test_user")

    def test_core_cache_manager_init(self):
        """
        Create a cache file in the user's .core_cache directory.
        """
        cache_manager = datadayessentials.utils.CoreCacheManager()
        expected_cache_directory = Path(pathlib.Path.home() / ".core_cache")
        self.assertEqual(cache_manager.cache_directory, expected_cache_directory)

    def test_create_core_cache_directory(self):
        cache_manager = datadayessentials.utils.CoreCacheManager()
        cache_manager.create_core_cache_directory()
        self.assertTrue(cache_manager.cache_directory.exists())
        self.assertTrue(cache_manager.cache_directory.is_dir())

    @patch("datadayessentials.utils.ConfigCacheReader.get_value_from_config", return_value="key_value")
    def test_get_value_from_config(self, mocked_get_value_from_config):
        cache_manager = datadayessentials.utils.CoreCacheManager()
        self.assertEqual(cache_manager.get_value_from_config("test_key"), "key_value")


class TestConfigCacheWriter(unittest.TestCase):

    def test_add_key_value_to_config(self):
        cache_writer = datadayessentials.utils.ConfigCacheWriter()
        cache_writer.add_key_value_to_config("test_key", "test_value")
        self.assertTrue(cache_writer.config_path.exists())
        self.assertTrue(cache_writer.config_path.is_file())

        with open(cache_writer.config_path, "r") as yaml_file:
            existing_data = yaml.safe_load(yaml_file)

        self.assertIn("test_key", existing_data)
        self.assertEqual(existing_data["test_key"], "test_value")

        # Clean up
        existing_data.pop("test_key")
        with open(cache_writer.config_path, "w") as yaml_file:
            yaml.dump(existing_data, yaml_file, default_flow_style=False)


class TestConfigReader(unittest.TestCase):

    def test_get_value_from_config(self):
        cache_writer = datadayessentials.utils.ConfigCacheWriter()
        cache_writer.add_key_value_to_config("test_key", "test_value")

        cache_reader = datadayessentials.utils.ConfigCacheReader()
        value = cache_reader.get_value_from_config("test_key")
        self.assertEqual(value, "test_value")

        # Clean up
        with open(cache_writer.config_path, "r") as yaml_file:
            existing_data = yaml.safe_load(yaml_file)

        existing_data.pop("test_key")
        with open(cache_writer.config_path, "w") as yaml_file:
            yaml.dump(existing_data, yaml_file, default_flow_style=False)
