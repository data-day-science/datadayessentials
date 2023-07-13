import unittest
from unittest.mock import patch, mock_open
from .._config import GlobalConfig, LocalConfig
import os

class TestGlobalConfig(unittest.TestCase):
    def test_read(self):
        # Test that the read function returns the correct values
        global_config = GlobalConfig()
        expected_config = {
            "local_cache_dir": "cache"
        }
        with patch("builtins.open", mock_open(read_data="local_cache_dir: cache")):
            config = global_config.read()
            self.assertEqual(config, expected_config)


 