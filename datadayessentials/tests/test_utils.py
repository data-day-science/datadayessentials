import pathlib
import unittest
from pathlib import Path
from unittest.mock import patch


class TestCoreCacheManager(unittest.TestCase):

    @patch("pathlib.Path.home", return_value=Path("C:/Users/test_user"))
    def test_cache_directory(self, mock_pathlib):
        assert mock_pathlib() == Path("C:/Users/test_user")
        assert pathlib.Path.home() == Path("C:/Users/test_user")
