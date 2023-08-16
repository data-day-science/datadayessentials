"""
This module provides classes for managing the core local config file, including for synchronizing the config file with a remote version stored in a machine learning workspace.
"""
from ._config import Config
from ._execution_environment_manager import ExecutionEnvironmentManager


__all__ = [
    'Config',
    'ExecutionEnvironmentManager'
]
