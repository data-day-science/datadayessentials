"""
# Objective
Data Science Core is a python package containing tools for accessing data, model inference and any reusable code that we will use in our other packages or services. 

# Purpose
To provide useful functionality for data scientists to use in business tools and data investigation/model development in jupyter notebooks. The purpose of this documentation is so that new staff/users can understand how to contribute to datadayessentials and how to use datadayessentials in any new projects.

# Modules

- authentications: Module containing authentication class for logging into azure (locally and with service principle) and logging into an SQL server
- data_retrieval: Module containing classes for loading, validating and saving pandas dataframes (to and from azure, from SQL Server)
- data_transformation: Module for transforming data (input pandas dataframe, output pandas dataframe)
- model_inference: Module for hitting batch endpoints hosted in Azure (Affordability and Scorecard models)
- models: Wrapper classes for Sklearn modules to add additional functionality
- utils: Small snippets or useful code & tools for jupyter notebooks

# Installation Guide

There are three options:

1. Clone the repository to your local computer.  From there, cd to the root of datadayessentials and activate the virtual env of interest
```bash
cd C:\\Users.....datadayessentials
conda activate <MY_ENV>
pip install .
```
2. Activate the virtual env of interest and pip install from the repo
```bash 
conda activate <MY_ENV>
pip install git+https://github.com/Carfinance247/datadayessentials.git@v0.1
```
3. For interactive development (editing datscience_core and not having to reinstall the package) we recommend that you clone the repository, enter the directory on the cmd line or terminal and then install the package. Here is an example for linux:
```
git clone https://github.com/Carfinance247/datadayessentials
cd datadayessentials
pip install -e .
```
## Config Setup

There are two config files required by the repository:
1. Global Config - This are config settings that cannot be changed by the user without a PR to the repo
2. Local Config - These are user specifig settings that can be changed and need to be initialised before some functionality is used

We advise that you keep a remote version of the local config so that your team has access to all the same config settings, this would be stored in an azure data lake and resource group that you specify. Your azure login will need to have access to this resource group and data lake.

For setting up the Local Config there are two scenarios:
1. You are working in a team that already has a remote config file that you want to use (containing existing config) that someone else has created that you want to use
2. You want to create a new local config and create a remote config file so that others can use your config
3. You want to create a new config file for your personal use and dont want to share this local config file

### Scenario 1: Using a pre existing team config file (registered dataset in machine learning workspace)
```python
from datadayessentials import initialise_core_config

initialise_core_config(
    environment_name='your_environment_name', 
    'subscription_id'='12345', 
    'resource_group'='rg-your-resource-group', 
    machine_learning_workspace='mlw-your-ml-workspace', 
    data_lake='your-data-lake', 
    create_new_config=False)

## Scenario 2: Creating a new config file for your team (registered dataset in the machine learning workspace)
```python
from datadayessentials import initialise_core_config

team_env_settings = {
  'environment_name': 'your_environment_name',

```


## Scenario 3: Creating a new config file for personal use (no registered dataset in the machine learning workspace)
```python

from datadayessentials.config import ConfigContentUpdater

config_updater = ConfigContentUpdater()
config_updater.add_environment(
    environment_name='your_environment_name', 
    subscription_id='12345', 
    resource_group='rg-your-resource-group', 
    machine_learning_workspace='mlw-your-ml-workspace', 
    data_lake='your-data-lake', 
)
```
This will ensure that if you make changes to core, they are usable the next time you import that module.

Scenarios 1 and 2 are recommended as they allow you to share your config settings with your team and ensure that you are all using the same config settings. Scenario 3 is for personal use only and is not recommended as it will not allow you to share your config settings with your team.
"""

import logging
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
from datadayessentials.config import (
    LocalConfig,
    GlobalConfig,
    ConfigManager,
    ConfigContentUpdater,
)
import os
from pathlib import Path
from datadayessentials.utils import set_global_loggers_to_warning
from datadayessentials.config import ConfigSetup


set_global_loggers_to_warning()
ConfigSetup.create_core_cache_dir_if_not_exists()
ConfigSetup.create_local_config_if_not_exists()

initialise_core_config = ConfigSetup.initialise_core_config

