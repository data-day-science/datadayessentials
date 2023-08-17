# Install pyodbc dependencies required by dtasience_core
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
curl https://packages.microsoft.com/config/ubuntu/16.04/prod.list > /etc/apt/sources.list.d/mssql-release.list
sudo apt-get update
sudo ACCEPT_EULA=Y apt-get -q -y install msodbcsql17

pip install --upgrade pip
TOKEN="$(</dbfs/Shared/dbx/cluster_init_scripts/core_token.txt)"
AZURE_ENVIRONMENT_NAME="$(</dbfs/Shared/dbx/cluster_init_scripts/azure_environment_name.txt)"
export AZURE_APP_CONFIG_CONNECTION_STRING="$(</dbfs/Shared/dbx/cluster_init_scripts/azure_app_config_connection_string.txt)"



# Install Core
COMMAND="pip install git+https://$TOKEN@github.com/Carfinance247/datadayessentials.git"
eval "$COMMAND"


COMMAND="python -c \"from datadayessentials.config import Config; config_manager=Config(); config_manager.set_default_variables()\""
eval "$COMMAND"



sudo apt-get install yes

yes | pip install mlflow azureml-mlflow
yes | pip uninstall azure-storage-blob azure-storage-file-datalake || true
yes | pip install azure-storage-blob==12.14.1
yes | pip install azure-storage-file-datalake==12.9.1
