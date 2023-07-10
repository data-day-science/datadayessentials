# Install pyodbc dependencies required by dtasience_core
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
curl https://packages.microsoft.com/config/ubuntu/16.04/prod.list > /etc/apt/sources.list.d/mssql-release.list
sudo apt-get update
sudo ACCEPT_EULA=Y apt-get -q -y install msodbcsql17

pip install --upgrade pip
TOKEN="$(</dbfs/Shared/dbx/cluster_init_scripts/core_token.txt)"
AZURE_ENVIRONMENT_NAME="$(</dbfs/Shared/dbx/cluster_init_scripts/azure_environment_name.txt)"
AZURE_SUBSCRIPTION_ID="$(</dbfs/Shared/dbx/cluster_init_scripts/azure_subscription_id.txt)"
AZURE_RESOURCE_GROUP="$(</dbfs/Shared/dbx/cluster_init_scripts/azure_resource_group.txt)"
AZURE_ML_WORKSPACE="$(</dbfs/Shared/dbx/cluster_init_scripts/azure_ml_workspace.txt)"
AZURE_DATA_LAKE="$(</dbfs/Shared/dbx/cluster_init_scripts/azure_data_lake.txt)"
export AZURE_TENANT_ID="$(</dbfs/Shared/dbx/cluster_init_scripts/azure_tenant_id.txt)"
export AZURE_CLIENT_SECRET="$(</dbfs/Shared/dbx/cluster_init_scripts/azure_client_secret.txt)"
export AZURE_CLIENT_ID="$(</dbfs/Shared/dbx/cluster_init_scripts/azure_client_id.txt)"

# Install Core
COMMAND="pip install git+https://$TOKEN@github.com/Carfinance247/datascience_core.git"
eval "$COMMAND"

# Initialise Core
COMMAND="python -c \"from datascience_core import initialise_core_config; initialise_core_config(environment_name='$AZURE_ENVIRONMENT_NAME', subscription_id='$AZURE_SUBSCRIPTION_ID', resource_group='$AZURE_RESOURCE_GROUP', machine_learning_workspace='$AZURE_ML_WORKSPACE', data_lake='$AZURE_DATA_LAKE', tenant_id='$AZURE_TENANT_ID', create_new_config=False)\""
eval "$COMMAND"

# Install the pricing tool
COMMAND="pip install --no-deps git+https://$TOKEN@github.com/Carfinance247/datascience_pricing_tool.git"
eval "$COMMAND"

sudo apt-get install yes

yes | pip install mlflow azureml-mlflow
yes | pip uninstall azure-storage-blob azure-storage-file-datalake || true
yes | pip install azure-storage-blob==12.14.1
yes | pip install azure-storage-file-datalake==12.9.1
