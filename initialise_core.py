from datadayessentials import initialise_core_config
from datadayessentials.config import LocalConfig
import os

print(os.environ.keys())

environment_name = os.environ.get('AZURE_ENVIRONMENT_NAME')
subscription_id = os.environ.get('AZURE_SUBSCRIPTION_ID')
resource_group = os.environ.get('AZURE_RESOURCE_GROUP')
machine_learning_workspace = os.environ.get('AZURE_ML_WORKSPACE')
data_lake = os.environ.get('AZURE_DATA_LAKE')
tenant_id = os.environ.get('AZURE_TENANT_ID')

initialise_core_config(
    environment_name=environment_name,
    subscription_id=subscription_id,
    resource_group=resource_group,
    machine_learning_workspace=machine_learning_workspace,
    data_lake=data_lake,
    tenant_id=tenant_id,
    create_new_config=True
)