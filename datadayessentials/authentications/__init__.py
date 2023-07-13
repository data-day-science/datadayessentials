"""
This module contains all classes associated with authentication. This includes DWHAuthentication for authenticating with the DWH datawarehouse, and DataLakeAuthentication. 

For all azure credentials there is a singleton class AzureAuthenticationSingleton that maintains one set of credentials only. This class attempts to authenticate the login using enrironment variables (AZURE_CLIENT_ID, AZURE_TENANT_ID and AZURE_CLIENT_SECRET), and if this fails it tries to authenticate using an interative browser popup.

"""
from ._authentications import (
    DatabaseAuthentication,
    DataLakeAuthentication,
    SQLServerConnection,
)
from ._base import IAuthentication, AzureAuthenticationSingleton
import logging


__all__ = [
    'IAuthentication',
    'DataLakeAuthentication',
    'DatabaseAuthentication',
    'SQLServerConnection',
    'AzureAuthenticationSingleton',
]
