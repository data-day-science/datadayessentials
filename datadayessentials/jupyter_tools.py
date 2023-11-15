from IPython.core.display import display, HTML
import pandas as pd
import os
import sys
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn import metrics
from datadayessentials.authentications import DatabaseAuthentication
from datadayessentials.data_retrieval import TableLoader


def set_screen_width(pecent_width: int = 90):
    """Resizes the cells in the Jupyter notebook to fit the screen
    Example Use Case:
    ```
    from datadayessentials.utils import set_screen_width

    set_screen_width(90)
    ```
    Args:
        pecent_width (int, optional): percentage screen width to use. Defaults to 90.
    """

    display(
        HTML("<style>.container { width:{}% !important; }</style>".format(pecent_width))
    )


def show_df(df: pd.DataFrame, allRows: bool = False):
    """Function to show all columns of a dataframe and not a truncated version

    Example Use Case:
    ```
    from datadayessentials.utils import show_df

    df = pd.read_csv('test.csv')

    show_df(df)
    ```

    Args:
        df (pd.DataFrame): dataframe to visualise
        allRows (bool, optional): indicator to show all rows aswell as columns. Defaults to False.
    """

    if allRows:
        with pd.option_context(
            "display.max_rows", df.shape[0], "display.max_columns", df.shape[1]
        ):
            display(df)
    else:
        with pd.option_context("display.max_columns", df.shape[1]):
            display(df)
    print(df.shape)


class TableScan:

    """
    A class for scanning an SQL Server and returning a list of tables that contain a given string in their column names.

    This class uses the TableLoader class to query the SQL Server and retrieve a list of tables containing the given string.
    The resulting table names are displayed to the user (not printed nor returned).

    Example usage:
        TableScan("Prime")

    Args:
        string_to_search (str): The string that table columns must contain to be included in the resulting list.
        server_name (str): The name of the SQL Server to be queried. Defaults to "readable-secondary".
        use_cache (bool): Whether or not to utilize caching. Defaults to True.
    """

    def __init__(self,
                 string_to_search,
                 server_name: str = "readable-secondary",
                 use_cache: bool = True):

        table_loader = TableLoader(
            authentication=DatabaseAuthentication(),
            sql_statement=self.format_string_to_table_scan_query(string_to_search),
            server_name=server_name,
            use_cache=use_cache,
        )
        output = table_loader.load()

        display(output)

    @staticmethod
    def format_string_to_table_scan_query(string_to_search):
        return f"SELECT DISTINCT TABLE_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE COLUMN_NAME LIKE" \
               f" '%{string_to_search}%'"
