from typing import List
from ._base import IDataFrameValidator
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DataFrameValidator(IDataFrameValidator):
    """
    Validate pandas dataframes using a provided schema. Any values that are not valid according to the schema are replaced by NaN during the validation. Any columns that are missing are added as full columns of NaN values.

    This dataframe validator is designed for schemas like:
    ```json
    {
    "col1":{
        "description": "",
        "uinique_categories": [
            "C",
            "M"
        ],
        "is_date": false,
        "min_val": "0",
        "max_val": "999",
        "dtype": "str"
        }
    }
    ```
    If there are unique categories provided then this class will check if the values either sit in the list of unique categories or are between the min and max numerical values. Otherwise if the column is numerical type then it will check if the value is between the min and max vals.

    Example Use Case:

    ```python
    from datadayessentials.data_retrieval import DataFrameValidator

    example_schema =     {
        "col1":{
            "Description": "",
            "Values": "M,C, 0 - 999",
            "letters": [
                "C",
                "M"
            ],
            "numericals": [],
            "is_date": false,
            "min_val": "0",
            "max_val": "999",
            "is_numeric": true,
            "is_categorical": true,
            "is_contaminated": true,
            "dtype": "str"
            }
        }
        
    validator = DataFrameValidator(example_schema)

    input_df = pd.DataFrame({'col1': ['M', 'C', '25', '144', '1001'']})
    output_df = validator.validate(input_df)
    ```


    """

    def __init__(self, input_schema: dict):
        """Creates an instance of the DataFrameValidator, specifying only the schema. A single validator can be used to validate multiple dataframes that follow the same schema.

        Args:
            input_schema (dict): Schema in the format specified above
        """
        self.input_schema = input_schema

    def validate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Based on the schema provided, values out of range will be replaced.
        Data frame will be returned with np.nan if the column is float.
        'NaN' will be returned instead if the column is in other types.

        Args:
            data (pd.DataFrame): DataFrame to validate

        Returns:
            pd.DataFrame: Same as input with invalid values replaced with NaN
        """
        logger.debug("validating each column")
        data_copy = {}
        for key, schema_for_this_column in tqdm(self.input_schema.items()):
            if key not in data.columns:
                if schema_for_this_column["is_date"] | (
                    schema_for_this_column["dtype"] == "str"
                ):
                    data_copy[key] = ["No Value"] * data.shape[0]
                else:
                    data_copy[key] = [np.nan] * data.shape[0]
            else:
                data_copy[key] = self._column_validate(
                    data[key], schema_for_this_column
                )

        data_copy = pd.DataFrame(data_copy)
        return data_copy

    def _column_validate(
        self, series: pd.Series, schema_for_this_col: dict
    ) -> pd.Series:
        """Validates a specific column with the schema for that column. Returning the same column but with the invalid values replaced with NaN. If the column is a date then no validation is performed

        Args:
            series (pd.Series): Pandas series to validate
            schema_for_this_col (dict): Schema for this specific column

        Returns:
            pd.Series: Pandas series with the invalid values replaced with NaN
        """
        series = series.copy(deep=True)
        if schema_for_this_col["is_date"]:
            return series
        elif (
            schema_for_this_col["dtype"] == "float"
            or schema_for_this_col["dtype"] == "int"
        ):
            series = pd.to_numeric(series, errors="coerce")
            series[
                (series < float(schema_for_this_col["min_val"]))
                | (series > float(schema_for_this_col["max_val"]))
            ] = np.nan

        else:
            unique_values = pd.Series(series.unique())
            valid_categorical = unique_values.isin(
                schema_for_this_col["unique_categories"]
            )
            # If there is no categorical information then there is no validation to do. As if there was numerical information then it should not be string type
            if len(schema_for_this_col["unique_categories"]) == 0 & (
                schema_for_this_col["dtype"] == "str"
            ):
                filled = series.fillna("No Value")
                return filled

            numeric_values = pd.to_numeric(unique_values, errors="coerce")

            min_val = (
                float(schema_for_this_col["min_val"])
                if schema_for_this_col["min_val"]
                else -float("inf")
            )
            max_val = (
                float(schema_for_this_col["max_val"])
                if schema_for_this_col["max_val"]
                else float("inf")
            )
            valid_numerical = (numeric_values >= min_val) & (numeric_values <= max_val)
            # All the valid values are either valid categorical or valid numerical numbers
            valid_bool_mask = valid_categorical | valid_numerical
            values_out_of_range = unique_values[valid_bool_mask]

            series[~series.isin(values_out_of_range)] = "NaN"
        return series
