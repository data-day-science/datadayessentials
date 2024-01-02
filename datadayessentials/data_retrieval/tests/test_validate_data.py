from unittest.mock import DEFAULT
from .test_data import test_path
from .._validate_data import DataFrameValidator
import pytest
import json
import pandas as pd
import numpy as np
import os


@pytest.fixture(scope="module")
def schema():
    test_schema = json.load(open(os.path.join(test_path, "test_schema.json")))
    return test_schema


class TestDataFrameValidator:
    def test_column_validator_float_case(self):
        schema_for_one_column = {
            "description": "All cards Number of Cards Supplying New Fields in Last 3 Months",
            "unique_categories": [],
            "is_date": False,
            "min_val": "0",
            "max_val": "999",
            "dtype": "float",
        }
        test_df = pd.Series([1, 10000, 4.3, -1], name="col3", dtype=float)

        test_df_validator = DataFrameValidator(DEFAULT)
        series_results = test_df_validator._column_validate(
            test_df, schema_for_one_column
        )
        assert np.isnan(series_results.to_list()[1])
        assert np.isnan(series_results.to_list()[3])

    def test_column_validator_other_cases(self):
        schema_for_one_column = {
            "description": "All cards Number of Cards Supplying New Fields in Last 6 Months",
            "unique_categories": ["C", "M"],
            "is_date": False,
            "min_val": "0",
            "max_val": "999",
            "dtype": "str",
        }
        test_df = pd.Series(["x", "M", "1", "-1"], name="col3", dtype=str)

        test_df_validator = DataFrameValidator(DEFAULT)
        series_results = test_df_validator._column_validate(
            test_df, schema_for_one_column
        )
        print(series_results)
        assert series_results.to_list() == ["NaN", "M", "1", "NaN"]

    def test_value_out_of_range(self, schema: dict):
        test_df = pd.DataFrame(
            {
                "col1": ["2010-01-01", "2011-01-02"],
                "col2": ["x", "M"],
                "col3": [3, 100000],
                "col4": ["M", "1000000"],
                "col5": ["xxx", "-11111"],
            }
        )
        # create dataframe with dtypes specified
        test_df = pd.DataFrame.from_dict(test_df)
        dtypes = {key: value["dtype"] for key, value in schema.items()}

        test_df = test_df.astype(
            dtype=dtypes,
        )
        test_df_validator = DataFrameValidator(input_schema=schema)
        df_results = test_df_validator.validate(test_df)

        assert df_results["col2"][0] == "NaN"
        assert np.isnan(df_results["col3"][1])
        assert df_results["col4"][1] == "NaN"
        assert df_results["col5"][0] == "NaN"
        assert df_results["col5"][1] == "NaN"

    def test_less_columns(self, schema: dict):
        test_df = pd.DataFrame({"col2": ["1"]})
        test_df_validator = DataFrameValidator(input_schema=schema)
        df_results = test_df_validator.validate(test_df)
        assert set(["col1", "col2", "col3", "col4", "col5"]) == set(df_results.columns)
