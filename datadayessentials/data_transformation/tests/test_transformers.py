import json
import os
import unittest
from datetime import datetime
from math import isclose
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import pytest
from numpy import float64

from .test_data import test_path
from .._transformers import (
    DataFrameTimeSlicer,
    CategoricalColumnSplitter,
    is_data_size_small,
    DataFrameColumnTypeSplitter,
)


def to_datetime(str_datetime):
    return datetime.strptime(str_datetime, "%Y%m%d")


class TestDataFrameTimeSlicer:
    @pytest.fixture
    def input_data(self):
        data = {
            "date": ["20220101", "20220201", "20220301", "20220401", "20220501"],
            "col2": [1, 2, 3, 4, 5],
            "col3": [1, 2, 3, 4, 5],
            "col4": [1, 2, 3, 4, 5],
            "col5": [1, 2, 3, 4, 5],
        }
        data = pd.DataFrame(data)
        return data

    def test_col_name_exists(self, input_data):
        input_data["date"] = pd.to_datetime(input_data["date"])

        slicer = DataFrameTimeSlicer(
            "UNKNOWN_COL", to_datetime("20220101"), to_datetime("20220301")
        )
        with pytest.raises(KeyError):
            slicer.process(input_data)

        slicer = DataFrameTimeSlicer(
            "date", to_datetime("20220101"), to_datetime("20220301")
        )
        actual = slicer.process(input_data)
        assert isinstance(actual, pd.DataFrame)

    def test_col_is_a_date(self, input_data):
        input_data["date"] = pd.to_datetime(input_data["date"])
        slicer = DataFrameTimeSlicer(
            "col2", to_datetime("20220101"), to_datetime("20220301")
        )
        with pytest.raises(TypeError):
            slicer.process(input_data)

    def test_slicer_slices(self, input_data):
        input_data["date"] = pd.to_datetime(input_data["date"])

        slicer = DataFrameTimeSlicer(
            "date", to_datetime("20220101"), to_datetime("20220301")
        )
        actual = slicer.process(input_data)
        assert isinstance(actual, pd.DataFrame)
        assert actual.shape[0] == 3

        slicer = DataFrameTimeSlicer(
            "date", to_datetime("20220101"), to_datetime("20220201")
        )
        actual = slicer.process(input_data)
        assert isinstance(actual, pd.DataFrame)
        assert actual.shape[0] == 2

    def test_convert_to_datetime(self, input_data):
        slicer = DataFrameTimeSlicer(
            "date",
            to_datetime("20220101"),
            to_datetime("20220301"),
            convert_to_datetime_format="%Y%m%d",
        )
        actual = slicer.process(input_data)
        assert isinstance(actual, pd.DataFrame)
        assert actual.shape[0] == 3


@pytest.fixture
def payload_init():
    payload_json = """[
    {
        "index": "AppIdentifier",
        "0": 2
    },
     {
        "index": "App PostcodeEmployment",
        "0": null
    },
    {
        "index": "BSB BIC",
        "0": 0
    },
    {
        "index": "BSB BLC",
        "0": "{ND}"
    },
    {
        "index": "BSB CQ",
        "0": "D"
    },
    {
        "index": "BSB DQC",
        "0": "M"
    },
    {
        "index": "App EmploymentStatusName",
        "0": "Employed Full Time"
    },
     {
        "index": "App AgeAtApplication",
        "0": 54.0
    }]"""

    payload_tall = (
        pd.read_json(payload_json, dtype=object)
        .set_index("index")
        .fillna(np.nan)
        .rename(index={"AppIdentifier": "ApplicationId"})
    )
    df_payload_flat = payload_tall.T
    return {"test_tall": payload_tall, "test_flat": df_payload_flat}


def test_is_data_size_small():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert is_data_size_small(df) is True


def test_is_data_size_small_false():
    df = pd.DataFrame(
        np.random.randint(0, 100, size=(110000, 1000)), columns=list(range(1000))
    )
    assert is_data_size_small(df) is False



class TestDataFrameColumnTypeSplitter(unittest.TestCase):
    def test_initialization(self):
        splitter = DataFrameColumnTypeSplitter()
        self.assertTrue(isinstance(splitter, DataFrameColumnTypeSplitter))

    def test_process(self):
        data = {
            "TextColumn1": ["1", 1, "2.25", np.nan, "789ghi"],
            "TextColumn2": ["1", 1, "2.25", np.nan, "789ghi"],
        }

        df = pd.DataFrame(data)
        splitter = DataFrameColumnTypeSplitter()
        result = splitter.process(df)

        expected_columns = [
            "TextColumn1_num",
            "TextColumn2_num",
            "TextColumn1",
            "TextColumn2",
        ]
        actual_columns = result.columns.tolist()

        print(result)
        self.assertListEqual(actual_columns, expected_columns)

        self.assertEqual(result["TextColumn1_num"].values[0], 1)
        self.assertEqual(result["TextColumn1_num"].values[1], 1)
        self.assertEqual(result["TextColumn1_num"].values[2], 2.25)
        # self.assertEqual(result['TextColumn1_num'].values[3], np.nan)
        # self.assertEqual(result['TextColumn1_num'].values[4], np.nan)

        self.assertEqual(result["TextColumn2_num"].values[0], 1.00)
        self.assertEqual(result["TextColumn2_num"].values[1], 1.00)
        self.assertEqual(result["TextColumn2_num"].values[2], 2.25)
        # self.assertEqual(result['TextColumn2_num'].values[3], np.nan)
        # self.assertEqual(result['TextColumn2_num'].values[4], np.nan)

        # self.assertEqual(result['TextColumn1'].values[0], np.nan)
        # self.assertEqual(result['TextColumn1'].values[1], np.nan)
        # self.assertEqual(result['TextColumn1'].values[2], np.nan)
        # self.assertEqual(result['TextColumn1_num'].values[3], np.nan)
        self.assertEqual(result["TextColumn1"].values[4], "789ghi")

        # self.assertEqual(result['TextColumn2'].values[0], np.nan)
        # self.assertEqual(result['TextColumn2'].values[1], np.nan)
        # self.assertEqual(result['TextColumn2'].values[2], np.nan)
        # self.assertEqual(result['TextColumn2'].values[3], np.nan)
        self.assertEqual(result["TextColumn2"].values[4], "789ghi")

        df = pd.DataFrame(data)
        splitter2 = DataFrameColumnTypeSplitter(only_process_columns=["TextColumn1"])
        result2 = splitter2.process(df)
        print(result2.columns)

        # Check if the DataFrame has the expected columns
        expected_columns = ["TextColumn1_num", "TextColumn1", "TextColumn2"]
        self.assertListEqual(result2.columns.tolist(), expected_columns)


class TestCategoricalColumnSplitter(unittest.TestCase):
    data = {
            "QCB.CreditCheckId": "1",
            "QCB.MonthsFromEpoch": "6",
            "QCB.RawResponseId": "4",
            "QCB.LSC898": "R",
            "QCB.LSC899": "R",
            "QCB.HSC415": "R",
            "QCB.MSC410": "0",
        }
    columns_to_split = [
                "QCB.CreditCheckId",
                "QCB.MonthsFromEpoch",
                "QCB.RawResponseId",
                "QCB.LSC898",
                "QCB.LSC899",
                "QCB.HSC415",
                "QCB.MSC410",
            ]
    def test_process_df(self):
        df = pd.DataFrame.from_dict(self.data, orient="index").T
        splitter = CategoricalColumnSplitter(
            categorical_columns_to_split=self.columns_to_split
        )
        output_df = splitter.process(df)
        expected_output_df = pd.DataFrame(
            {
                "QCB.CreditCheckId": [np.nan],
                "QCB.CreditCheckId_num": [1],
                "QCB.MonthsFromEpoch": ["D"],
                "QCB.MonthsFromEpoch_num": [6],
                "QCB.RawResponseId": ["D"],
                "QCB.RawResponseId_num": [4],
                "QCB.LSC898": ["R"],
                "QCB.LSC898_num": [6],
                "QCB.LSC899": ["R"],
                "QCB.LSC899_num": [6],
                "QCB.HSC415": ["R"],
                "QCB.HSC415_num": [6],
                "QCB.MSC410": [np.nan],
                "QCB.MSC410_num": [0],
            }
        )
        expected_output_df['QCB.CreditCheckId'] = expected_output_df['QCB.CreditCheckId'].astype('object')
        expected_output_df['QCB.MSC410'] = expected_output_df['QCB.MSC410'].astype('object')
        num_columns = [col for col in expected_output_df.columns if col.endswith('_num')]
        for col in num_columns:
            expected_output_df[col] = expected_output_df[col].astype('float64')
        output_df = output_df[expected_output_df.columns]
        pd.testing.assert_frame_equal(
            output_df,
            expected_output_df,
        )
    
    def test_process_series(self):
        series = pd.Series(self.data)
        splitter = CategoricalColumnSplitter(
            categorical_columns_to_split=self.columns_to_split
        )
        output_series = splitter.process(series)
        expected_series = pd.Series(
            {
                "QCB.CreditCheckId": np.nan,
                "QCB.CreditCheckId_num": 1,
                "QCB.MonthsFromEpoch": "D",
                "QCB.MonthsFromEpoch_num": 6,
                "QCB.RawResponseId": "D",
                "QCB.RawResponseId_num": 4,
                "QCB.LSC898": "R",
                "QCB.LSC898_num": 6,
                "QCB.LSC899": "R",
                "QCB.LSC899_num": 6,
                "QCB.HSC415": "R",
                "QCB.HSC415_num": 6,
                "QCB.MSC410": np.nan,
                "QCB.MSC410_num": 0,
            }
        )
        output_series = output_series[expected_series.index]
        for key in expected_series.index:
            if isinstance(expected_series[key], float64):
                self.assertTrue(isclose(output_series[key], expected_series[key]))
            elif expected_series[key] is np.nan:
                self.assertTrue(np.isnan(output_series[key]))
            else:
                self.assertEqual(output_series[key], expected_series[key])
      
    def test_series_and_df_same_output(self):
        df = pd.DataFrame.from_dict(self.data, orient="index").T
        series = pd.Series(self.data)
        splitter = CategoricalColumnSplitter(
            categorical_columns_to_split=self.columns_to_split
        )
        output_df = splitter.process(df)
        output_series = splitter.process(series)
        output_series_df = pd.DataFrame(output_series).T
        for key in self.data.keys():
            if output_df[key].isna().values[0]:
                self.assertTrue(np.isnan(output_df[key].values[0]))
                self.assertTrue(np.isnan(output_series_df[key].values[0]))
            else:
                self.assertEqual(output_df[key].values[0], output_series_df[key].values[0])
