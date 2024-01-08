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
from decimal import Decimal
from datetime import datetime, timedelta
import re
from enum import Enum

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
        "QCB.CreditCheckId": 1,  # Integer
        "QCB.MonthsFromEpoch": 6,  # Float
        "QCB.RawResponseId": "4",  # String
        "QCB.LSC898": "R",  # String
        "QCB.LSC899": "R",  # String
        "QCB.HSC415": "R",  # String
        "QCB.MSC410": 0,  # Integer
        "QCB.A": True,  # Boolean
        "QCB.B": False,  # Boolean
        "QCB.C": 3.1415,  # Float
        "QCB.D": "D",  # String
        "QCB.E": [1, 2, 3],  # List
        "QCB.F": {"key": "value"},  # Dictionary
        "QCB.G": (1, 2, 3),  # Tuple
        "QCB.H": None,  # NoneType
        "QCB.J": bytearray(b'hello'),  # ByteArray
        "QCB.K": b'world',  # Bytes
        "QCB.L": Decimal('10.5'),  # Decimal
        "QCB.M": frozenset({1, 2, 3}),  # FrozenSet
        "QCB.N": memoryview(b'abcdef'),  # MemoryView
        "QCB.O": datetime(2022, 1, 5),  # DateTime
        "QCB.P": timedelta(days=7),  # Timedelta
        "QCB.Q": re.compile(r'\d+'),  # Regular Expression
        "QCB.R": Enum('Color', 'RED GREEN BLUE'),  # Enum
        "QCB.S": 42,  # Integer
        "QCB.T": "T",  # String
        "QCB.U": "U",  # String
        "QCB.V": 0.12345,  # Float
        "QCB.W": [4, 5, 6],  # List
        "QCB.X": {"a": 1, "b": 2},  # Dictionary
        "QCB.Y": (4, 5, 6),  # Tuple
        "QCB.Z": None,  # NoneType
    }

    columns_to_split = [
        "QCB.CreditCheckId",
        "QCB.MonthsFromEpoch",
        "QCB.RawResponseId",
        "QCB.LSC898",
        "QCB.LSC899",
        "QCB.HSC415",
        "QCB.MSC410",
        "QCB.A",
        "QCB.B",
        "QCB.C",
        "QCB.D",
        "QCB.E",
        "QCB.F",
        "QCB.G",
        "QCB.H",
        "QCB.J",
        "QCB.K",
        "QCB.L",
        "QCB.M",
        "QCB.N",
        "QCB.O",
        "QCB.P",
        "QCB.Q",
        "QCB.R",
        "QCB.S",
        "QCB.T",
        "QCB.U",
        "QCB.V",
        "QCB.W",
        "QCB.X",
        "QCB.Y",
        "QCB.Z",
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
        expected_output_df["QCB.CreditCheckId"] = expected_output_df[
            "QCB.CreditCheckId"
        ].astype("object")
        expected_output_df["QCB.MSC410"] = expected_output_df["QCB.MSC410"].astype(
            "object"
        )
        num_columns = [
            col for col in expected_output_df.columns if col.endswith("_num")
        ]
        for col in num_columns:
            expected_output_df[col] = expected_output_df[col].astype("float64")
        output_df = output_df[expected_output_df.columns]
        pd.testing.assert_frame_equal(
            output_df,
            expected_output_df,
        )

    def test_process_series(self):

        self.data = {
            "QCB.CreditCheckId": 1,  # Integer
            "QCB.MonthsFromEpoch": 6,  # Float
            "QCB.RawResponseId": "4",  # String
            "QCB.LSC898": "R",  # String
            "QCB.LSC899": "R",  # String
            "QCB.HSC415": "R",  # String
            "QCB.MSC410": 0,  # Integer
            "QCB.A": True,  # Boolean
            "QCB.B": False,  # Boolean
            "QCB.C": 3.1415,  # Float
            "QCB.D": "D",  # String
            #"QCB.E": [1, 2, 3],  # List
            "QCB.F": {"key": "value"},  # Dictionary
            "QCB.G": (1, 2, 3),  # Tuple
            "QCB.H": None,  # NoneType
            "QCB.J": bytearray(b'hello'),  # ByteArray
            "QCB.K": b'world',  # Bytes
            "QCB.L": Decimal('10.5'),  # Decimal
            "QCB.M": frozenset({1, 2, 3}),  # FrozenSet
            "QCB.N": memoryview(b'abcdef'),  # MemoryView
            "QCB.O": datetime(2022, 1, 5),  # DateTime
            "QCB.P": timedelta(days=7),  # Timedelta
            "QCB.Q": re.compile(r'\d+'),  # Regular Expression
            #"QCB.R": Enum('Color', 'RED GREEN BLUE'),  # Enum
            "QCB.S": 42,  # Integer
            "QCB.T": "T",  # String
            "QCB.U": "U",  # String
            "QCB.V": 0.12345,  # Float
        }

        self.columns_to_split = [
            "QCB.CreditCheckId",
            "QCB.MonthsFromEpoch",
            "QCB.RawResponseId",
            "QCB.LSC898",
            "QCB.LSC899",
            "QCB.HSC415",
            "QCB.MSC410",
            "QCB.A",
            "QCB.B",
            "QCB.C",
            "QCB.D",
            #"QCB.E",
            "QCB.F",
            "QCB.G",
            "QCB.H",
            "QCB.J",
            "QCB.K",
            "QCB.L",
            "QCB.M",
            "QCB.N",
            "QCB.O",
            "QCB.P",
            "QCB.Q",
            #"QCB.R",
            "QCB.S",
            "QCB.T",
            "QCB.U",
            "QCB.V",
        ]

        series = pd.Series(self.data)
        splitter = CategoricalColumnSplitter(
            categorical_columns_to_split=self.columns_to_split
        )
        output_series = splitter.process(series)
        output_series.to_csv(os.path.join(test_path, "output_df.csv"), index=False)
        expected_series = pd.Series(
            data={
                "QCB.CreditCheckId": np.nan,  # Integer
                "QCB.CreditCheckId_num": 1,  # Float
                "QCB.MonthsFromEpoch": "D",  # Float
                "QCB.MonthsFromEpoch_num": 6,  # Float
                "QCB.RawResponseId": "D",  # String
                "QCB.RawResponseId_num": 4,  # Float
                "QCB.LSC898": "R",  # String
                "QCB.LSC898_num": 6,  # Float
                "QCB.LSC899": "R",  # String
                "QCB.LSC899_num": 6,  # Float
                "QCB.HSC415": "R",  # String
                "QCB.HSC415_num": 6,  # Float
                "QCB.MSC410": np.nan,  # Integer
                "QCB.MSC410_num": 0,  # Float
                "QCB.A": np.nan,  # Boolean
                "QCB.A_num": 1,  # Float
                "QCB.B": np.nan,  # Boolean
                "QCB.B_num": 0,  # Float
                "QCB.C": 3.1415,  # Float
                "QCB.C_num": 3.1415,  # Float
                "QCB.D": "D",  # String
                "QCB.D_num": 5,  # Float
                #"QCB.E": [1, 2, 3],  # List
                #"QCB.E_num": np.nan,  # Float
                "QCB.F": {"key": "value"},  # Dictionary
                "QCB.F_num": np.nan,  # Float
                "QCB.G": (1, 2, 3),  # Tuple
                "QCB.G_num": np.nan,  # Float
                "QCB.H": np.nan,  # NoneType
                "QCB.H_num": np.nan,  # Float
                "QCB.J": bytearray(b'hello'),  # ByteArray
                "QCB.J_num": np.nan,  # Float
                "QCB.K": b'world',  # Bytes
                "QCB.K_num": np.nan,  # Float
                "QCB.L": Decimal('10.5'),  # Decimal
                "QCB.L_num": 10.5,  # Float
                "QCB.M": frozenset({1, 2, 3}),  # FrozenSet
                "QCB.M_num": np.nan,  # Float
                "QCB.N": memoryview(b'abcdef'),  # MemoryView
                "QCB.N_num": np.nan,  # Float
                "QCB.O": datetime(2022, 1, 5),  # DateTime
                "QCB.O_num": np.nan,  # Float
                "QCB.P": timedelta(days=7),  # Timedelta
                "QCB.P_num": np.nan,  # Float
                "QCB.Q": re.compile(r'\d+'),  # Regular Expression
                "QCB.Q_num": np.nan,  # Float
                #"QCB.R": Enum('Color', 'RED GREEN BLUE'),  # Enum
                #"QCB.R_num": np.nan,  # Float
                "QCB.S": 42,  # Integer
                "QCB.S_num": 42,  # Float
                "QCB.T": "T",  # String
                "QCB.T_num": np.nan,  # Float
                "QCB.U": "U",  # String
                "QCB.U_num": np.nan,  # Float
                "QCB.V": 0.12345,  # Float
                "QCB.V_num": 0.12345,  # Float
            }
        )
        output_series = output_series[expected_series.index]

        for key in expected_series.index:
            expected_value = expected_series[key]
            actual_value = output_series[key]

            print(key)
            print(output_series[key])

            if isinstance(expected_value, np.float64):
                if np.isnan(expected_value):
                    self.assertTrue(np.isnan(actual_value), f"Mismatch in {key}: expected NaN, got {actual_value}")
                else:
                    self.assertTrue(np.isclose(actual_value, expected_value),
                                    f"Mismatch in {key}: expected {expected_value}, got {actual_value}")
            else:
                # For non-floats, handle NaN comparison separately
                if pd.isna(expected_value)and pd.isna(actual_value):
                    # Both are NaN (or None, NaT, etc.)
                    self.assertTrue(True)
                else:
                    # Standard equality check
                    self.assertEqual(actual_value, expected_value,
                                     f"Mismatch in {key}: expected {expected_value}, got {actual_value}")

    def test_series_and_df_same_output(self):

        self.data = {
            "QCB.CreditCheckId": 1,  # Integer
            "QCB.MonthsFromEpoch": 6,  # Float
            "QCB.RawResponseId": "4",  # String
            "QCB.LSC898": "R",  # String
            "QCB.LSC899": "R",  # String
            "QCB.HSC415": "R",  # String
        }

        self.columns_to_split = [
            "QCB.CreditCheckId",
            "QCB.MonthsFromEpoch",
            "QCB.RawResponseId",
            "QCB.LSC898",
            "QCB.LSC899",
            "QCB.HSC415",
        ]

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
                self.assertEqual(
                    output_df[key].values[0], output_series_df[key].values[0]
                )


class TestMapWithType(unittest.TestCase):
    def test_correct_type_no_mapping_needed(self):
        # same type, not in mapping
        self.assertEqual(CategoricalColumnSplitter.map_with_type("number", 5, {1: "one"}), 5)

    def test_correct_type_with_mapping(self):
        # same type, in mapping
        self.assertEqual(CategoricalColumnSplitter.map_with_type("string", 1, {1: "one"}), "one")

    def test_incorrect_type_with_mapping(self):
        # different type, in mapping
        self.assertEqual(CategoricalColumnSplitter.map_with_type("string", 2, {2: "two"}), "two")

    def test_incorrect_type_no_mapping(self):
        # different type, not in mapping
        self.assertTrue(np.isnan(CategoricalColumnSplitter.map_with_type("number", "test", {"a": 1})))

    def test_unsupported_type(self):
        # unsupported type
        self.assertTrue(np.isnan(CategoricalColumnSplitter.map_with_type("number", [], {1: 1})))

    def test_invalid_mapping(self):
        # invalid mapping
        with self.assertRaises(TypeError):
            CategoricalColumnSplitter.map_with_type("number", 2, "not a dictionary")

    def test_none_value(self):
        # None value
        self.assertTrue(np.isnan(CategoricalColumnSplitter.map_with_type("number", None, {1: 1})))
