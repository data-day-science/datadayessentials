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

from .test_data import test_path
from .._transformers import (
    ColumnRenamer,
    DataFrameTimeSlicer,
    DataFrameCaster,
    InvalidPayloadDropperByPrefix,
    GranularColumnDropper,
    ColumnFiller,
    PreprocessingError,
    LowerCaseTransformer,
    CatTypeConverter,
    ColumnDotRenamer,
    CategoricalColumnSplitter,
    is_data_size_small,
    SimpleCatTypeConverter, DataFrameColumnTypeSplitter,
)


def to_datetime(str_datetime):
    return datetime.strptime(str_datetime, "%Y%m%d")


class TestColumnRenamer:
    @pytest.fixture
    def input_data(self):
        data = {
            "col1": [1, 2, 3, 4, 5],
            "col2": [1, 2, 3, 4, 5],
            "col3": [1, 2, 3, 4, 5],
            "col4": [1, 2, 3, 4, 5],
            "col5": [1, 2, 3, 4, 5],
        }
        return pd.DataFrame(data)

    def test_mapper_is_dict(self):
        input_mapper_dict = ["Fail"]
        with pytest.raises(TypeError):
            ColumnRenamer(name_mapping=input_mapper_dict)

    def test_column_renamer_renames(self, input_data):
        input_mapper_dict = {"col1": "colB"}
        expected = ["colB", "col2", "col3", "col4", "col5"]
        col_renamer = ColumnRenamer(name_mapping=input_mapper_dict)
        renamed_df = col_renamer.process(input_data)
        assert all(x in renamed_df.columns for x in expected)
        assert isinstance(renamed_df, pd.DataFrame)

    def test_invalid_col_does_not_rename(self, input_data):
        input_mapper_dict = {"UNKNOWN_COLUMN": "colB"}
        expected = ["col1", "col2", "col3", "col4", "col5"]
        col_renamer = ColumnRenamer(name_mapping=input_mapper_dict)
        renamed_df = col_renamer.process(input_data)
        assert all(x in renamed_df.columns for x in expected)
        assert "colB" not in renamed_df.columns


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


class TestDataFrameCaster:
    @pytest.fixture
    def input_data(self):
        data = {
            "col1": ["20220101", "20220201", "20220301", "20220401", "20220501"],
            "col2": ["M", "C", "C", "M", "M"],
            "col3": [1, 2, 3, 4, 5],
            "col4": [1, 2, "M", 4, "C"],
            "col5": ["M", 2, "M", "C", 5],
        }
        data = pd.DataFrame(data)
        data["col1"] = pd.to_datetime(data["col1"])
        return data

    @pytest.fixture
    def schema(self):
        with open(os.path.join(test_path, "test_schema.json"), "r") as schema_file:
            return json.load(schema_file)

    def test_casts_correctly(self, input_data, schema):
        caster = DataFrameCaster(target_schema=schema)
        actual_df = caster.process(input_data)

        assert ptypes.is_datetime64_any_dtype(actual_df["col1"])
        assert ptypes.is_string_dtype(actual_df["col2"])
        assert ptypes.is_float_dtype(actual_df["col3"])
        assert ptypes.is_string_dtype(actual_df["col4"])
        assert ptypes.is_string_dtype(actual_df["col5"])

    def test_missing_col_raises_error(self, input_data, schema):
        input_data.drop(columns="col1", inplace=True)
        caster = DataFrameCaster(target_schema=schema)
        with pytest.raises(KeyError):
            caster.process(input_data)

    def test_bad_schema_type_fails(self, input_data):
        schema = [1, 2, 3, 4, 5]
        caster = DataFrameCaster(target_schema=schema)
        with pytest.raises(AttributeError):
            caster.process(input_data)


class TestInvalidPayloadDropperByPrefix:
    def test_drop_invalid_payload(self):
        test_payload = pd.DataFrame(
            {
                "ApplicationId": [1, 2, 3, 4, 5, 6],
                "App X": ["a", "b", "c", None, None, "d"],
                "App XX": ["b", "c", None, None, "d", "a"],
                "BSB_X": ["c", None, None, "d", "a", "b"],
                "BSB_XX": ["b", "c", None, None, "d", "a"],
                "QCB_X": ["b", "x", None, None, None, "a"],
            }
        )

        prefixes = ["App ", "BSB_", "QCB_"]
        actual_results = InvalidPayloadDropperByPrefix(prefixes).process(test_payload)
        assert actual_results.shape[0] == 3
        assert all([id in [1, 2, 6] for id in actual_results["ApplicationId"].values])


class TestGranularColumnDropper:
    def test_drops_column(self):
        df = pd.DataFrame(
            {
                "col1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "col2": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
                "col3": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "col4": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3],
            }
        )
        dropper = GranularColumnDropper(threshold=0.4)
        actual = dropper.process(df)

        expected = ["col3", "col4"]

        assert set(actual.columns) == set(expected)


class TestColumnFiller:
    @pytest.fixture
    def payload_init(self):
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
        df_payload_tall = (
            pd.read_json(payload_json, dtype=object)
            .set_index("index")
            .fillna(np.nan)
            .rename(index={"AppIdentifier": "ApplicationId"})
        )
        df_payload_flat = df_payload_tall.T
        return {"dfTestTall": df_payload_tall, "dfTestFlat": df_payload_flat}

    def test_force_modify_one_field(self, payload_init):
        enforce = True
        index_name = ["App AgeAtApplication"]
        expected_value = 100
        step_speedy = ColumnFiller(index_name, [], expected_value, enforce, "speedy")
        speedy_value = step_speedy.process(payload_init["dfTestTall"]).loc[
            "App AgeAtApplication", "0"
        ]
        step_flat = ColumnFiller(index_name, [], expected_value, enforce, "flat")
        flat_value = step_flat.process(payload_init["dfTestFlat"])[
            "App AgeAtApplication"
        ].values[0]
        assert speedy_value == 100
        assert flat_value == 100

    def test_unforced_modify_one_field(self, payload_init):
        enforce = False
        index_name = ["App AgeAtApplication"]
        expected_value = 100
        step_speedy = ColumnFiller(index_name, [], expected_value, enforce, "speedy")
        actual_speedy_value = step_speedy.process(payload_init["dfTestTall"]).loc[
            "App AgeAtApplication", "0"
        ]
        step_flat = ColumnFiller(index_name, [], expected_value, enforce, "flat")
        actual_flat_value = step_flat.process(payload_init["dfTestFlat"])[
            "App AgeAtApplication"
        ].values[0]
        assert actual_speedy_value == 54
        assert actual_flat_value == 54

    def test_unforced_modify_two_field(self, payload_init):
        enforce = False
        ls_name = ["App AgeAtApplication", "test field"]

        step_speedy = ColumnFiller(ls_name, [], 100, enforce, "speedy")
        actual_speedy_age_value = step_speedy.process(payload_init["dfTestTall"]).loc[
            "App AgeAtApplication", "0"
        ]
        actual_speedy_test_value = step_speedy.process(payload_init["dfTestTall"]).loc[
            "test field", "0"
        ]
        step_flat = ColumnFiller(ls_name, [], 100, enforce, "flat")
        actual_flat_age_value = step_flat.process(payload_init["dfTestFlat"])[
            "App AgeAtApplication"
        ].values[0]
        actual_flat_test_value = step_flat.process(payload_init["dfTestFlat"])[
            "test field"
        ].values[0]
        assert actual_speedy_age_value == 54
        assert actual_speedy_test_value == 100
        assert actual_flat_age_value == 54
        assert actual_flat_test_value == 100

    def test_unforced_modify_two_field_with_one_top_feature(self, payload_init):
        enforce = False
        name = ["App AgeAtApplication", "test field"]
        step_speedy = ColumnFiller(name, ["test field"], 100, enforce, "speedy")
        with pytest.raises(PreprocessingError):
            actual_speedy_age_value = step_speedy.process(payload_init["dfTestTall"])

        step_flat = ColumnFiller(name, ["test field"], 100, enforce, "flat")
        with pytest.raises(PreprocessingError):
            actual_flat_age_value = step_flat.process(payload_init["dfTestFlat"])

    def test_forced_modify_two_field_with_one_top_feature(self, payload_init):
        enforce = True
        name = ["App AgeAtApplication", "test field"]

        step_speedy = ColumnFiller(name, ["test field"], 100, enforce, "speedy")
        with pytest.raises(PreprocessingError):
            speedy_age_value = step_speedy.process(payload_init["dfTestTall"])

        step_flat = ColumnFiller(name, ["test field"], 100, enforce, "flat")
        with pytest.raises(PreprocessingError):
            flat_age_value = step_flat.process(payload_init["dfTestFlat"])


class TestLowerCaseTransformer:
    @pytest.fixture
    def payload_init(self):
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
        df_payload_tall = (
            pd.read_json(payload_json, dtype=object)
            .set_index("index")
            .fillna(np.nan)
            .rename(index={"AppIdentifier": "ApplicationId"})
        )
        df_payload_flat = df_payload_tall.T
        return {"dfTestTall": df_payload_tall, "dfTestFlat": df_payload_flat}

    def test_lower_case_transformer_flat(self, payload_init):
        # Build
        cat_features = ["BSB CQ"]
        step = LowerCaseTransformer(col_names=cat_features, fmt="flat")

        # Run
        df_out = step.process(payload_init["dfTestFlat"], verbose=False)

        # Test
        assert df_out["BSB CQ"].values == ["d"]

    def test_lower_case_transformer_speedy(self, payload_init):
        # Build
        cat_features = ["BSB CQ"]
        step = LowerCaseTransformer(col_names=cat_features, fmt="speedy")

        # Run
        df_out = step.process(payload_init["dfTestTall"], verbose=False)

        # Test
        assert df_out.loc["BSB CQ"].values == ["d"]


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


class TestCatTypeConverters:
    def test_cat_type_converter_empty_list(self, payload_init):
        cat_features = []

        step = CatTypeConverter(cat_features)

        actual_processed_payload = step.process(payload_init["test_flat"])

        assert actual_processed_payload["BSB BIC"].values[0] == 0

    def test_cat_type_converter_tall(self, payload_init):
        cat_features = ["BSB BIC", "App PostcodeEmployment"]
        step = CatTypeConverter(cat_features)
        processed_payload = step.process(payload_init["test_tall"])
        assert processed_payload["BSB BIC"][0] == "0"
        assert processed_payload["App PostcodeEmployment"][0] == "NaN"

    def test_cat_type_converter_flat(self, payload_init):
        cat_features = ["BSB BIC", "App PostcodeEmployment"]
        step = CatTypeConverter(cat_features)
        processed_payload = step.process(payload_init["test_flat"])
        assert processed_payload["BSB BIC"][0] == "0"
        assert processed_payload["App PostcodeEmployment"][0] == "NaN"

    def test_simple_cat_type_converter_no_dates_pass(self):
        test_df = pd.DataFrame(
            {"col1": [1, 2, 3], "col2": ["a", "b", "c"], "col3": [True, False, True]}
        )
        cat_features = ["col2", "col3"]
        simple_converter = SimpleCatTypeConverter(cat_features)
        processed_payload = simple_converter.process(test_df)

        assert processed_payload["col1"].dtype == "int64"
        assert processed_payload["col2"].dtype == "category"
        assert processed_payload["col3"].dtype == "category"
        pass

    def test_simple_cat_type_converter_dates_pass(self):
        test_df = pd.DataFrame(
            {"col1": [1, 2, 3], "col2": ["a", "b", "c"], "col3": [True, False, True]}
        )
        simple_converter = SimpleCatTypeConverter(
            categorical_columns=["col2"], date_columns=["col3"]
        )
        processed_payload = simple_converter.process(test_df)

        assert processed_payload["col1"].dtype == "int64"
        assert processed_payload["col2"].dtype == "category"
        assert processed_payload["col3"].dtype == "bool"
        pass

    def test_ensure_categorical_not_passed_converted_to_nan(self):
        test_df = pd.DataFrame(
            {"col1": [1, 2, 3], "col2": ["a", "b", "c"], "col3": [True, False, True]}
        )
        simple_converter = SimpleCatTypeConverter(
            categorical_columns=["col1"], date_columns=["col3"]
        )
        test_df["col1"] = test_df["col1"].astype("str")

        processed_payload = simple_converter.process(test_df)
        assert processed_payload["col1"].dtype == "category"
        assert processed_payload["col2"].dtype == "float"


class TestColumnDotRenamer:
    def payload_init(self):
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

        df_payload_tall = (
            pd.read_json(payload_json, dtype=object)
            .set_index("index")
            .fillna(np.nan)
            .rename(index={"AppIdentifier": "ApplicationId"})
        )
        df_payload_flat = df_payload_tall.T
        return {"test_tall": df_payload_tall, "test_flat": df_payload_flat}

    def test_column_dot_renamer(self):
        df = pd.DataFrame({"a b": [1, 2, 3], "c d": [4, 5, 6], "c.b": [7, 8, 9]})
        df_out = ColumnDotRenamer(fmt="flat", from_name=" ", to_name=".").process(df)

        assert df_out.columns.tolist() == ["a.b", "c.d", "c.b"]

    def test_column_dot_renamer_speedy(self):
        df = pd.DataFrame([1, 4, 7], index=["a b", "c d", "c.b"])
        df_out = ColumnDotRenamer(fmt="speedy", from_name=" ", to_name=".").process(df)

        assert df_out.index.tolist() == ["a.b", "c.d", "c.b"]

    def test_column_dot_renamer_speedy_json(self, payload_init):
        df = payload_init["test_tall"]
        df_out = ColumnDotRenamer(fmt="speedy", from_name=" ", to_name=".").process(df)

        assert df_out.index.tolist()[2] == "BSB.BIC"


def test_is_data_size_small():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert is_data_size_small(df) is True


def test_is_data_size_small_false():
    df = pd.DataFrame(
        np.random.randint(0, 100, size=(110000, 1000)), columns=list(range(1000))
    )
    assert is_data_size_small(df) is False


class TestDataTransformer:
    def test_split_categorical_column(self):
        data_transformer = CategoricalColumnSplitter(
            categorical_columns_to_split=["QCB1", "QCB2"]
        )
        df = pd.DataFrame(
            {
                "QCB1": ["0", "1", "2", "3", "4", "5", "6", "D", "R", "V", "S", "A"],
                "QCB2": ["D", "R", "V", "S", "A", "0", "1", "2", "3", "4", "5", "6"],
            }
        )
        df_out = data_transformer.process(df)
        ["D", "R", "V", "S", "A"], [5, 6, 6, 0, 2]
        assert df_out["QCB1_num"].equals(
            pd.Series([0, 1, 2, 3, 4, 5, 6, 5, 6, 6, 0, 2])
        )
        assert df_out["QCB2_num"].equals(
            pd.Series([5, 6, 6, 0, 2, 0, 1, 2, 3, 4, 5, 6])
        )
        assert df_out["QCB1"].equals(
            pd.Series(
                [np.nan, np.nan, np.nan, "D", "D", "D", "D", "D", "R", "V", "S", "A"]
            )
        )
        assert df_out["QCB2"].equals(
            pd.Series(
                ["D", "R", "V", "S", "A", np.nan, np.nan, np.nan, "D", "D", "D", "D"]
            )
        )

    def test_split_categorical_column_with_nan(self):
        data_transformer = CategoricalColumnSplitter(
            categorical_columns_to_split=["QCB1", "QCB2"]
        )
        df = pd.DataFrame(
            {
                "QCB1": [
                    "0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "D",
                    "R",
                    "V",
                    "S",
                    "A",
                    np.nan,
                ],
                "QCB2": [
                    "D",
                    "R",
                    "V",
                    "S",
                    "A",
                    "0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    np.nan,
                ],
            }
        )
        df_out = data_transformer.process(df)

        assert df_out["QCB1_num"].equals(
            pd.Series([0, 1, 2, 3, 4, 5, 6, 5, 6, 6, 0, 2, np.nan])
        )
        assert df_out["QCB2_num"].equals(
            pd.Series([5, 6, 6, 0, 2, 0, 1, 2, 3, 4, 5, 6, np.nan])
        )
        assert df_out["QCB1"].equals(
            pd.Series(
                [
                    np.nan,
                    np.nan,
                    np.nan,
                    "D",
                    "D",
                    "D",
                    "D",
                    "D",
                    "R",
                    "V",
                    "S",
                    "A",
                    np.nan,
                ]
            )
        )
        assert df_out["QCB2"].equals(
            pd.Series(
                [
                    "D",
                    "R",
                    "V",
                    "S",
                    "A",
                    np.nan,
                    np.nan,
                    np.nan,
                    "D",
                    "D",
                    "D",
                    "D",
                    np.nan,
                ]
            )
        )


class TestDataFrameColumnTypeSplitter(unittest.TestCase):
    def test_initialization(self):
        splitter = DataFrameColumnTypeSplitter()
        self.assertTrue(isinstance(splitter, DataFrameColumnTypeSplitter))

    def test_process(self):
        data = {
            'TextColumn1': ['1', 1, "2.25", np.nan, '789ghi'],
            'TextColumn2': ['1', 1, "2.25", np.nan, '789ghi']
        }

        df = pd.DataFrame(data)
        splitter = DataFrameColumnTypeSplitter()
        result = splitter.process(df)

        expected_columns = ['TextColumn1_num', 'TextColumn2_num','TextColumn1', 'TextColumn2']
        actual_columns = result.columns.tolist()

        print(result)
        self.assertListEqual(actual_columns, expected_columns)

        self.assertEqual(result['TextColumn1_num'].values[0], 1)
        self.assertEqual(result['TextColumn1_num'].values[1], 1)
        self.assertEqual(result['TextColumn1_num'].values[2], 2.25)
        # self.assertEqual(result['TextColumn1_num'].values[3], np.nan)
        # self.assertEqual(result['TextColumn1_num'].values[4], np.nan)

        self.assertEqual(result['TextColumn2_num'].values[0], 1.00)
        self.assertEqual(result['TextColumn2_num'].values[1], 1.00)
        self.assertEqual(result['TextColumn2_num'].values[2], 2.25)
        # self.assertEqual(result['TextColumn2_num'].values[3], np.nan)
        # self.assertEqual(result['TextColumn2_num'].values[4], np.nan)

        # self.assertEqual(result['TextColumn1'].values[0], np.nan)
        # self.assertEqual(result['TextColumn1'].values[1], np.nan)
        # self.assertEqual(result['TextColumn1'].values[2], np.nan)
        # self.assertEqual(result['TextColumn1_num'].values[3], np.nan)
        self.assertEqual(result['TextColumn1'].values[4], "789ghi")

        # self.assertEqual(result['TextColumn2'].values[0], np.nan)
        # self.assertEqual(result['TextColumn2'].values[1], np.nan)
        # self.assertEqual(result['TextColumn2'].values[2], np.nan)
        # self.assertEqual(result['TextColumn2'].values[3], np.nan)
        self.assertEqual(result['TextColumn2'].values[4], "789ghi")

        df = pd.DataFrame(data)
        splitter2 = DataFrameColumnTypeSplitter(only_process_columns=['TextColumn1'])
        result2 = splitter2.process(df)
        print(result2.columns)


        # Check if the DataFrame has the expected columns
        expected_columns = ['TextColumn1_num', 'TextColumn1', 'TextColumn2']
        self.assertListEqual(result2.columns.tolist(), expected_columns)


