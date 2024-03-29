import re
from abc import ABC

from ._base import IDataFrameTransformer
import pandas as pd
import copy
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import logging
from datetime import datetime
import numpy as np
from typing import List, Any, Union, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PreprocessingError(Exception):
    """
    Error raised if there is an issue during an IDataTransformer process step
    """

    def __init__(
        self, step_name: str = "preprocessing", message: str = "preprocessing error"
    ):
        """Instantiates a preprocessing error, based on the step that it occurred and the error message

        Args:
            step_name (str, optional): Name of the step currently being applied. Defaults to "preprocessing".
            message (str, optional): Error message to raise. Defaults to "preprocessing error".
        """
        self.stepName = step_name
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        """Format the error as a string

        Returns:
            str: Error message
        """
        return "{}: {}".format(self.stepName, self.message)


def is_data_size_small(data_object: Union[pd.DataFrame, pd.Series]) -> bool:
    """
        Checks if the size of the data is smaller than 100million and returns a boolean

        Args:
            data_object (pd.DataFrame): Dataframe to check the size of
        returns:
            bool: True if the size less than 100,000,000, False otherwise

        Example:
            is_data_size_small(pd.DataFrame({'col1': [1, 2, 3], 'col2': [1, 2, 3]}))
    >>> True
            is_data_size_small(pd.DataFrame(np.random.randint(0,100,size=(110000, 1000)), columns=list(range(1000))))
    >>> False

    """

    return True if data_object.size < 100000000 else False


class DataFrameTimeSlicer(IDataFrameTransformer):
    """
    Accepts a dataframe with a datetime column, a min time and max time.  The class with slice the dataframe and return
    a new dataframe in the date range

    Typical usage example:
    ```python
    from datascinece_core.data_transformation import DataFrameTimeSlicer

    input_df = pd.DataFrame({'date': [date1, date2, date3], 'col2': [1, 2, 3]})

    start_time = datetime.strptime('2022-01-01', '%Y-%m-%d')
    end_time = datetime.strptime('2022-02-01', '%Y-%m-%d')
    date_col = 'date'

    col_slicer = DataFrameTimeSlicer(date_col, start_time, end_time)
    sliced_df = col_slicer.process(input_data)
    ```

    """

    # date range inclusive (>=, <=)
    def __init__(
        self,
        col_name_for_time: str,
        min_time: datetime,
        max_time: datetime,
        convert_to_datetime_format: str = "",
    ):
        """Instantiate a DataFrameTimeSlicer

        Args:
            col_name_for_time (str): The column name that contains the timestamp to use for slicing
            min_time (datetime): The minimum time for rows to pass through the filter
            max_time (datetime): The maximum time for rows to pass throught the filter
            convert_to_datetime_format (bool, optional): If set, use this format string to convert the col_name_for_time column into a datetime dtype.

        Raises:
            ValueError: Raised if the time ranges are not datetime objects
        """
        self.col_name_for_time = col_name_for_time
        if not isinstance(min_time, datetime):
            raise ValueError("Argument min_time must be a datetime object")
        if not isinstance(max_time, datetime):
            raise ValueError("Argument max_time must be a datetime object")
        self.convert_to_datetime_format = convert_to_datetime_format
        self.min_time = min_time
        self.max_time = max_time
        logger.debug(
            f"Creating a DataFrameTimeSlicer on column {col_name_for_time} between {self.min_time} and {self.max_time}"
        )

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the time slicing transformer

        Args:
            data (pd.DataFrame): Data to be sliced

        Returns:
            pd.DataFrame: Time Sliced dataframe
        """
        if self.convert_to_datetime_format:
            logger.debug(f"Converting to datetime")
            data[self.col_name_for_time] = pd.to_datetime(
                data[self.col_name_for_time], format=self.convert_to_datetime_format
            )
            logger.debug(
                f"After conversion the dtype of the column is: {data[self.col_name_for_time].dtype}"
            )
            logger.debug(
                f"THe month of the first date in the data is {pd.DatetimeIndex(data[self.col_name_for_time]).month}"
            )
        logger.debug("DataFrameTimeSlicer is procesing")

        data_max_date = data[self.col_name_for_time].max()
        data_min_date = data[self.col_name_for_time].min()
        logger.debug(f"The max date in the data is {data_max_date}")
        logger.debug(f"The min date in the data is {data_min_date}")
        if self.max_time > data_max_date:
            logger.warn(
                "Input maximum date is greater than the maximum date within the dataframe"
            )
        if self.min_time < data_min_date:
            logger.warn(
                "Input minimum date is less than the minimum date within the dataframe"
            )
        logger.debug(f"The data at this point is {data}")
        return data[
            (data[self.col_name_for_time] >= self.min_time)
            & (data[self.col_name_for_time] <= self.max_time)
        ]


class ValueReplacer(IDataFrameTransformer):
    """
    Replaces values with NaN that are either missing data, mistakes or outliers in the credit-check payloads
    Example Use Case:
    ```python
    from datadayessentials.data_transformation import ValueReplacer

    value_replacer = ValueReplacer(unwanted_values=['bad_value_1', 'bad_value_2'], replacement_value=0)

    input_df = pd.DataFrame({'col_name': ['bad_value_2', 1, 2, 3, 'bad_value_1']})
    output_df = value_replacer.process(input_df)
    ```
    """

    def __init__(
        self,
        unwanted_values: List = [
            "M",
            "C",
            "{ND}",
            "ND",
            "OB",
            "Not Found",
            "{OB}",
            "T",
            "__",
            -999997,
            -999999,
            999999,
            999997,
            -999997.0,
            -999999.0,
            999999.0,
            999997.0,
            "-999997",
            "-999999",
            "999999",
            "999997",
            "-999997.0",
            "-999999.0",
            "999999.0",
            "999997.0",
        ],
        replacement_value: Any = np.nan,
    ):
        """Instantiate the ValueReplacer

        Args:
            unwanted_values (List, optional): Values to replace. Defaults to [ "M", "C", "{ND}", "ND", "OB", "Not Found", "{OB}", "T", "__", -999997, -999999, 999999, 999997, -999997.0, -999999.0, 999999.0, 999997.0, "-999997", "-999999", "999999", "999997", "-999997.0", "-999999.0", "999999.0", "999997.0", ].
            replacement_value (Any, optional): Value to replace with. Defaults to np.nan.
        """
        self.unwanted_values = unwanted_values
        self.replacement_value = replacement_value

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the value replacer, replacing the unwanted values with NaN in the entire dataframe

        Args:
            data (pd.DataFrame): DataFrame to replace values in

        Raises:
            PreprocessingError: Raised if there is any issue during processing

        Returns:
            pd.DataFrame: Dataframe with values replaced
        """
        try:
            return data.replace(self.unwanted_values, self.replacement_value)
        except Exception as err:
            raise PreprocessingError(type(self).__name__, err)


class DominatedColumnDropper(IDataFrameTransformer):
    """
    Drops columns based on a dominance threshold. For instance if the threshold is 0.6 then columns with more than 60% of the vales that are the same are dropped.
    Example Use Case:
    ```python
    from datadayessentials.data_transformation import DominatedColumnDropper

    dom_threshold = 0.6
    ignore_cols = ['ignore_this']

    dom_col_dropper = DominatedColumnDropper(dominance_threshold=dom_threshold, ignore_cols=ignore_cols)

    input_df = pd.DataFrame({
        'above_thresh': [1, 1, 1, 2],
        'below_thresh': [1, 1, 2, 2],
        'ignore_this': [1, 1, 1, 1]
    })
    output_df = dom_col_dropper.process(input_df)
    # Output DF will only drop the 'above thresh' column, as more than 60% of the values are the same (1)
    ```
    """

    def __init__(self, dominance_threshold: float = 0.99, ignore_cols: List[str] = []):
        """Instantiate the column dropper

        Args:
            dominance_threshold (float, optional): Threshold to drop columns if the dominance is higher than it. Defaults to 0.99.
            ignore_cols (List[str], optional): Any columns to exclude from the dominance check. Defaults to [].
        """
        self.dominance_threshold = dominance_threshold
        self.ignore_cols = ignore_cols

    def process(self, data: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """Apply the column dropper and retun a dataframe without domicated columns

        Args:
            data (pd.DataFrame): Dataframe to apply dominance threshold to
            verbose (bool): If true, enable additional logging, with print statements

        Raises:
            PreprocessingError: Raised if there is any error raised during the processing

        Returns:
            pd.DataFrame: Pandas dataframe with columns dropped that are above the dominance threshold
        """
        try:
            df_out = copy.deepcopy(data)
            if verbose:
                print("start to remove the columns with the identical values...")
            cols_to_check = [x for x in df_out.columns if x not in self.ignore_cols]
            col_to_remove = []
            for col in cols_to_check:
                nan_sum = df_out[col].isna().sum()
                if nan_sum / df_out.shape[0] >= self.dominance_threshold:
                    col_to_remove.append(col)
                    continue
                if (
                    df_out[col].value_counts().iloc[0] / df_out.shape[0]
                    >= self.dominance_threshold
                ):
                    col_to_remove.append(col)

            if verbose:
                print(
                    """total number of columns in the input data frame is {}
                number of columns to be removed is {}
                number of remaining columns is {}""".format(
                        data.shape[1],
                        len(col_to_remove),
                        data.shape[1] - len(col_to_remove),
                    )
                )
            df_out.drop(col_to_remove, axis=1, inplace=True)
            if verbose:
                print("finished removing columns that have only one value")
        except Exception as err:
            raise PreprocessingError(type(self).__name__, err)
        return df_out


class GranularColumnDropper(IDataFrameTransformer):
    """Drops columns that have to many categorical values above a threshold.
    Example Use Case:
    ```python

    from datadayessentials.data_transformation import GranularColumnDropper

    granular_dropper = GranularColumnDropper(threshold=0.6, list_of_cols=['col1', 'col2'])
    input_df = pd.DataFrame({
        'col1': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'],
        'col2': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'],
        })
    output_df = granular_dropper.process(input_df)
    # output_df will have 'col1' dropped because it has more than 60% unique values
    # output_df will not have 'col2' dropped because it has less than 60% unique values
    ```
    """

    def __init__(self, threshold: float = 0.6, list_of_cols: List[str] = []):
        """transformer class to drop a column if the number of unique values in a column exceeds a threshold for total columns


        Args:
            threshold (float, optional): threshold for dropping. Defaults to 0.6.
            list_of_cols (List[str], optional): list of columns to evaluate. Defaults to [].
             If empty, all columns will be evaluated
        """
        self.threshold = threshold
        self.list_of_cols = list_of_cols

    def process(self, df: pd.DataFrame, create_copy=False) -> pd.DataFrame:
        """run function for the transformer.  Can be used by a datadayessentials IDataFrameTransformer

        Args:
            df (pd.DataFrame): input data to process
            create_copy (bool, optional): if true, return a copy of the dataframe. Defaults to False.

        Raises:
            PreprocessingError: raises an error if the process fails

        Returns:
            pd.DataFrame: processed dataframe
        """

        try:
            if create_copy:
                if not is_data_size_small(df):
                    print(
                        "Data Size is too large to copy. Applying transformation on original dataframe"
                    )
                    df_copy = df
                else:
                    df_copy = copy.deepcopy(df)
            else:
                df_copy = df

            cols_to_drop = []

            if len(self.list_of_cols) > 0:
                columns_to_evaluate = list(
                    set(self.list_of_cols).intersection(set(df_copy.columns))
                )
                self.missing_column_warning(columns_to_evaluate)
            else:
                columns_to_evaluate = df_copy.columns

            for col in columns_to_evaluate:
                if len(df_copy[col].unique()) / df_copy.shape[0] > self.threshold:
                    cols_to_drop.append(col)

            return df_copy.drop(columns=cols_to_drop)
        except Exception as err:
            raise PreprocessingError(type(self).__name__, err)

    def missing_column_warning(self, columns_to_evaluate):
        if len(columns_to_evaluate) < len(self.list_of_cols):
            print("Some of the columns requested are not in the dataframe")
            logger.warn("Some of the columns requested are not in the dataframe")


class CategoricalColumnSplitter(IDataFrameTransformer):

    #The series mapper fails to remove series and remove floats from the categorical columns
    numerical_mapping = {"D": 5, "R": 6, "V": 6, "S": 0, "A": 2}

    cat_mapping = {
        0: np.nan,
        1: np.nan,
        2: np.nan,
        3: "D",
        4: "D",
        5: "D",
        6: "D",
        "0": np.nan,
        "1": np.nan,
        "2": np.nan,
        "3": "D",
        "4": "D",
        "5": "D",
        "6": "D",
    }

    @staticmethod
    def map_with_type(wanted_type: Union["number","string"], value: Any, mapping: dict):

        """
        Convert the given value to the wanted type using a mapping dictionary.
        If the conversion is not possible, returns np.nan.

        :param wanted_type: The type to which the value should be converted.
        :param value: The value to be converted.
        :param mapping: A dictionary for converting the value if direct conversion is not possible.
        :return: The converted value or np.nan if conversion is not possible.
        """

        if isinstance(value, (str, int, float)):
            if wanted_type == "number":
                try:
                    value = float(value)
                    if isinstance(value, (int, float)):
                        return mapping[value] if value in mapping else value
                    return mapping.get(value, np.nan)
                except ValueError:
                    if isinstance(value, (int, float)):
                        return mapping[value] if value in mapping else value
                    return mapping.get(value, np.nan)

            if wanted_type == "string":
                if isinstance(value, str):
                    return mapping[value] if value in mapping else value
                return mapping.get(value, np.nan)

        else:
            return np.nan

    def __init__(self, categorical_columns_to_split: list):
        self.categorical_columns_to_split = categorical_columns_to_split

    def process(self, data: Union[pd.DataFrame, pd.Series]):
        if isinstance(data, pd.DataFrame):
            return self.process_df(data)
        elif isinstance(data, pd.Series):
            return self.process_series(data)
        else:
            raise TypeError(f"Expected DataFrame or Series, got {type(data)}")

    def dataframe_split_categorical_column(
        self, series: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:

        numerical_series = series.map(
            lambda x: CategoricalColumnSplitter.map_with_type("number", x, self.numerical_mapping)
        ).astype("float64")

        cat_series = series.copy(deep=True)

        cat_series = cat_series.map(
            lambda x: CategoricalColumnSplitter.map_with_type("string", x, self.cat_mapping)
        ).astype("object")

        return cat_series, numerical_series

    def process_df(self, df_in: pd.DataFrame) -> pd.DataFrame:
        for column in self.categorical_columns_to_split:
            if column in df_in.columns:
                cat_series, numerical_series = self.dataframe_split_categorical_column(
                    df_in[column]
                )

                # Update DataFrame in place
                df_in[f"{column}"] = cat_series
                df_in[f"{column}_num"] = numerical_series
        return df_in

    def create_numeric_columns(self, series):
        series_copy = series.copy(deep=True)[self.categorical_columns_to_split]
        series_copy = series_copy.map(self.numerical_mapping).astype("float64")
        series_copy.index = series_copy.index + "_num"
        return series_copy

    def create_categorical_columns(self, series):
        series_copy = series.copy(deep=True)[self.categorical_columns_to_split]
        series_copy = series_copy.map(self.cat_mapping)
        return series_copy

    def process_series(self, series_to_split: pd.Series) -> pd.Series:
        # Filter out the non-categorical series
        non_categorical_series = series_to_split.drop(
            self.categorical_columns_to_split, errors="ignore"
        )

        # Apply categorical mapping to the relevant subset. Floats are not removed from this
        categorical_series = series_to_split[self.categorical_columns_to_split].replace(
            self.cat_mapping
        )
        # Apply numerical mapping to the same subset and adjust index
        numerical_series = pd.to_numeric(
            series_to_split[self.categorical_columns_to_split].replace(
                self.numerical_mapping
            ),
            errors="coerce",
        )
        numerical_series.index = [f"{idx}_num" for idx in numerical_series.index]

        # Concatenate all series
        output_series = pd.concat(
            [non_categorical_series, categorical_series, numerical_series]
        )
        return output_series


class DataFrameColumnTypeSplitter(IDataFrameTransformer):
    def __init__(self, only_process_columns: list = None):
        self.columns_to_process = None
        if only_process_columns:
            self.columns_to_process = only_process_columns

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.columns_to_process:
            self.columns_to_process = data.columns

        data_to_transform_as_str = data[self.columns_to_process].astype(str)
        nums = data_to_transform_as_str.apply(
            lambda col: pd.to_numeric(col, errors="coerce")
        )
        strings = pd.DataFrame(
            np.where(~nums.notna(), data_to_transform_as_str, np.nan)
        )

        nums.columns = [f"{col}_num" for col in data_to_transform_as_str.columns]
        strings.columns = [col for col in data_to_transform_as_str.columns]
        untransformed_columns = [
            item for item in data.columns if item not in self.columns_to_process
        ]

        nums.reset_index(drop=True, inplace=True)
        strings.reset_index(drop=True, inplace=True)

        return pd.concat(
            [nums, strings, data[untransformed_columns].reset_index(drop=True)], axis=1
        )


class CatTypeConverter(IDataFrameTransformer):
    """
    Converts the type of specified columns to category and the rest to numeric.
    This worked for speedy card (one column dataframe) and normal card
    Example Use Case:
    ```python
    from datadayessentials.data_transformation import CatTypeConverter

    categorical_columns = ['shape', 'color']
    cat_converter = CatTypeConverter(categorical_columns)

    input_df = pd.DataFrame({
        'shape': ['square', 'circle', 'oval'],
        'color': ['brown', 'brown', 'white'],
        'size': ['5', '5', '6']
    })
    output_df = cat_converter.process(input_df)
    # output_df will have categorical dtypes for 'shape' and 'color' and 'size' will be converted into float type
    ```

    """

    def __init__(self, cat_col_names: List[str] = [], date_col_names: List[str] = []):
        """Instantiate a CatTypeConverter

        Args:
            cat_col_names (List[str], optional): the column names to be converted to category type. Defaults to [].
            date_col_names (List[str], optional): The names of the date columns (and not to convert)
        """
        self.cat_col_names = cat_col_names
        self.date_col_names = date_col_names

    def process(
        self, df: pd.DataFrame, verbose: bool = False, create_copy=False
    ) -> pd.DataFrame:
        """Apply the column conversion returning a new dataframe

        Args:
            df (pd.DataFrame): input dataframe to convert to categorical type
            verbose (bool, optional): Enable additional logging if enabled. Defaults to False.
            create_copy (bool, optional): If true, return a copy of the dataframe. Defaults to False.

        Raises:
            PreprocessingError: Raised if there is any issue during applying the processing

        Returns:
            pd.DataFrame: Converted dataframe

        """
        try:
            if create_copy:
                if not is_data_size_small(df):
                    print(
                        "Data Size is too large to copy. Applying transformation on original dataframe"
                    )
                    df_out = df
                else:
                    df_out = copy.deepcopy(df)
            else:
                df_out = df

            # We still want the function to complete in the case when no categorical columns are passed.
            if len(self.cat_col_names) == 0:
                if "0" in df_out.columns:
                    df_out["0"] = pd.to_numeric(df_out["0"], errors="coerce")
                else:
                    df_out = df_out.apply(pd.to_numeric, errors="coerce")
                return df_out

            if set(self.cat_col_names).issubset(set(df_out.index)):
                df_cat = copy.deepcopy(df_out.loc[self.cat_col_names])
                df_cat.fillna("NaN", inplace=True)
                df_cat["0"] = df_cat["0"].astype("str")
                df_cat["0"] = df_cat["0"].astype("category")

                df_dates = df_out.loc[self.date_col_names]

                df_out.drop(self.cat_col_names, inplace=True)
                df_out.drop(self.date_col_names, inplace=True)

                df_out["0"] = pd.to_numeric(df_out["0"], errors="coerce")

                df_out = pd.concat([df_out.T, df_cat.T, df_dates.T], axis=1)

                if verbose:
                    print(
                        "{} out of {} features changed to category type and other {} to numeric".format(
                            len(df_cat.index), len(df.index), len(df_out.index)
                        )
                    )
            elif set(self.cat_col_names).issubset(set(df_out.columns)):
                df_cat = copy.deepcopy(df_out[self.cat_col_names])
                df_cat.fillna("NaN", inplace=True)
                df_cat = df_cat.astype("str")
                df_cat = df_cat.astype("category")
                df_out.drop(columns=self.cat_col_names, inplace=True)
                numeric_cols = [
                    col for col in df_out.columns if col not in self.date_col_names
                ]
                df_out[numeric_cols] = df_out[numeric_cols].apply(
                    lambda x: pd.to_numeric(x, errors="coerce")
                    if x.name not in self.cat_col_names
                    else x
                )
                df_out = pd.concat([df_out, df_cat], axis=1)
                if verbose:
                    print(
                        "{} out of {} features changed to category type and other {} to numeric".format(
                            len(df_cat.columns), len(df.columns), len(df_out.columns)
                        )
                    )
            else:
                msg = "some of the requested feature names are not included in the dataframe"
                raise PreprocessingError(type(self).__name__, msg)
        except Exception as err:
            raise PreprocessingError(type(self).__name__, err)
        return df_out


class SimpleCatTypeConverter:
    def __init__(self, categorical_columns, date_columns=[]):
        self.categorical_columns = categorical_columns
        self.date_columns = date_columns

    def process(self, data: Union[pd.DataFrame, pd.Series]):
        if isinstance(data, pd.DataFrame):
            return self.process_df(data)
        elif isinstance(data, pd.Series):
            return self.process_series(data)
        else:
            raise TypeError(f"Expected DataFrame or Series, got {type(data)}")

    def process_df(self, df):
        # Filter columns present in the DataFrame
        validated_categorical_columns = [
            col for col in self.categorical_columns if col in df.columns
        ]
        validated_date_columns = [col for col in self.date_columns if col in df.columns]

        # Convert to categorical
        for col in validated_categorical_columns:
            if df[col].dtype != "category":
                df[col] = df[col].astype("category")

        # Convert non-date and non-categorical columns to numeric
        non_categorical_columns = df.columns.difference(
            validated_categorical_columns + validated_date_columns
        )
        for col in non_categorical_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def process_series(self, series):
        # Process a single transposed row (Pandas Series)
        for col in series.index:
            # Convert to categorical if in categorical_columns
            if (
                col in self.categorical_columns
                and not pd.api.types.is_categorical_dtype(series[col])
            ):
                series[col] = pd.Categorical([series[col]])

            # Convert to numeric if not in categorical_columns or date_columns
            elif col not in self.date_columns and not pd.api.types.is_numeric_dtype(
                series[col]
            ):
                series[col] = pd.to_numeric(series[col], errors="coerce")

        return series
