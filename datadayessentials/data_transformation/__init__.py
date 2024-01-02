"""
Module for storing Transformers (functions that map from DataFrame -> DataFrame), and the DataFramePipe (which applies a sequence of transformers to a DataFrame). Many of the functions are not complex, but these modules are well tested and so there is less likely of mistakes when using these premade modules than implementing a new solution.

Below is a summary of what is available in this module, followed by an example use case for each of the transformers:

### Transformers
- ColumnRenamer: Renames columns based on a dictionary mapping
- DataFrameTimeSlicer: Slices (or filters) the DataFrame based on a date range and the date column name
- TierMapper: Map from one set of Scorecard tiers to another
- InvalidPayloadDropperByPrefix: Drops rows where there are NaN values in a subset of columns that begin with the prefix's provided
- ValueReplacer: Replace a set of unwanted values with a fill value across the entire DataFrame
- DominatedColumnDropper: Drop columns based on a dominance threshold (for instance more than 60% of the column is a single value)
- CatTypeConverter: Convert columns to categorical based on an input column list, and convert the rest to numerical type by defualt
- ColBasedQuantiler: Legacy Code
- MissingColumnReplacer: Given a set of expected columns, add additional columns that are missing and use a fill value


### other
- DataFramePipe: Apply a sequence of Transformers to a dataframe
"""
from ._transformers import (
    DataFrameTimeSlicer,
    ValueReplacer,
    DominatedColumnDropper,
    GranularColumnDropper,
    CategoricalColumnSplitter,
    DataFrameColumnTypeSplitter,
    CatTypeConverter,
    SimpleCatTypeConverter
)

from ._data_pipe import (
    DataFramePipe,
    run_pipeline_with_multi_threading,
)
from ._base import IDataFrameTransformer, IDataFramePipe


__all__ = [
    "CatTypeConverter",
    "DataFrameTimeSlicer",
    "DataFramePipe",
    "ValueReplacer",
    "DominatedColumnDropper",
    "GranularColumnDropper",
    "IDataFramePipe",
    "IDataFrameTransformer",
    "run_pipeline_with_multi_threading",
    "CategoricalColumnSplitter",
    "DataFrameColumnTypeSplitter",
    "InferenceSpeedCategoricalColumnSplitter",
    "SimpleCatTypeConverter"
]
