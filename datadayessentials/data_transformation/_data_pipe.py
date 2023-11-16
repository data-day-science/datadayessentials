from ._base import IDataFramePipe, IDataFrameTransformer


from typing import List, Tuple
import pandas as pd
import multiprocessing as mp
from functools import partial


class DataFramePipe(IDataFramePipe):
    """
    Class for applying multiple IDataFrameTransformers in one go to an input pandas DataFrame. Outputting a pandas DataFrame

    Example Use Case:
    ```python
    from datadayessentials.data_transformation import DataFramePipe, ColumnRenamer, ValueReplacer

    value_replacer = ValueReplacer(unwanted_values=['bad_value_1', 'bad_value_2'], replacement_value=0)
    column_renamer = ColumnRenamer({'old_name': 'new_name'})
    pipe = DataFramePipe([value_replacer, column_renamer])

    input_df = pd.DataFrame({'old_name': ['bad_value_2', 1, 2, 3, 'bad_value_1']})
    output_df = pipe.run(input_df)
    # output_df should look like: {'new_name': [0, 1, 2, 3, 0]}
    ```
    """

    def __init__(self, transformers: List[IDataFrameTransformer]):
        self.transformers = transformers

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sequentially apply the list of IDataTransformers to the input `df`

        Args:
            df (pd.DataFrame): Dataframe to apply transformations to

        Returns:
            pd.DataFrame: output dataframe
        """
        for transformer in self.transformers:
            df = transformer.process(df)
        return df





def run_pipeline_with_multi_threading(pre_processor,
                                      data_frame: pd.DataFrame,
                                      unwanted_features: list,
                                      required_features: list,
                                      categorical_features: list,
                                      target_feature: str,
                                      n_processes: int,
                                      n_splits: int):
    """
    Runs a preprocessor via multithreading on a dataframe.
    Takes a dataframe and splits it into n_splits slices. Each slice is then run through the preprocessor in parallel.
    The results are then concatenated back together.

    args:
        pre_processor: preprocessor object to run
        data_frame: dataframe to run the preprocessor on
        unwanted_features: list of unwanted features
        required_features: list of required features
        categorical_features: list of categorical features
        target_feature: name of target feature
        n_processes: number of processes to run
        n_splits: number of splits to create

    returns:
        dataframe: processed dataframe
    """

    feature_combinations = _get_nth_columns_at_increasing_indexes(dataframe=data_frame, n_splits=n_splits)
    feature_combinations_with_target = _append_target_in_each_list_without_it(feature_combinations,
                                                                              target=target_feature)
    dataframe_slices = [data_frame[feature_combination] for feature_combination in feature_combinations_with_target]
    del data_frame

    flexible_pre_processor = partial(pre_processor.run,
                                     unwanted_features=unwanted_features,
                                     required_features=required_features,
                                     categorical_features=categorical_features,
                                     target=target_feature,
                                     verbose=False)

    processed_data = _perform_computation(flexible_pre_processor, dataframe_slices, n_processes)

    return _clean_up_processed_data(processed_data, target_feature)


def _clean_up_processed_data(processed_data, target: str = None):
    """
    Takes a processed data from a multithreading process and concatenates it back together

    args:
        processed_data: list of dataframes
        target: name of target column

    returns:
        dataframe: concatenated dataframe
    """
    result_slices = [output[0] for output in processed_data]
    results = pd.concat(result_slices, axis=1)
    popped_target = results.pop(target)
    results[target] = popped_target.iloc[:, 0]

    return results


def _get_nth_columns_at_increasing_indexes(dataframe: pd.DataFrame, n_splits: int) -> list:
    """
    Returns a list of lists of columns at increasing indexes.  This is used to split up the dataframe into
    n_splits number of chunks for parallel processing.

    args:
        dataframe: dataframe to split
        n_splits: number of splits to create

    returns:
        list: list of lists of columns

    Example:

    df = pd.DataFrame({

    'A': [1, 2, 3, 4, 5],

    'B': [6, 7, 8, 9, 10],

    'C': [11, 12, 13, 14, 15],

    'D': [16, 17, 18, 19, 20],

    'E': [21, 22, 23, 24, 25],

    'F': [1, 2, 3, 4, 5],

    'G': [6, 7, 8, 9, 10],

    'H': [11, 12, 13, 14, 15],

    'I': [16, 17, 18, 19, 20],

    'J': [21, 22, 23, 24, 25]
    })

    get_nth_columns_at_increasing_indexes(df, 3)
    returns [['A', 'D', 'G', 'J'],['B', 'E', 'H'],['C', 'F', 'I']]
    """

    column_names = dataframe.columns.tolist()
    return [column_names[index::n_splits] for index in range(n_splits)]


def _append_target_in_each_list_without_it(columns: List[List[any]], target: str) -> List[List[any]]:
    """
    Update the columns by adding the target element if it is not already present.

    args:
        columns: A list of lists representing the columns.
        target: The element to add to each column if it is not already present.

    returns:
        The updated list of columns with the target element added if not present.
    """
    updated_columns = []

    for columns_to_slice in columns:
        if target not in columns_to_slice:
            updated_columns.append(columns_to_slice + [target])
        else:
            updated_columns.append(columns_to_slice)

    return updated_columns


def _perform_computation(worker_functions, dataframe_slices, n_processes):
    """
    Performs the computation in parallel using the worker functions and dataframe slices.

    args:
        worker_functions: a function to run on each dataframe slice
        dataframe_slices: a list of dataframe slices
        n_processes: number of processes to run

    returns:
        list: list of results from the worker functions


    """
    with mp.Pool(processes=n_processes) as working_computation_pool:
        results = working_computation_pool.map(worker_functions, dataframe_slices)
    return results
