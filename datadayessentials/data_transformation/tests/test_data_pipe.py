import unittest

from datadayessentials.data_transformation import GranularColumnDropper, CreditDataPreProcessor

from .._transformers import (
    ColumnRenamer,
)
from .._data_pipe import DataFramePipe, _get_nth_columns_at_increasing_indexes, _append_target_in_each_list_without_it, \
    _clean_up_processed_data, run_pipeline_with_multi_threading
import pandas as pd
from unittest.mock import MagicMock


class TestDataPipe:
    def test_run(self):
        mock_transformer_1 = ColumnRenamer({})
        mock_transformer_2 = ColumnRenamer({})
        mock_transformer_1.process = MagicMock(return_value=pd.DataFrame())
        mock_transformer_2.process = MagicMock(return_value=pd.DataFrame())
        transformers = [mock_transformer_1, mock_transformer_2]
        data_pipe = DataFramePipe(transformers)
        data_pipe.run(pd.DataFrame())
        assert mock_transformer_1.process.called
        assert mock_transformer_2.process.called


class TestRunPipelineWithMultiThreading(unittest.TestCase):

    def test_get_nth_columns_at_increasing_indexes(self):
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
            'J': [21, 22, 23, 24, 25]})

        nth_columns = _get_nth_columns_at_increasing_indexes(df, 3)
        assert nth_columns == [['A', 'D', 'G', 'J'], ['B', 'E', 'H'], ['C', 'F', 'I']]

    def test_append_target_in_each_list_without_it(self):
        test_feature_combinations = [['A', 'D', 'G', 'J'], ['B', 'E', 'H'], ['C', 'F', 'I']]
        test_target = 'A'
        updated_feature_combinations = _append_target_in_each_list_without_it(test_feature_combinations, test_target)
        assert updated_feature_combinations == [['A', 'D', 'G', 'J'], ['B', 'E', 'H', 'A'], ['C', 'F', 'I', 'A']]

    def test_clean_up_processed_data(self):
        # Create sample processed data
        processed_data = [
            (pd.DataFrame({'A': [1, 2, 3]}),),
            (pd.DataFrame({'B': [4, 5, 6]}),),
            (pd.DataFrame({'C': [7, 8, 9]}),),
            (pd.DataFrame({'D': [1, 2, 3]}),),
            (pd.DataFrame({'E': [4, 5, 6]}),),
            (pd.DataFrame({'C': [7, 8, 9]}),)
        ]
        target = 'C'

        cleaned_data = _clean_up_processed_data(processed_data, target)
        print(cleaned_data)
        expected_result = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'D': [1, 2, 3], 'E': [4, 5, 6], 'C': [7, 8, 9]})
        pd.testing.assert_frame_equal(cleaned_data, expected_result)

    def test_append_target_in_each_list_without_it(self):
        step1 = GranularColumnDropper(threshold=0.5, list_of_cols=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])
        pipe = DataFramePipe(transformers=[step1])
        pre_processor = CreditDataPreProcessor(pipe=pipe)

        df = pd.DataFrame({
            'A': ["Car", "Finance", "Car", "Car", "Car"],
            'B': ["Car", "Finance", "Is", "So", "Amazing"],
            'C': ["Car", "Finance", "Car", "Car", "Car"],
            'D': ["Car", "Finance", "Car", "Car", "Car"],
            'E': ["Car", "Finance", "Car", "Car", "Car"],
            'F': ["Car", "Finance", "Car", "Car", "Car"],
            'G': ["Car", "Finance", "Car", "Car", "Car"],
            'H': ["Car", "Finance", "Car", "Car", "Car"],
            'I': ["Car", "Finance", "Car", "Car", "Car"],
            'J': ["Car", "Car", "Car", "Car", "Car"],
        })

        pipe = DataFramePipe(transformers=[step1])
        pre_processor = CreditDataPreProcessor(pipe=pipe)

        unwanted_features = ["C"]
        required_features = ["A", "B", "D", "E", "F", "G", "H", "I", "J"]
        categorical_features = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        target_feature = 'A'
        n_processes = 2
        n_splits = 3

        result = run_pipeline_with_multi_threading(pre_processor, df, unwanted_features, required_features,
                                                   categorical_features, target_feature, n_processes, n_splits)

        assert "C" not in result.columns
        assert "B" not in result.columns
        assert all(x in result.columns for x in ["A", "D", "E", "F", "G", "H", "I", "J"])

