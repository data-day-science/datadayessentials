from unittest.mock import patch
from datadayessentials.jupyter_tools import TableScan


class TestTableScan:
    @patch('datadayessentials.jupyter_tools.TableLoader')
    def test_table_scan(self, mock_table_loader):
        mock_output = 'mock_output'
        mock_table_loader.return_value.load.return_value = mock_output

        TableScan('Prime')

        mock_table_loader.assert_called_once()

    def test_format_string_to_table_scan_query(self):
        string_to_search = 'Prime'
        expected_query = "SELECT DISTINCT TABLE_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE COLUMN_NAME LIKE '%Prime%'"

        assert TableScan.format_string_to_table_scan_query(string_to_search) == expected_query
