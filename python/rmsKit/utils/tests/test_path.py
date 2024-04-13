import pytest
from ..path import (
    extract_to_rmskit,
    find_info_txt_files,
    find_summary_files,
    get_df_from_summary_files,
    get_sim_result,
    extract_info_from_txt,
    get_worm_path
)
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestPathUtils:
    def test_extract_to_rmskit_combined(self):
        # Test with a valid rmsKit path
        file_path = Path("python/rmsKit/utils/tests/test_path.py")
        result = extract_to_rmskit(file_path)
        assert result.as_posix().endswith('rmsKit'), "The extracted path does not end with 'rmsKit'"

        # Test with an invalid rmsKit path
        with pytest.raises(ValueError) as e:
            non_rmskit_path = Path('/path/without/rmsKit')
            extract_to_rmskit(non_rmskit_path)
        assert "The given path is not under 'rmsKit'." in str(e.value)

    def test_find_info_txt_files(self):
        file = Path(__file__).parent / "example"
        results = find_info_txt_files(file)
        result_path = results[0].resolve()
        assert result_path == (file / "info.txt").resolve(), "The found info.txt file does not match the expected file."

    def test_extract_info_from_txt(self):
        # Path to the info.txt file for testing
        file_path = Path("python/rmsKit/utils/tests/example/info.txt")
        
        # Expected result based on the content of info.txt
        expected_result = {
            "best_loss": 2.814614930013448e-10,
            "initial_loss": 4.102281093597412,
            "hamiltonian_path": Path("/home/keisuke/worms/python/rmsKit/array/torch/FF1D_loc/s_3_r_2_d_1_seed_3000/1_stoq/H"),
            "unitary_path": Path("/home/keisuke/worms/python/rmsKit/array/torch/FF1D_loc/s_3_r_2_d_1_seed_3000/1_stoq/Adam/lr_0.0003_epoch_20000/loss_0.0000000/u")
        }

        # Call the function under test
        result = extract_info_from_txt(file_path)

        # Assert that the result matches the expected result
        assert result["best_loss"] == expected_result["best_loss"]
        assert result["initial_loss"] == expected_result["initial_loss"]
        assert result["hamiltonian_path"] == expected_result["hamiltonian_path"]
        assert result["unitary_path"] == expected_result["unitary_path"]

    def test_find_summary_files_and_get_df(self):
        # Test with a valid directory containing summary files
        directory_path = Path("python/rmsKit/utils/tests/example/summary1")
        summary_files = find_summary_files(directory_path)
        assert len(summary_files) > 0, "No summary files found in the directory"
        for res_dict in summary_files:
            assert "summary" in res_dict and "info" in res_dict, "Missing 'summary' or 'info' key in the result dictionary"
        
        # Test with an invalid directory path
        with pytest.raises(ValueError) as e:
            invalid_directory = Path("nonexistent/directory")
            find_summary_files(invalid_directory)
        assert "The given path" in str(e.value) and "is not a directory" in str(e.value)

        # Use the results of find_summary_files as the input for get_df_from_summary_files
        N = 1000000 
        df = get_df_from_summary_files(summary_files, N)

        # Assert the expected properties of the returned DataFrame
        assert isinstance(df, pd.DataFrame), "The returned object is not a pandas DataFrame"
        assert "init_loss" in df.columns, "The 'init_loss' column is missing in the DataFrame"
        assert "ham_path" in df.columns, "The 'ham_path' column is missing in the DataFrame"
        assert len(df) > 0, "The DataFrame is empty"

        print(df.columns)

    def test_get_sim_result(self):
        # Prepare test data
        directory_path = Path("python/rmsKit/utils/tests/example/summary1")
        N = 1000000

        # Call the function under test
        df = get_sim_result(directory_path, N)

        # Assert the expected properties of the returned DataFrame
        assert isinstance(df, pd.DataFrame), "The returned object is not a pandas DataFrame"
        assert len(df) > 0, "The DataFrame is empty"


    def test_get_worm_path(self):
        # Example test case for get_worm_path
        # Assuming it returns a path string
        assert True