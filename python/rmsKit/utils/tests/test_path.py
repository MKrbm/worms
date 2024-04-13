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

    def test_find_summary_files(self):
        # Example test case for find_summary_files
        assert True

    def test_get_df_from_summary_files(self):
        # Example test case for get_df_from_summary_files
        # Assuming it returns a pandas DataFrame
        assert True

    def test_get_sim_result(self):
        # Example test case for get_sim_result
        # Assuming it returns a simulation result object or dict
        assert True


    def test_get_worm_path(self):
        # Example test case for get_worm_path
        # Assuming it returns a path string
        assert True