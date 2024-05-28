import subprocess
import os
import pytest
import logging
import re
import numpy as np

from ..utils.logdata_handler import parse_log_file

class TestLogDataHandler:

    @pytest.fixture(scope="class")
    def log_file_path(self):
        
        # First check the file exists
        file_path = "optimize_loc.py"
        assert os.path.exists(file_path), f"{file_path} does not exist"

        # Set the environment variable for the RMSKIT_PATH
        env = os.environ.copy()

        # Run the optimize_loc.py script
        result = subprocess.run(
            f"python {file_path} -m FF1D --sps 3 --seed 23224 --loss mel -M 10 -e 20 --dtype complex128",
            text=True,
            shell=True,
            capture_output=True,
            env=env
        )

        logging.debug(result)

        # Check if the script ran successfully
        assert result.returncode == 0, f"Script failed with error: {result.stderr}"

        # Extract the log file path from the script output
        log_file_path = None
        logging.debug(result.stdout)
        for line in result.stdout.splitlines():
            logging.debug(line)
            if "Log file" in line:
                log_file_path = line.split("Log file will be saved to ")[1].strip()
                break
        assert log_file_path is not None, "Log file path not found in script output"
        assert os.path.exists(log_file_path), f"Log file does not exist: {log_file_path}"

        return log_file_path


    def test_parse_log_file(self, log_file_path):
        # Parse the log file
        df, meta = parse_log_file(log_file_path)

        # Perform some basic checks on the DataFrame
        assert not df.empty, "DataFrame is empty"
        assert 'Iteration' in df.columns, "Column 'Iteration' not found in DataFrame"
        assert 'Loss at Epoch Start' in df.columns, "Column 'Loss at Epoch Start' not found in DataFrame"
        assert 'Best Loss at Iteration' in df.columns, "Column 'Best Loss at Iteration' not found in DataFrame"
        assert 'Saved Path' in df.columns, "Column 'Saved Path' not found in DataFrame"

        assert meta.dtype == "complex128", f"Expected dtype 'complex128', got '{meta.dtype}'"
        assert meta.seed == 23224, f"Expected seed '23224', got '{meta.seed}'"
        assert meta.model == "FF1D", f"Expected model 'FF1D', got '{meta.model}'"


        # check no elements have None
        assert not df.isnull().values.any()

        # also check start and best loss are same
        assert df["Loss at Epoch Start"].iloc[0] == df["Best Loss at Iteration"].iloc[0]


        # Iterate all rows and check the Saved Path exists
        for index, row in df.iloc[1:].iterrows():
            path = row["Saved Path"]
            assert os.path.exists(path), f"Saved Path does not exist: {path}"

            init_loss = row["Loss at Epoch Start"]
            best_loss = row["Best Loss at Iteration"]
            assert best_loss <= init_loss, f"Best Loss at Iteration is greater than Loss at Epoch Start: {best_loss} > {init_loss}"

            match = re.search(r'loss_([0-9]*\.?[0-9]+)', path)
            assert match is not None, f"Loss value not found in path: {path}"
            loss_value = float(match.group(1))
            assert np.isclose(loss_value, best_loss, atol=1e-5), f"Loss value in path {loss_value} does not match Best Loss at Iteration {best_loss}"

            
    # Check properties of the saved matrices
    def test_check_unitary_and_complex(self, log_file_path):
        # Parse the log file
        df, meta = parse_log_file(log_file_path)

        # load path from the first row
        path = df["Saved Path"].iloc[0]
        assert os.path.exists(path), f"Saved Path does not exist: {path}"
        H = np.load(path)
        # Check if this is float
        assert not np.iscomplexobj(H), f"Matrix is not float: {path}"
        assert np.allclose(H, H.conj().T, atol=1e-5), f"Matrix is not hermitian: {path}"

        # Iterate all rows and check the Saved Path exists
        for index, row in df.iloc[1:].iterrows():
            path = row["Saved Path"]
            assert os.path.exists(path), f"Saved Path does not exist: {path}"

            # Load the matrix from the .npy file
            u = np.load(path)

            # Check if the matrix is complex-valued
            assert np.iscomplexobj(u), f"Matrix is not complex-valued: {path}"

            # Check if the matrix is unitary: U * U.H = I
            identity = np.eye(u.shape[0], dtype=u.dtype)
            unitary_check = np.allclose(np.dot(u, u.conj().T), identity, atol=1e-5)
            assert unitary_check, f"Matrix is not unitary: {path}"
    
    # Check the saved best loss values is actually the best loss
    def test_mel_loss(self, log_file_path):
        # Parse the log file
        log_df, meta = parse_log_file(log_file_path)
        initial_path = log_df["Saved Path"].iloc[0]
        neg_H = -np.load(initial_path) #saved as -H
        neg_H = neg_H - 10 * np.eye(neg_H.shape[0], dtype=neg_H.dtype)
        eigenvalues = np.linalg.eigvalsh(neg_H)
        min_loss = eigenvalues[0]
        #check all eigenvalues are negative
        assert np.all(eigenvalues < 0), f"Matrix is not negative definite: {initial_path}"
        initial_loss0 = np.linalg.eigvalsh(-np.abs(neg_H))[0]
        initial_loss = min_loss - initial_loss0
        # assert log_df["Best Loss at Iteration"].iloc[0] == initial_loss
        assert np.isclose(log_df["Best Loss at Iteration"].iloc[0], initial_loss, atol=1e-5)

        for idx, row in log_df.iloc[1:].iterrows():
            saved_path = row["Saved Path"]
            best_loss_iter = row["Best Loss at Iteration"]
            unitary_matrix = np.load(saved_path)
            U = np.kron(unitary_matrix, unitary_matrix)
            transformed_H = U @ neg_H @ U.conj().T
            tmp_loss = np.linalg.eigvalsh(-np.abs(transformed_H))[0]
            loss = min_loss - tmp_loss
            assert np.isclose(loss, best_loss_iter, atol=1e-5)




    

