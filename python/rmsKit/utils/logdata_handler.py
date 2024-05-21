import re
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def parse_log_file(file_path):
    with open(file_path, 'r') as file:
        log_content = file.read()

    # Fetch initial loss and initial saved path
    initial_loss_match = re.search(r'initial loss = (\d+\.\d+(?:e[+-]?\d+)?)', log_content)
    if not initial_loss_match:
        raise ValueError("Initial loss not found in log file")
    initial_loss = float(initial_loss_match.group(1))

    initial_saved_path_match = re.search(r'save matrix \(\(\d+, \d+\)\): (.+\.npy)', log_content)
    if not initial_saved_path_match:
        raise ValueError("Initial saved path not found in log file")
    initial_saved_path = initial_saved_path_match.group(1)

    # Fetch dtype, seed, and model
    dtype_match = re.search(r'dtype: (\w+)', log_content)
    if not dtype_match:
        raise ValueError("dtype not found in log file")
    dtype = dtype_match.group(1)

    seed_match = re.search(r'seed: (\d+)', log_content)
    if not seed_match:
        raise ValueError("seed not found in log file")
    seed = int(seed_match.group(1))

    model_match = re.search(r'model: (\w+)', log_content)
    if not model_match:
        raise ValueError("model not found in log file")
    model = model_match.group(1)

    # Split log data by sections starting with 'Start iteration'
    iterations_data = log_content.split('------ Start iteration')[1:]
    if len(iterations_data) == 0:
        raise ValueError("No iterations found in log file")

    # Create a DataFrame to store the extracted data
    data = []

    # Add initial loss and initial saved path as the first row
    data.append({'Iteration': 0, 'Loss at Epoch Start': initial_loss, 'Best Loss at Iteration': initial_loss, 'Saved Path': initial_saved_path})

    # Extract necessary information from each iteration
    for i, iteration_data in enumerate(iterations_data):
        iteration_match = re.search(r'(\d+)', iteration_data)
        if not iteration_match:
            logger.error(f"{i}-th iteration failed. Data is {iteration_data}")
            raise ValueError("Iteration number not found in log file")
        iteration = int(iteration_match.group(1))

        loss_at_epoch_start_match = re.search(r'Epoch: 1/\d+, Loss: (\d+\.\d+(?:e[+-]?\d+)?)', iteration_data)
        if not loss_at_epoch_start_match:
            logger.error(f"{i}-th iteration failed. Data is {iteration_data}")
            raise ValueError("Loss at epoch start not found in log file")
        loss_at_epoch_start = float(loss_at_epoch_start_match.group(1))

        best_loss_match = re.search(r'best loss at iteration \d+: (\d+\.\d+(?:e[+-]?\d+)?), best loss so far: (\d+\.\d+(?:e[+-]?\d+)?),', iteration_data)
        if not best_loss_match:
            logger.error(f"{i}-th iteration failed. Data is {iteration_data}")
            logger.error(f"Match is {best_loss_match}")
            raise ValueError("Best loss at iteration not found in log file")
        best_loss_at_iteration = float(best_loss_match.group(1))
        best_loss_so_far = float(best_loss_match.group(2))

        saved_path_match = re.search(r'save matrix \(\(\d+, \d+\)\): (.+\.npy)', iteration_data)
        if not saved_path_match:
            logger.error(f"{i}-th iteration failed. Data is {iteration_data}")
            raise ValueError("Saved path not found in log file")
        saved_path = saved_path_match.group(1)

        data.append({
            'Iteration': iteration,
            'Loss at Epoch Start': loss_at_epoch_start,
            'Best Loss at Iteration': best_loss_at_iteration,
            'Saved Path': saved_path
        })

    df = pd.DataFrame(data)
    
    # Return the DataFrame along with dtype, seed, and model
    return df, {'dtype': dtype, 'seed': seed, 'model': model}
    return df, {'dtype': dtype, 'seed': seed, 'model': model}