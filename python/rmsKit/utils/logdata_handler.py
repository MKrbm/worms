import re
import pandas as pd

def parse_log_file(file_path):
    with open(file_path, 'r') as file:
        log_content = file.read()

    # Fetch initial loss and initial saved path
    initial_loss_match = re.search(r'initial loss = (\d+\.\d+)', log_content)
    if not initial_loss_match:
        raise ValueError("Initial loss not found in log file")
    initial_loss = float(initial_loss_match.group(1))

    initial_saved_path_match = re.search(r'save matrix \(\(\d+, \d+\)\): (.+\.npy)', log_content)
    if not initial_saved_path_match:
        raise ValueError("Initial saved path not found in log file")
    initial_saved_path = initial_saved_path_match.group(1)

    # Split log data by sections starting with 'Start iteration'
    iterations_data = log_content.split('------ Start iteration')[1:]
    if len(iterations_data) == 0:
        raise ValueError("No iterations found in log file")

    # Create a DataFrame to store the extracted data
    data = []

    # Add initial loss and initial saved path as the first row
    data.append({'Iteration': 0, 'Loss at Epoch Start': initial_loss, 'Best Loss at Iteration': initial_loss, 'Saved Path': initial_saved_path})

    # Extract necessary information from each iteration
    for iteration_data in iterations_data:
        iteration_match = re.search(r'(\d+)', iteration_data)
        if not iteration_match:
            raise ValueError("Iteration number not found in log file")
        iteration = int(iteration_match.group(1))

        loss_at_epoch_start_match = re.search(r'Epoch: 1/\d+, Loss: (\d+\.\d+)', iteration_data)
        if not loss_at_epoch_start_match:
            raise ValueError("Loss at epoch start not found in log file")
        loss_at_epoch_start = float(loss_at_epoch_start_match.group(1))

        best_loss_match = re.search(r'best loss at iteration \d+: (\d+\.\d+), best loss so far: (\d+\.\d+),', iteration_data)
        if not best_loss_match:
            raise ValueError("Best loss at iteration not found in log file")
        best_loss_at_iteration = float(best_loss_match.group(1))
        best_loss_so_far = float(best_loss_match.group(2))

        saved_path_match = re.search(r'save matrix \(\(\d+, \d+\)\): (.+\.npy)', iteration_data)
        if not saved_path_match:
            raise ValueError("Saved path not found in log file")
        saved_path = saved_path_match.group(1)

        data.append({
            'Iteration': iteration,
            'Loss at Epoch Start': loss_at_epoch_start,
            'Best Loss at Iteration': best_loss_at_iteration,
            'Saved Path': saved_path
        })

    df = pd.DataFrame(data)
    return df