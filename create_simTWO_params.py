import json
import os
import numpy as np

def create_single_json_file(directory, file_name, variations, base_params):
    """
    Creates a single JSON file containing a list of parameters, each set representing a variation.

    :param directory: The directory where the JSON file will be saved.
    :param file_name: The name of the JSON file (e.g. 'params.json').
    :param variations: A list of dictionaries, each containing the parameters to vary.
    :param base_params: A dictionary with the base parameters.
    """
    # Ensure the directory exists (create it if needed)
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, file_name)

    all_params = []
    for variation in variations:
        params = base_params.copy()
        params.update(variation)
        all_params.append(params)

    with open(file_path, 'w') as file:
        json.dump(all_params, file, indent=4)

    print(f"Created JSON file with {len(all_params)} variations: {file_path}")

# Base parameters
base_params = {
    "t": 0.05,
    "Gd": 0.575,
    "Gl": 0.02,
    "Gr": 0.02,
    "Bz": 0,
    "Bx": 0,
    "gs": 2,
    "gv": 14,
    "soc": 0.07,
    "dkk": 0,
    "Vbias": 1,
    "delta_Vl_start": -7,
    "delta_Vl_stop": -4.5,
    "delta_Vr_start": 1,
    "delta_Vr_stop": 3.5,
    "split": 250
}

# Choose varied parameter and sweep space
param_name = "dkk"
sweep_values = np.linspace(0.001, 0.0022, 9)

variations = [{param_name: val} for val in sweep_values]

# Define new directory and file name
current_dir = os.getcwd()
new_dir = os.path.join(current_dir, "ss_vs_tvf")  # <--- specify your directory name
file_name = "params.json"

# Create file
create_single_json_file(new_dir, file_name, variations, base_params)
