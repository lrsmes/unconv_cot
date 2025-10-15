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
    "t_f0": 5,
    "t_us": 0.015,
    "t": 0.1,
    "t_vf": 0.04,
    "Gd": 0.475,
    "Gl": 0.4,
    "Gr": 0.4,
    "Bz": 0,
    "Bx": 0,
    "gs": 2,
    "gv": 14,
    "soc": 0.07,
    "Vbias": 0,
    "delta_Vl_start": -0.25,
    "delta_Vl_stop": 2.5,
    "delta_Vr_start": -2,
    "delta_Vr_stop": -4.5,
    "split": 250,
    "pulse_dir": 1
}

# Choose varied parameter and sweep space
param_name = "t_us"
sweep_values = np.linspace(0.01, 1.2, 21)

variations = [{param_name: val} for val in sweep_values]

# Define new directory and file name
current_dir = os.getcwd()
new_dir = os.path.join(current_dir, "blockade_vs_tus")  # <--- specify your directory name
file_name = "params.json"

# Create file
create_single_json_file(new_dir, file_name, variations, base_params)
