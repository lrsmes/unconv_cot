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
    "t_us": 0.1,
    "t": 0.2,
    "t_vf": 0.15,
    "Gd": 0.35,
    "Gl": 0.23,
    "Gr": 0.23,
    "Bz": 1,
    "Bx": 0,
    "gs": 2,
    "gv": 14,
    "soc": 0.07,
    "Vbias": 0,
    "delta_Vl": {"start": -0.75, "stop": 1.75},
    "delta_Vr": {"start": -1.25, "stop": -3.75},
    "decay_points": [
        (-0.12, -2.33), (0.46, -1.77), (0.6, -2.84)
    ],
    #"decay_points": {
    #     "Vl": [0.5, 0.5, 0.8, 0.7, 0.75],   #0T: [0.5, 0.5, 0.8, 0.7, 0.75]
    #     "Vr": [-2.5, -3.0, -2.9, -2.65, -3.2], #0T: [-2.5, -3.0, -2.9, -2.65, -3.2]
    # },
    "split": 150,
    "pulse_dir": -1
}

# Choose varied parameter and sweep space
param_name = "t_us"
sweep_values = np.linspace(0.11, 5, 11)

variations = [{param_name: val} for val in sweep_values]

# Define new directory and file name
current_dir = os.getcwd()
new_dir = os.path.join(current_dir, "transport_vs_tus_1T_reg2")  # <--- specify your directory name
file_name = "params.json"

# Create file
create_single_json_file(new_dir, file_name, variations, base_params)

# reg2: (-0.12, -2.33), (0.46, -1.77), (0.6, -2.84)
# reg1: (-0.11, -2.33), (0.46, -1.77), (-0.3, -2.14), (0.44, -1.42)
