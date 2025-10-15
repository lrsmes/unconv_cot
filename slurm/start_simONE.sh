#!/bin/bash

# Default answer in case of no input (e.g., running with nohup)
DEFAULT_ANSWER="y"

# Check if running interactively
if [ -t 0 ]; then
    # Prompt the user only if the script is run interactively
    read -p "Do you want to create a new parameter file? (y/n) " answer
else
    echo "No terminal detected, defaulting to '$DEFAULT_ANSWER'."
    answer=$DEFAULT_ANSWER
fi

# <<< conda initialize <<<
conda activate unconv_co

# Check the user's answer
case $answer in
    [Yy]* )
        # If yes, run the first Python script to create the parameter file
        python create_simONE_params.py
        ;;
    [Nn]* )
        echo "Skipping parameter file creation."
        ;;
    * )
        echo "Invalid input. Please answer yes or no."
        exit 1
        ;;
esac

# run the training
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u /home/lmester/unconv_cot/simONE.py