#!/bin/bash

PROJECT_DIR="/home/lmester/unconv_cot"
SIM_NAME="${1:-transport_vs_tus_1T_reg2}"
JOB_CHOICE="${2:-}"
SIM_DIR="${PROJECT_DIR}/runs/simulations/${SIM_NAME}"
PARAM_FILE="${SIM_DIR}/params.json"

# Default answer in case of no input (e.g., running with nohup)
DEFAULT_ANSWER="y"
DEFAULT_JOB_CHOICE="maps_decay_once"

# Check if running interactively
if [ -t 0 ]; then
    # Prompt the user only if the script is run interactively
    read -p "Do you want to create a new parameter file? (y/n)" answer
else
    echo "No terminal detected, defaulting to '$DEFAULT_ANSWER'."
    answer=$DEFAULT_ANSWER
fi

if [ -z "$JOB_CHOICE" ]; then
    if [ -t 0 ]; then
        read -p "What should be created? (map/decay/both/maps_decay_once) " JOB_CHOICE
    else
        echo "No terminal detected, defaulting jobs to '$DEFAULT_JOB_CHOICE'."
        JOB_CHOICE=$DEFAULT_JOB_CHOICE
    fi
fi

case $JOB_CHOICE in
    map|maps )
        JOBS="map"
        ;;
    decay|decays )
        JOBS="decay"
        ;;
    both|all )
        JOBS="map decay"
        ;;
    maps_decay_once|map_decay_once|once )
        JOBS="map decay_once"
        ;;
    cut|cuts )
        JOBS="cut"
        ;;
    * )
        echo "Invalid job choice. Use map, decay, both, maps_decay_once, or cut."
        exit 1
        ;;
esac

# <<< conda initialize <<<
conda activate unconv_co

cd "$PROJECT_DIR" || exit 1

# Check the user's answer
case $answer in
    [Yy]* )
        GENERATE_ARGS="--generate-params"
        ;;
    [Nn]* )
        echo "Skipping parameter file creation."
        GENERATE_ARGS=""
        ;;
    * )
        echo "Invalid input. Please answer yes or no."
        exit 1
        ;;
esac

# run the training
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH="${PROJECT_DIR}/src" python -u -m unconv_cot.sim_code \
    --sim-dir "$SIM_DIR" \
    --params "$PARAM_FILE" \
    --jobs $JOBS \
    $GENERATE_ARGS
