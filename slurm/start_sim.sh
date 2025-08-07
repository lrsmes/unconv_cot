#!/bin/bash

# <<< conda initialize <<<
conda activate unconv_co

# run the training
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u /home/lmester/unconv_cot/sim_code.py