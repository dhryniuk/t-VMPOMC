#!/bin/bash -l

#================================================
# Generic HPC Job Submission Script
# Purpose: Submit parallel Julia calculations
#================================================

# Resource Requirements
#---------------------
# Memory allocation (accepts suffix M, G, or T)
#$ -l mem=9G

# Wall clock time limit (format HH:MM:SS)
#$ -l h_rt=48:00:0

# Parallel environment setup
#--------------------------
# Request MPI with specified number of processes
#$ -pe mpi 12

# Array job specification (tasks 1-N)
#$ -t 1-8

#================================================
# Module Loading
#================================================
module load default-modules/2018
module load julia/1.9.0

# Set working directory for output files
#$ -wd ./results/${JOB_NAME}/${SGE_TASK_ID}

#================================================
# Parameter Processing
#================================================
# Read parameters from configuration file
PARAM_FILE="./config/parameters.txt"

if [ ! -f "$PARAM_FILE" ]; then
    echo "Error: Parameter file not found!"
    exit 1
fi

# Extract parameters for current task
task_number=$SGE_TASK_ID
index=$(sed -n ${task_number}p $PARAM_FILE | awk '{print $1}')
chi=$(sed -n ${task_number}p $PARAM_FILE | awk '{print $2}')

#================================================
# Launch Calculation
#================================================
# Arguments:
# 1. System size
# 2. Bond dimension (chi)
# 3. Unit cell size
# 4. Number of samples
# 5. Total evolution time
# 6-9. Auxiliary hyperparameters (refer to script)
gerun julia ./scripts/Heisenberg_spin_chain.jl \
    200 "$chi" 1 3000 5.0 \
    1e-12 1e-6 0.1 0.05