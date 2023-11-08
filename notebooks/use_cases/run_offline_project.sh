#!/bin/bash

# Positional input arguments: <OUTPUT_PATH> <DEVICE>
OUTPUT_PATH=$1
DEVICE=$2

# Debugging Information
echo VENV_PATH=$VENV_PATH
echo PBS_O_WORKDIR=$PBS_O_WORKDIR
echo PBS_JOBID=$PBS_JOBID
echo OUTPUT_PATH=$OUTPUT_PATH
echo DEVICE=$DEVICE

# Activate Virtual Environment
echo "Activating virtual environment ${VENV_PATH}"
source ${VENV_PATH}/bin/activate

# The default path for the job is the user's home directory,
# change directory to where the files are.
cd $PBS_O_WORKDIR

# Make sure that the output directory exists.
mkdir -p $OUTPUT_PATH

# Run the project
python3 run_offline_project.py \
    -o $OUTPUT_PATH \
    -d $DEVICE
