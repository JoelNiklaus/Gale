#!/bin/bash
source activate gale
CUDA_V="cu"$(conda list pytorch --json | python -c "import sys, json; print(json.load(sys.stdin)[0]['dist_name'].split('_')[1][4:8].replace('.', ''))")
PYTORCH_VERSION=$(conda list pytorch --json | python -c "import sys, json; print(json.load(sys.stdin)[0]['version'])")

# make sure CUDA version is valid
if ! [[ "cu92cu100cu101" =~ ${CUDA_V} ]]
then
   echo "Current CUDA version ${CUDA_V} is not valid. CUDA version 9.2, 10.0 or 10.1 needed."
fi

# make sure pytorch version is valid
if ! [[ "1.4.01.5.0" =~ ${PYTORCH_VERSION} ]]
then
   echo "Current PyTorch version ${PYTORCH_VERSION} is not valid. PyTorch version 1.4.0 or 1.5.0 is needed."
fi

echo "Installing the additional environment requirements for the loss landscape runner"
pip install loss_landscape==0.0.6.dev2
conda install openmpi

# test the installation
python -c "import importlib
found = importlib.util.find_spec('loss_landscape') != None;
if found:
    print('Loss Landscape was successfully installed.')
else:
    print('Installation was unsuccessful :(! Checkout https://github.com/joelniklaus/loss_landscape to get the latest installation information.')
"
