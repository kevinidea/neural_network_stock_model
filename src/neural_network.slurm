#!/bin/bash

#SBATCH -J neural_network                   # Job name
#SBATCH -p gpu                         # Partition (queue) to submit to
#SBATCH -c 40                          # Number of CPU cores
#SBATCH -G 2                           # Max GPUs per user: 2
#SBATCH --mem=40G                    # memory
#SBATCH -N 1                           # Request one node
#SBATCH -t 1-                          # Max job time
#SBATCH -o '/zfs/projects/darc/wolee_edehaan_suzienoh-exploratory-ml/kevin/output/neural_network_%j.out'           # Output file
#SBATCH --mail-type=ALL                # Email notifications
#SBATCH --mail-user=kevin128@stanford.edu

# change to the correct directory
cd '/zfs/projects/darc/wolee_edehaan_suzienoh-exploratory-ml/kevin'
# show current directory
echo 'Current working directory'
pwd

# activate python virtual environment
echo 'source venv/bin/activate'
source venv/bin/activate
# show which python
echo 'which python3'
which python3

# run the script
ray start --num-gpus=2
python3 src/main.py