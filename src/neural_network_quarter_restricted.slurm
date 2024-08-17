#!/bin/bash

#SBATCH -J NN_quarter                   # Job name
#SBATCH -p gpu                         # Partition (queue) to submit to
#SBATCH -G 4                           # Max GPUs per user: 2
#SBATCH -c 24                          # Number of CPU cores
#SBATCH --mem=120G                    # memory
#SBATCH -t 1-                        # Max job time
#SBATCH -o '/zfs/projects/darc/wolee_edehaan_suzienoh-exploratory-ml/kevin/output/log/quarterly_new_restricted_%j.out'
#SBATCH --mail-type=ALL                # Email notifications
#SBATCH --mail-user=kevin128@stanford.edu

# change to the correct working directory
cd '/zfs/projects/darc/wolee_edehaan_suzienoh-exploratory-ml'
# show current directory
echo 'Current working directory'
pwd

# activate python virtual environment
echo 'source kevin/venv/bin/activate'
source kevin/venv/bin/activate
# show which python
echo 'which python3'
which python3

# starting Ray Tune before running the main script
ray start \
    --num-cpus=${SLURM_CPUS_PER_TASK} \
    --num-gpus=${SLURM_GPUS}

# start the main script
python3 kevin/src/main_restricted.py \
    --infile_path '/zfs/projects/faculty/suzienoh-ml/Data/pf_vars_240620.csv' \
    --period 'quarter' \
    --prediction_parent_path 'kevin/output/prediction/quarterly_new_restricted' \
    --num_samples 200 \
    --max_num_epochs 50 \
    --num_cpus ${SLURM_CPUS_PER_TASK} \
    --cpus_per_trial 1 \
    --num_gpus ${SLURM_GPUS} \
    --gpus_per_trial 1 \
    --patience 5 \
    --prediction_years 2000 2009 2013 && \

# post process prediction files
python3 kevin/src/postprocess_predictions.py \
    --prediction_parent_path 'kevin/output/prediction/quarterly_new_restricted' \
    --period 'quarter'