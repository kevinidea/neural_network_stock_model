#!/bin/bash

#SBATCH -J NN_month                   # Job name
#SBATCH -p normal                         # Partition (queue) to submit to
#SBATCH -G 0                           # Max GPUs per user: 2
#SBATCH -c 26                         # Number of CPU cores
#SBATCH --mem=160G                    # memory
#SBATCH -t 2-                        # Max job time
#SBATCH -o '/zfs/projects/darc/wolee_edehaan_suzienoh-exploratory-ml/kevin/output/log/monthly_new_restricted_%j.out'
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
    --period 'month' \
    --prediction_parent_path 'kevin/output/prediction/monthly_new_restricted' \
    --num_samples 200 \
    --max_num_epochs 40 \
    --num_cpus ${SLURM_CPUS_PER_TASK} \
    --cpus_per_trial 1 \
    --num_gpus ${SLURM_GPUS} \
    --gpus_per_trial 0 \
    --patience 3 \
    --prediction_years 1996 && \

# post process prediction files
python3 kevin/src/postprocess_predictions.py \
    --prediction_parent_path 'kevin/output/prediction/monthly_new_restricted' \
    --period 'month'