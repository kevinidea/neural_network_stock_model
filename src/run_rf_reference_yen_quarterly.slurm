#!/bin/bash

# Example of running python script in a batch mode

#SBATCH -J rf_yen_q
#SBATCH -p long
#SBATCH -c 32   # CPU cores (up to 256 on normal partition)
#SBATCH --mem=200GB
#SBATCH -t 7-00:00:0
#SBATCH -o rf_yen_q-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jeffott@stanford.edu

#activate environment
source /zfs/projects/darc/wolee_edehaan_suzienoh-exploratory-ml/Phase3/venv/bin/activate

# Run python script
python3 rf_yen.py --infile_path './pf_vars_240620.csv' --outfile_path './results_Q' --cum 1 --lower_percentile 5 --upper_percentile 95 --period 'quarter'