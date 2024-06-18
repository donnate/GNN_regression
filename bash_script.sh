#!/bin/bash
#SBATCH --job-name=sim
#SBATCH --output=logs/run_sim_%A_%a.out
#SBATCH --error=logs/run_sim_%A_%a.err
#SBATCH --array=1-20
#SBATCH --time=2:00:00
#SBATCH --partition=cdonnat
#SBATCH --mem=3G
#SBATCH --account=pi-cdonnat
#SBATCH --qos=cdonnat
#SBATCH --mail-type=ALL
#SBATCH --nodes=midway3-0[419-421]


STARTTIME=$(date +%s).$(date +%N | cut -c1-6)
echo “$SLURM_ARRAY_TASK_ID Start time: $STARTTIME”

result_file="experiment_grid_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

module load python
module load pytorch

python3 main.py --seed $SLURM_ARRAY_TASK_ID --namefile $result_file --dim_grid $1




