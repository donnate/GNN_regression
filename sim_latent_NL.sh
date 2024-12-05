#!/bin/bash

#SBATCH --job-name=array
#SBATCH --output=experiments/logs/array_%A_%a.out
#SBATCH --error=experiments/logs/array_%A_%a.err
#SBATCH --array=1-12
#SBATCH --time=12:00:00
#SBATCH --partition=cdonnat
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --account=pi-cdonnat

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "My SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
# Add lines here to run your computations
job_id=$SLURM_ARRAY_JOB_ID
module load gcc
module load gsl

cd $SCRATCH/$USER/GNN_regression/GNN_regression
module load python
source activate py311
# Run one experiment  to create the dataset
python3 simulation_latent_non_linearity.R --add_node $1 --noise $2 --seed ${SLURM_ARRAY_TASK_ID}
