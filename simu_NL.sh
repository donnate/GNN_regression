#!/bin/bash

#SBATCH --job-name=array_other
#SBATCH --output=experiments/logs/array_other_%A_%a.out
#SBATCH --error=experiments/logs/array_other_%A_%a.err
#SBATCH --array=1-20
#SBATCH --time=6:00:00
#SBATCH --partition=caslake
#SBATCH --ntasks=1
#SBATCH --mem=5G
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
if [ "$4" == "Barbell" ]
then
    python3 simulation_nonlinear_barbell_graph.py --add_node $1 --noise $2 --non_linear $3 --seed ${SLURM_ARRAY_TASK_ID}
else
    if [ "$4" == "Tree" ]
    then
        python3 simulation_nonlinear_tree.py --add_node $1 --noise $2 --non_linear $3 --seed ${SLURM_ARRAY_TASK_ID}
    else
        python3 simulation_nonlinear_powlaw.py --add_node $1 --noise $2 --non_linear $3 --seed ${SLURM_ARRAY_TASK_ID}
 fi
fi
