#!/bin/bash
  # Define the values for the variables
#n_values="300 500 1000 10000"
NL_values="0"
add_node_values="1"
noise_values="0.1 0.5 1. 2. 5. 10."

for noise in $noise_values; do
  for add_node in $add_node_values; do
    for NL in $NL_values; do
        sbatch sim_latent_NL.sh "$add_node" "$noise" "$NL"
    done
  done
done
