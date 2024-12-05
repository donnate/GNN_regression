#!/bin/bash
  # Define the values for the variables
#n_values="300 500 1000 10000"
NL_values="1"
graph_values="Barbell Tree PL"
add_node_values="0 1"
noise_values="0.1 0.5 1. 2 5. 10."

for noise in $noise_values; do
  for add_node in $add_node_values; do
    for NL in $NL_values; do
      for graph in $graph_values; do
          sbatch simu_NL.sh "$add_node" "$noise" "$NL" "$graph"
        done
    done
  done
done