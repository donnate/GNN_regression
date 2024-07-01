import argparse
import copy
import numpy as np
import pandas as pd
import networkx as nx
import sys, os


sys.path.append('/scratch/midway3/cdonnat/GNN/GNN_regression')
#sys.path.append('~/Documents/GNN_regression')
from functions import *

parser = argparse.ArgumentParser()
parser.add_argument('--namefile', default = "constant_delta", type=str)
parser.add_argument('--seed', type=int,  default = 1)
parser.add_argument('--dim_grid', type=int, default=2)
parser.add_argument('--n_nodes_x', type=int, default=33)
parser.add_argument('--p_sbm_within', type=float, default=0.5)
parser.add_argument('--p_er', type=float, default=0.5)
parser.add_argument('--m_pa', type=int, default=1)
#parser.add_argument('--r', type=int, default=2)
parser.add_argument('--p_sbm_between', type=float, default=0.1)
parser.add_argument('--graph_type', type=str, default="grid")
args = parser.parse_args()


np.random.seed(args.seed)
n_nodes_x =  args.n_nodes_x
# Create the grid graph
if args.graph_type == "grid":
    G = nx.grid_graph(dim=[n_nodes_x] * args.dim_grid)
    namefile =  "experiment_" + args.namefile + "_seed_" + str(args.seed) +  "gridgraph_dim_grid" + str(args.dim_grid) + ".csv"
    # Create the meshgrid for the coordinates
    x = np.linspace(-1, 1, n_nodes_x)
    y = np.linspace(-1, 1, n_nodes_x)
    x, y = np.meshgrid(x, y)
    # Reshape and combine the coordinates
    X_pos = np.hstack([x.reshape([-1, 1]), y.reshape([-1, 1])])
    # Create the position dictionary



    pos = {(i % n_nodes_x, i // n_nodes_x): [X_pos[i, 0], X_pos[i, 1]] for i in np.arange(X_pos.shape[0])}
elif args.graph_type == "SBM":
    sizes = [n_nodes_x//3, n_nodes_x//3, n_nodes_x//3]
    probs = [[args.p_sbm_within, args.p_sbm_between, args.p_sbm_between], 
             [args.p_sbm_between, args.p_sbm_within, args.p_sbm_between], 
             [args.p_sbm_between, args.p_sbm_between, args.p_sbm_within]]
    G = nx.stochastic_block_model(sizes, probs, seed=args.seed)
    namefile =  "experiment_" + args.namefile + "_seed_" + str(args.seed) +  "SBM_p_within" + str(args.p_sbm_within) + "_p_between" + str(args.p_sbm_between) + ".csv"
elif args.graph_type == "ER":
    namefile =  "experiment_" + args.namefile + "_seed_" + str(args.seed) +  "ER_p_er" + str(args.p_er) + ".csv"
    G = nx.erdos_renyi_graph(n_nodes_x, args.p_er, 
                             seed=args.seed, directed=False)
elif args.graph_type == "PA":
    namefile =  "experiment_" + args.namefile + "_seed_" + str(args.seed) +  "PA_m_pa" + str(args.m_pa) + ".csv"
    G = nx.barabasi_albert_graph(n_nodes_x, args.m_pa, seed=args.seed)
else:
    print("Graph type not recognized")


n = nx.number_of_nodes(G)
A = nx.adjacency_matrix(G)
Gamma  = nx.incidence_matrix(G, oriented=True)
Gamma = Gamma.T

inv_degree = np.diag(1./ np.sum(A, 1))
A_norm = inv_degree @ A
G2 = nx.from_numpy_array(A_norm)
Gamma2 = nx.incidence_matrix(G2, oriented=True).toarray()
edges_with_weights = G2.edges(data=True)
weights = np.array([data['weight'] for u, v, data in edges_with_weights])
# Apply weights to the incidence matrix
Gamma2 = Gamma2 @ np.diag(weights)
Gamma2 = Gamma2.T

A_tilde = A.todense() + np.eye(n)
D_tilde = np.diag(1./ np.sum(A_tilde, 1)**0.5)
S = D_tilde @ A @ D_tilde

D2 = np.diag(1./ np.sum(A_tilde, 1))
T = D2 @ A_tilde

L = nx.laplacian_matrix(G)

nb_exp = 10
scale_grid = [0.01, 0.5,  1.0, 5.0]
M_grid = [1]
alpha_grid = [0.01, 0.1, 0.5, 1]
r_grid = [2, 4, 10]

results = pd.DataFrame(np.zeros(( 3 * nb_exp *  len(r_grid) *  len(scale_grid) *  len(M_grid) * len(alpha_grid), 36)),
                       columns=['exp_id', 
                                'method', 
                                'graph_type',
                                  'dim_grid', 
                                  'p_er', 
                                  'm_pa', 
                                  'p_sbm_between', 
                                  'p_sbm_within', 'nb_conv',
                                  'noise_scale', 'r',
                                  'M', 'alpha', 'train_mse', 'smoothness_condition_1', 
                                  'smoothness_condition_2'] + ['test_mse_' + str(i) for i in np.arange(20)])
it = 0
S_k = np.eye(n)
T_k = np.eye(n)

for k in np.arange(1, 10):
    S_k = S @ S_k
    T_k = T @ T_k
    B_k = (T != 0).astype(int)
    inv_degree = np.diag(1./ np.sum(B_k, 1))
    B_k = inv_degree @ B_k


    for scale in scale_grid:
        for r in r_grid:
            for M in M_grid:
                for alpha in alpha_grid:
                    print([k, scale, M, alpha])
                    for exp in np.arange(nb_exp):
                        ###
                        f, X =  smooth_sobolev_function(L, r=r, alpha=alpha, M=M) 
                        #### Check smoothness condition
                        smoothness_condition_1 = np.max(Gamma.dot(f))
                        smoothness_condition_2 = np.max(Gamma2.dot(f))

                        Y = np.random.normal(f, scale=scale)
                        Y_pred = S_k @ Y
                        train_acc = np.mean(np.square(Y_pred - Y)) 
                        res_temp = [exp, 0, args.graph_type,
                                    args.dim_grid,
                                    args.p_er,
                                    args.m_pa,
                                    args.p_sbm_between,
                                    args.p_sbm_within,
                                    k,  scale, r, M, alpha, float(train_acc),
                                    smoothness_condition_1, smoothness_condition_2]
                        for u in range(20):
                            new_Y = np.random.normal(f, scale=scale)
                            test_mse = (np.square(Y_pred - new_Y)).mean()
                            res_temp += [float(test_mse)]
                        results.iloc[it, :] = res_temp
                        results.to_csv("results/" + namefile)
                        it += 1
                        
                        Y_pred2 = T_k @ Y
                        train_acc = np.mean(np.square(Y_pred2 - f)) 
                        res_temp =  [exp, 1, args.graph_type,
                                    args.dim_grid,
                                    args.p_er,
                                    args.m_pa,
                                    args.p_sbm_between,
                                    args.p_sbm_within,
                                    k,  scale, r, M, alpha, float(train_acc),
                                    smoothness_condition_1, smoothness_condition_2]
                        for u in range(20):
                            new_Y = np.random.normal(f, scale=scale)
                            test_mse = (np.square(Y_pred2 - f)).mean()
                            res_temp += [float(test_mse)]
                        results.iloc[it, :] = res_temp
                        results.to_csv("results/" + namefile)
                        it += 1

                        Y_pred3 = B_k @ Y
                        train_acc = np.mean(np.square(Y_pred3 - f)) 
                        res_temp =  [exp, 2, args.graph_type,
                                    args.dim_grid,
                                    args.p_er,
                                    args.m_pa,
                                    args.p_sbm_between,
                                    args.p_sbm_within,
                                    k,  scale, r, M, alpha, float(train_acc),
                                    smoothness_condition_1, smoothness_condition_2]
                        for u in range(20):
                            new_Y = np.random.normal(f, scale=scale)
                            test_mse = (np.square(Y_pred3 - f)).mean()
                            res_temp += [float(test_mse)]
                        results.iloc[it, :] = res_temp
                        results.to_csv("results/" + namefile)
                        it += 1
                        #print(it)

                    
        
            








