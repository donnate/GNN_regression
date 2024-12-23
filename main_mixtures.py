import argparse
import copy
import numpy as np
import pandas as pd
import networkx as nx
import sys, os


#sys.path.append('/scratch/midway3/cdonnat/GNN/GNN_regression')
sys.path.append('~/Documents/GNN_regression')
from functions import *

parser = argparse.ArgumentParser()
parser.add_argument('--namefile', type=str)
parser.add_argument('--seed', type=int)
parser.add_argument('--dim_grid', type=int)
parser.add_argument('--n_nodes_x', type=int, default=30)
args = parser.parse_args()


np.random.seed(args.seed)
n_nodes_x =  args.n_nodes_x
namefile =  "experiment_mixture_" + args.namefile + "_seed_" + str(args.seed) +  "dim_grid" + str(args.dim_grid) +".csv"
# Create the grid graph
G = nx.grid_graph(dim=[n_nodes_x] * args.dim_grid)
# Create the meshgrid for the coordinates
x = np.linspace(-1, 1, n_nodes_x)
y = np.linspace(-1, 1, n_nodes_x)
x, y = np.meshgrid(x, y)
# Reshape and combine the coordinates
X_pos = np.hstack([x.reshape([-1, 1]), y.reshape([-1, 1])])
# Create the position dictionary



pos = {(i % n_nodes_x, i // n_nodes_x): [X_pos[i, 0], X_pos[i, 1]] for i in np.arange(X_pos.shape[0])}
n = nx.number_of_nodes(G)
A = nx.adjacency_matrix(G)
A = A.todense() + np.eye(n)
D = np.diag(1./ np.sum(A, 1)**0.5)
S = D @ A @ D

D2 = np.diag(1./ np.sum(A, 1))
T = D2 @ A

L = nx.laplacian_matrix(G)

nb_exp = 1
results = np.zeros(( 8 *  6 *  3 * 10, 27))
it = 0
S_k = np.eye(n)
T_k = np.eye(n)

for k in np.arange(1, 10):
    S_k = S @ S_k
    T_k = T @ T_k
    for scale in [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 10]:
        for M in [0.1, 1, 5, 10]:
            for alpha in [0.01, 0.1, 0.25, 0.5, 0.75, 1]:
                print([k, scale, M, alpha])
                for exp in np.arange(nb_exp):
                    ###
                    Z_1 =  generate_mixture_of_gaussians(X_pos,  3, 4, limit_mu = 0.9, limit_sigma_high = 1, limit_sigma_low= 0.1)
                    #Z_2 = generate_mixture_of_cosines(X_pos, 3, 3, low_freq=1, high_freq=5)
                    X = Z_1
                    
                    #### Create signal
                    beta = np.random.uniform(size= X.shape[1])
                    f = X @ beta 
                    Y = np.random.normal(f, scale=scale)
                    
                    
                    Y_pred = S_k @ Y
                    ### Fit a GNN to predict what Y is:
                    #data = Data(x=torch.from_numpy(X).float(),
                    #            y = torch.from_numpy(Y).float(),
                    #            edge_index=edge_index,
                    #            edge_weight=edge_weights)
                    #model = GCN(input_dim= X.shape[1], output_dim=1)
                    #data.train_mask = np.random.binomial(n=1, p=0.75, size=n)
                    #data.test_mask = 1-data.train_mask
                    #train_mask  = np.ones((n,))
                    
                    #for epoch in range(1, 101):
                    #    loss = model.train_data(data, train_mask)
                        #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            
                    #pred = model(data.x, data.edge_index)
                    train_acc = np.mean(np.square(Y_pred - Y)) 
                    res_temp = [exp, 0, k,  scale, M, alpha, float(train_acc)]
                    for u in range(20):
                        new_Y = np.random.normal(f, scale=scale)
                        test_mse = (np.square(Y_pred - new_Y)).mean()
                        res_temp += [float(test_mse)]
                    results[it, :] = res_temp
                    pd.DataFrame(results).to_csv("results/" + namefile)
                    it += 1
                    
                    Y_pred2 = T_k @ Y
                    ### Fit a GNN to predict what Y is:
                    #data = Data(x=torch.from_numpy(X).float(),
                    #            y = torch.from_numpy(Y).float(),
                    #            edge_index=edge_index,
                    #            edge_weight=edge_weights)
                    #model = GCN(input_dim= X.shape[1], output_dim=1)
                    #data.train_mask = np.random.binomial(n=1, p=0.75, size=n)
                    #data.test_mask = 1-data.train_mask
                    #train_mask  = np.ones((n,))
                    
                    #for epoch in range(1, 101):
                    #    loss = model.train_data(data, train_mask)
                        #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            
                    #pred = model(data.x, data.edge_index)
                    train_acc = np.mean(np.square(Y_pred2 - Y)) 
                    res_temp = [exp, 1,  k,  scale, M, alpha, float(train_acc)]
                    for u in range(20):
                        new_Y = np.random.normal(f, scale=scale)
                        test_mse = (np.square(Y_pred2 - new_Y)).mean()
                        res_temp += [float(test_mse)]
                    results[it, :] = res_temp
                    pd.DataFrame(results).to_csv("results/" + namefile)
                    it += 1
                    #print(it)

                
    
        








