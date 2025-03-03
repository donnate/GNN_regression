import argparse
import copy
import numpy as np
import pandas as pd
import scipy as sc
import networkx as nx
import sys, os
from torch_geometric.nn import GCNConv, SAGEConv
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Data
from scipy.sparse import coo_matrix

#sys.path.append('~/Documents/GNN_regression')
sys.path.append('/scratch/midway3/cdonnat/GNN_regression/GNN_regression/GNN_regression')
from functions_alt import *

parser = argparse.ArgumentParser()
parser.add_argument('--add_node', required=True,  type=int)  # Positional argument
parser.add_argument('--noise', required=True,  type=float) 
parser.add_argument('--non_linear', required=True,  type=int) 
parser.add_argument('--seed', required=True,  type=int) 

args = parser.parse_args()

beta_matrix = np.array([[-0.1, 0.1],[-1., 1.],
                       [-2., 2],[-5., 5],
                       [-10., 10.],
                       [-15., 15.],
                       [-20., 20.],
                        [-100., 100.]])
##### GNNs
n_nodes_x = 100
results_train = []
results_test = []
### Create a latent space
np.random.seed(12345)
n_nodes_x =  100
dim_latent = 2
graph_type = "latent"
lambda_seq = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1., 2.5, 5., 10, 25, 50, 75, 100, 500]
sparse_seq = [0., 0.05, 0.1, 0.2, 0.5, 0.75,  0.95]


                        
for it_exp, exp in enumerate([args.seed]):
    np.random.seed(exp)
    for scale_noise  in [args.noise]:
        for sparsity in sparse_seq:
            for index_beta in np.arange(8):
                ##### Define signal on graph
                n_nodes_x = 100
                U = np.random.uniform(size=n_nodes_x * dim_latent ).reshape([n_nodes_x, dim_latent])
                ##### create associated GraphSageConv
                dist = sc.spatial.distance_matrix(U, U)
                W = 1 / (1 + np.exp(5 *dist))
                np.fill_diagonal(W, 0)
                A = np.random.binomial(n=1, p=W)
                degree = A.sum(1)
                connected_nodes = np.where(degree>0)[0]
                A = A[connected_nodes,:][:,connected_nodes]
                U = U[connected_nodes,:]
                G = nx.from_numpy_array(A)
                for u, v in G.edges:
                    G[u][v]['weight'] = 1

                mst = nx.minimum_spanning_tree(G, weight='weight')
                if sparsity > 0:
                    for edge in list(G.edges()):
                        if not mst.has_edge(*edge):  # If the edge is not part of the MST
                            if np.random.rand() < sparsity:  # With probability p, remove the edge
                                G.remove_edge(*edge)

                n_nodes_x =  nx.number_of_nodes(G)
                D = incidence_matrix(G)


                distances = dict(nx.all_pairs_shortest_path_length(G))
                dist_matrix = np.zeros((n_nodes_x, n_nodes_x))

                # Fill in the distance matrix
                for i, node_i in enumerate(G.nodes()):
                    for j, node_j in enumerate(G.nodes()):
                        dist_matrix[i, j] = distances[node_i][node_j]

                if args.add_node ==1:
                    Atilde = nx.adjacency_matrix(G).todense() + np.eye(n_nodes_x)
                else:
                    Atilde = nx.adjacency_matrix(G).todense()
                deg = Atilde.sum(1)
                S = np.diag(1./deg).dot(Atilde)
                T = np.diag(1./np.sqrt(deg)).dot(Atilde.dot(np.diag(1./np.sqrt(deg))))

                beta = beta_matrix[index_beta,:]
                Z = np.random.normal(scale = scale_noise, size=n_nodes_x)
                y_true = 2 * np.cos(U.dot(beta))
                if args.non_linear == 1:
                    y_true = relu(y_true) + 0.5
                
                Y = y_true + Z

                edge_index = adjacency_to_edge_index(A)

                # Convert node features and signal to tensors
                X = torch.tensor(Y, dtype=torch.float).reshape([-1,1])
                y = torch.tensor(Y, dtype=torch.float).reshape([-1,1])

                X_GT = torch.tensor(y_true, dtype=torch.float).reshape([-1,1])

                # Create PyTorch Geometric data object
                data = Data(x=X, edge_index=edge_index, y=y)

                data_GT = Data(x=X_GT, edge_index=edge_index, y=y)


                smoothness = (np.abs(D@y_true)).mean()
                smoothness2 = (np.square(D@y_true)).mean()
                smoothness_max = (np.abs(D@y_true)).max()
                smoothness_cor = (np.abs(D@(np.diag(1./np.sqrt(deg)).dot(y_true)))).max()
                #### Run experiment

                ### Define train and test
                indices = np.arange(n_nodes_x)
                np.random.shuffle(indices)
                train_index = indices[0:20]
                test_index = indices[21:40]
          
                
                results_train += [[graph_type, scale_noise,  sparsity, beta[1],  exp, smoothness, smoothness2,smoothness_max, smoothness_cor,
                                   "Benchmark",0,  0,np.square(Y[train_index] - y_true[train_index]).mean(), 
                                    0,
                                    0,
                                    np.square(Y[train_index] - y_true[train_index]).mean() ]]
                results_test += [[graph_type, scale_noise,  sparsity, beta[1],  exp, smoothness, smoothness2,smoothness_max, smoothness_cor,
                                   "Benchmark",0,  0,np.square(Y[test_index] - y_true[test_index]).mean(), 
                                    0,
                                    0,
                                    np.square(Y[test_index] - y_true[test_index]).mean() ]]
                
                
                train_mask = torch.zeros(n_nodes_x, dtype=torch.bool)
                test_mask = torch.zeros(n_nodes_x, dtype=torch.bool)
                train_mask[train_index] = True
                test_mask[test_index] = True


                # Add masks to the data object
                data.train_mask = train_mask
                data.test_mask = test_mask
                data_GT.train_mask = train_mask
                data_GT.test_mask = test_mask

                print("here")
                fold = 0

                #for fold in np.arange):
                Y_copy = CV_signal(G, Y)
                Y_tilde = copy.deepcopy(Y)
                Y_tilde[list(test_index) + list(train_index)] = Y_copy[list(test_index) + list(train_index)]

                for L in np.arange(1,10):


                    pred_train = [average_signal_within_radius(node, dist_matrix, Y_tilde, L, include_node =(args.add_node==1)) for node in train_index]
                    pred_train_gt = [average_signal_within_radius(node, dist_matrix, y_true, L, include_node =(args.add_node==1)) for node in train_index]

                    results_train += [[ graph_type, scale_noise,sparsity, beta[1],  exp, smoothness, 
                                        smoothness2,smoothness_max, smoothness_cor, "Local Average", L, 
                                        fold, np.square(pred_train - y_true[train_index]).mean(), 
                                        np.square(pred_train - Y[train_index]).mean(),
                                        np.square(np.array(y_true)[train_index] - np.array(pred_train_gt)).mean(),
                                        np.square(np.array(pred_train_gt) - np.array(pred_train)).mean()]]
                    pred_test = [average_signal_within_radius(node, dist_matrix, Y, L, include_node =(args.add_node==1)) for node in test_index]
                    pred_test_gt = [average_signal_within_radius(node, dist_matrix, y_true, L, include_node =(args.add_node==1)) for node in test_index]
                    
                    results_test += [[ graph_type, scale_noise,sparsity, beta[1],  exp, smoothness,
                                        smoothness2,smoothness_max, smoothness_cor, "Local Average", L, 
                                        fold, np.square(pred_test - y_true[test_index]).mean(), 
                                        np.square(pred_test - Y[test_index]).mean(),
                                        np.square(np.array(y_true)[test_index] - np.array(pred_test_gt)).mean(),
                                        np.square(np.array(pred_test_gt) - np.array(pred_test)).mean()]]
                    #####
                    train_pred, test_pred, train_pred_gt, test_pred_gt = fit_GNN(data, data_GT, GNN_type="GCN", L=L)
                   
                
                    results_train += [[ graph_type, scale_noise, sparsity, beta[1],  exp, smoothness,smoothness2,smoothness_max, smoothness_cor,  
                                        "GNN: Learned", L, fold, np.square(train_pred.numpy() - y_true[train_index]).mean(), 
                                        np.square(train_pred.numpy() - Y[train_index]).mean(),
                                        np.square(np.array(y_true)[train_index] - train_pred_gt.numpy()).mean(),
                                        np.square(train_pred_gt.numpy() - train_pred.numpy()).mean()]]
                    results_test += [[ graph_type, scale_noise,sparsity, beta[1],  exp, smoothness,
                                        smoothness2,smoothness_max, smoothness_cor,   "GNN: Learned", 
                                        L, fold, np.square(test_pred.numpy() - y_true[test_index]).mean(), 
                                        np.square(test_pred.numpy() - Y[test_index]).mean(),
                                        np.square(np.array(y_true)[test_index] - test_pred_gt.numpy()).mean(),
                                        np.square(test_pred_gt.numpy() - test_pred.numpy()).mean()]]
                    
                    train_pred, test_pred, train_pred_gt, test_pred_gt = fit_GNN(data, data_GT, GNN_type="SAGEGCN", L=L)
                   

                    results_train += [[ graph_type, scale_noise, sparsity, beta[1],  exp, smoothness,smoothness2,smoothness_max, smoothness_cor,  
                                        "GNN-SAGE: Learned", L, fold, np.square(train_pred.numpy() - y_true[train_index]).mean(), 
                                        np.square(train_pred.numpy() - Y[train_index]).mean(),
                                         np.square(np.array(y_true)[train_index] - train_pred_gt.numpy()).mean(),
                                        np.square(train_pred_gt.numpy() - train_pred.numpy()).mean()]]
                    results_test += [[ graph_type, scale_noise,sparsity, beta[1],  exp, smoothness,
                                        smoothness2,smoothness_max, smoothness_cor,   "GNN-SAGE: Learned", 
                                        L, fold, np.square(test_pred.numpy() - y_true[test_index]).mean(), 
                                        np.square(test_pred.numpy() - Y[test_index]).mean(),
                                        np.square(np.array(y_true)[test_index] - test_pred_gt.numpy()).mean(),
                                        np.square(test_pred_gt.numpy() - test_pred.numpy()).mean()]]

                    S_k = np.linalg.matrix_power(S, L)
                    T_k = np.linalg.matrix_power(T, L)
                    Y_pred_GNN_S = S_k @ Y_tilde
                    Y_pred_GNN_T = T_k @Y_tilde
                    Y_pred_GNN_S_gt = S_k @ y_true
                    Y_pred_GNN_T_gt = T_k @ y_true
                    results_train += [[ graph_type, scale_noise,sparsity, beta[1],  exp, smoothness,smoothness2,smoothness_max, smoothness_cor,  
                                        "GNN: S/GraphSage convolution", L, fold, np.square(Y_pred_GNN_S[train_index] - y_true[train_index]).mean(), 
                                        np.square(Y_pred_GNN_S[train_index] - Y[train_index]).mean(),
                                        np.square(y_true[train_index] - Y_pred_GNN_T_gt[train_index]).mean(),
                                        np.square(Y_pred_GNN_S[train_index] - Y_pred_GNN_S_gt[train_index]).mean()]]
                    results_train += [[ graph_type, scale_noise,sparsity, beta[1],  exp, smoothness, 
                                        smoothness2,smoothness_max, smoothness_cor, "GNN: T/GCN convolution", 
                                        L, fold, np.square(Y_pred_GNN_T[train_index] - y_true[train_index]).mean(), 
                                        np.square(Y_pred_GNN_T[train_index] - Y[train_index]).mean(),
                                        np.square(y_true[train_index] - Y_pred_GNN_T_gt[train_index]).mean(),
                                        np.square(Y_pred_GNN_T[train_index] - Y_pred_GNN_T_gt[train_index]).mean()]]
                    results_test += [[ graph_type, scale_noise,sparsity, beta[1],  exp, smoothness,
                                        smoothness2,smoothness_max, smoothness_cor,  "GNN: S/GraphSage convolution", 
                                        L, fold, np.square(Y_pred_GNN_S[test_index] - y_true[test_index]).mean(), 
                                        np.square(Y_pred_GNN_S[test_index] - Y[test_index]).mean(),
                                        np.square(y_true[test_index] - Y_pred_GNN_T_gt[test_index]).mean(),
                                        np.square(Y_pred_GNN_T[test_index] - Y_pred_GNN_T_gt[test_index]).mean()]]
                    results_test += [[ graph_type, scale_noise,sparsity, beta[1],  exp, smoothness,
                                        smoothness2,smoothness_max, smoothness_cor,  "GNN: T/GCN convolution", 
                                        L, fold, np.square(Y_pred_GNN_T[test_index] - y_true[test_index]).mean(), 
                                        np.square(Y_pred_GNN_T[test_index] - Y[test_index]).mean(),
                                        np.square(y_true[test_index] - Y_pred_GNN_T_gt[test_index]).mean(),
                                        np.square(Y_pred_GNN_T[test_index] - Y_pred_GNN_T_gt[test_index]).mean()]]


                    # if args.add_node ==1:
                    #     for lambda_ in lambda_seq:
                    #         graph_trend_predictions = graph_trend_filtering(Y_tilde, G, D, lambda_=lambda_)
                    #         results_train += [[ graph_type, scale_noise,sparsity, beta[1],  exp, smoothness,smoothness2,smoothness_max, smoothness_cor,  "Graph Trend Filtering", lambda_, fold, np.square(graph_trend_predictions[train_index] - y_true[train_index]).mean(), 
                    #                         np.square(graph_trend_predictions[train_index] - Y[train_index]).mean()]]
                    #         results_test += [[ graph_type, scale_noise,sparsity, beta[1],  exp, smoothness,smoothness2,smoothness_max, smoothness_cor,  "Graph Trend Filtering", lambda_, fold, np.square(graph_trend_predictions[test_index] - y_true[test_index]).mean(), 
                    #                         np.square(graph_trend_predictions[test_index] - Y[test_index]).mean()]]

                    #         graph_trend_predictions = graph_trend_filtering_l2(Y_tilde, G, D, lambda_=lambda_)
                    #         results_train += [[ graph_type, scale_noise,sparsity, beta[1],  exp, smoothness,smoothness2,smoothness_max, smoothness_cor,  "Graph Smoothing", lambda_, fold, np.square(graph_trend_predictions[train_index] - y_true[train_index]).mean(), 
                    #                         np.square(graph_trend_predictions[train_index] - Y[train_index]).mean()]]
                    #         results_test += [[ graph_type, scale_noise,sparsity, beta[1],  exp, smoothness,smoothness2,smoothness_max, smoothness_cor,  "Graph Smoothing", lambda_, fold, np.square(graph_trend_predictions[test_index] - y_true[test_index]).mean(), 
                    #                         np.square(graph_trend_predictions[test_index] - Y[test_index]).mean()]]
                    print([ graph_type, scale_noise, sparsity, beta[1],  exp, smoothness])

                    #print(results_train)
                    df_train = pd.DataFrame(results_train, columns = ["graph_type", "noise_level", "sparsity", "beta", "exp",  "smoothness1", "smoothness2",
                                                        "smoothness_max", "smoothness_cor",
                                                        "Method", "L","fold", "Error", "Prediction",
                                                        "Bias", "Variance"])
                    df_test = pd.DataFrame(results_test, columns = ["graph_type", "noise_level", "sparsity", "beta", "exp",  "smoothness1", "smoothness2",
                                                        "smoothness_max", "smoothness_cor",
                                                        "Method", "L","fold", "Error", "Prediction",
                                                        "Bias", "Variance"])
                    df_train.to_csv("results/new_results4_non_linearity_train_simu_latent_topo_NL" + str(args.non_linear) + "_seed" +  str(args.seed)+  "_add_nodes"+str(args.add_node)  + "_prediction.csv")
                    df_test.to_csv("results/new_results4_non_linearity_test_simu_latent_topo_NL" + str(args.non_linear) + "_seed" +   str(args.seed)+  "_add_nodes"+str(args.add_node)  + "_prediction.csv")


                
