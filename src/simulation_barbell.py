import argparse
import copy
import numpy as np
import pandas as pd
import scipy as sc
import networkx as nx
import sys, os

sys.path.append('~/Documents/GNN_regression')
from functions_alt import *

parser = argparse.ArgumentParser()
parser.add_argument('--add_node', required=True,  type=int)  # Positional argument
args = parser.parse_args()

beta_matrix = np.array([[-0.1, 0.1],[-1., 1.],
                       [-5., 5],[-10., 10.],
                       [-15., 15.],
                       [-20., 20.],
                        [-50., 50.],
                        [-100., 100.]])
##### GNNs
n_nodes_x = 100
results_train = []
results_test = []
### Create a latent space
np.random.seed(12345)
n_nodes_x =  100
dim_latent = 2
lambda_seq = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1., 2.5, 5., 10, 25, 50, 75, 100, 500]
sparse_seq = [0,0,0,0]
h = [4, 10, 20, 40]
graph_type = [
             "Barbell (m=4)",
              "Barbell (m=10)",
              "Barbell (m=20)",
              "Barbell (m=40)"]
for it, sparsity in enumerate(sparse_seq):
    for index_beta in np.arange(8):
        for exp in np.arange(20):
            ##### Define signal on graph
            G = nx.barbell_graph(h[it], 100 - 2 * h[it], create_using=None)
            U = nx.spectral_layout(G)
            U = pd.DataFrame.from_dict(U).to_numpy().T

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
            Z = np.random.normal(scale = 1., size=n_nodes_x)
            y_true = 5 * np.cos(U.dot(beta))
            Y = y_true + Z
            smoothness = (np.abs(D@y_true)).mean()
            smoothness2 = (np.square(D@y_true)).mean()
            smoothness_max = (np.abs(D@y_true)).max()
            smoothness_cor = (np.abs(D@(np.diag(1./np.sqrt(deg)).dot(y_true)))).max()
            #### Run experiment

            ### Define train and test
            indices = np.arange(n_nodes_x)
            np.random.shuffle(indices)
            train = indices[0:20]
            test = indices[21:40]
            results_train += [[graph_type[it], sparsity, beta[1],  exp, smoothness, smoothness2,smoothness_max, smoothness_cor, "Benchmark",0,  0,np.square(Y[train] - y_true[train]).mean(), 
                               np.square(Y[train] - y_true[train]).mean(), 0, 0]]
            results_test += [[graph_type[it],sparsity, beta[1],  exp, smoothness, smoothness2,smoothness_max, smoothness_cor, "Benchmark",0, 0,np.square(Y[test] - y_true[test]).mean(), 
                                           0, 0, 0]]
            
            print("here")
            fold = 0

            #for fold in np.arange):
            Y_copy = CV_signal(G, Y)
            Y_tilde = copy.deepcopy(Y)
            Y_tilde[list(test) + list(train)] = Y_copy[list(test) + list(train)]
            for L in np.arange(1,10):
                pred_train = [average_signal_within_radius(node, dist_matrix, Y_tilde, L, include_node =(args.add_node==1)) for node in train]
                pred_train_gt = [average_signal_within_radius(node, dist_matrix, y_true, L, include_node =(args.add_node==1)) for node in train]

                results_train += [[graph_type[it],sparsity, beta[1],  exp, smoothness, 
                                    smoothness2,smoothness_max, smoothness_cor, "Local Average", L, 
                                    fold, np.square(pred_train - y_true[train]).mean(), 
                                    np.square(pred_train - Y[train]).mean(),
                                    np.square(np.array(y_true)[train] - np.array(pred_train_gt)).mean(),
                                    np.square(np.array(pred_train_gt) - np.array(pred_train)).mean()]]
                pred_test = [average_signal_within_radius(node, dist_matrix, Y, L, include_node =(args.add_node==1)) for node in test]
                pred_test_gt = [average_signal_within_radius(node, dist_matrix, y_true, L, include_node =(args.add_node==1)) for node in test]
                
                results_test += [[graph_type[it],sparsity, beta[1],  exp, smoothness,
                                    smoothness2,smoothness_max, smoothness_cor, "Local Average", L, 
                                    fold, np.square(pred_test - y_true[test]).mean(), 
                                    np.square(pred_test - Y[test]).mean(),
                                    np.square(np.array(y_true)[test] - np.array(pred_test_gt)).mean(),
                                    np.square(np.array(pred_test_gt) - np.array(pred_test)).mean()]]
                #####
                S_k = np.linalg.matrix_power(S, L)
                T_k = np.linalg.matrix_power(T, L)
                Y_pred_GNN_S = S_k @ Y_tilde
                Y_pred_GNN_T = T_k @Y_tilde
                Y_pred_GNN_S_gt = S_k @ y_true
                Y_pred_GNN_T_gt = T_k @ y_true
                results_train += [[graph_type[it],sparsity, beta[1],  exp, smoothness,smoothness2,smoothness_max, smoothness_cor,  
                                    "GNN: S/GraphSage convolution", L, fold, np.square(Y_pred_GNN_S[train] - y_true[train]).mean(), 
                                    np.square(Y_pred_GNN_S[train] - Y[train]).mean(),
                                    np.square(y_true[train] - Y_pred_GNN_T_gt[train]).mean(),
                                    np.square(Y_pred_GNN_S[train] - Y_pred_GNN_S_gt[train]).mean()]]
                results_train += [[graph_type[it],sparsity, beta[1],  exp, smoothness, 
                                    smoothness2,smoothness_max, smoothness_cor, "GNN: T/GCN convolution", 
                                    L, fold, np.square(Y_pred_GNN_T[train] - y_true[train]).mean(), 
                                    np.square(Y_pred_GNN_T[train] - Y[train]).mean(),
                                    np.square(y_true[train] - Y_pred_GNN_T_gt[train]).mean(),
                                    np.square(Y_pred_GNN_T[train] - Y_pred_GNN_T_gt[train]).mean()]]
                results_test += [[graph_type[it],sparsity, beta[1],  exp, smoothness,
                                    smoothness2,smoothness_max, smoothness_cor,  "GNN: S/GraphSage convolution", 
                                    L, fold, np.square(Y_pred_GNN_S[test] - y_true[test]).mean(), 
                                    np.square(Y_pred_GNN_S[test] - Y[test]).mean(),
                                    np.square(y_true[test] - Y_pred_GNN_T_gt[test]).mean(),
                                    np.square(Y_pred_GNN_T[test] - Y_pred_GNN_T_gt[test]).mean()]]
                results_test += [[graph_type[it],sparsity, beta[1],  exp, smoothness,
                                    smoothness2,smoothness_max, smoothness_cor,  "GNN: T/GCN convolution", 
                                    L, fold, np.square(Y_pred_GNN_T[test] - y_true[test]).mean(), 
                                    np.square(Y_pred_GNN_T[test] - Y[test]).mean(),
                                    np.square(y_true[test] - Y_pred_GNN_T_gt[test]).mean(),
                                    np.square(Y_pred_GNN_T[test] - Y_pred_GNN_T_gt[test]).mean()]]


            # if args.add_node ==1:
            #     for lambda_ in lambda_seq:
            #         graph_trend_predictions = graph_trend_filtering(Y_tilde, G, D, lambda_=lambda_)
            #         results_train += [[graph_type[it],sparsity, beta[1],  exp, smoothness,smoothness2,smoothness_max, smoothness_cor,  "Graph Trend Filtering", lambda_, fold, np.square(graph_trend_predictions[train] - y_true[train]).mean(), 
            #                         np.square(graph_trend_predictions[train] - Y[train]).mean()]]
            #         results_test += [[graph_type[it],sparsity, beta[1],  exp, smoothness,smoothness2,smoothness_max, smoothness_cor,  "Graph Trend Filtering", lambda_, fold, np.square(graph_trend_predictions[test] - y_true[test]).mean(), 
            #                         np.square(graph_trend_predictions[test] - Y[test]).mean()]]

            #         graph_trend_predictions = graph_trend_filtering_l2(Y_tilde, G, D, lambda_=lambda_)
            #         results_train += [[graph_type[it],sparsity, beta[1],  exp, smoothness,smoothness2,smoothness_max, smoothness_cor,  "Graph Smoothing", lambda_, fold, np.square(graph_trend_predictions[train] - y_true[train]).mean(), 
            #                         np.square(graph_trend_predictions[train] - Y[train]).mean()]]
            #         results_test += [[graph_type[it],sparsity, beta[1],  exp, smoothness,smoothness2,smoothness_max, smoothness_cor,  "Graph Smoothing", lambda_, fold, np.square(graph_trend_predictions[test] - y_true[test]).mean(), 
            #                         np.square(graph_trend_predictions[test] - Y[test]).mean()]]
            print([graph_type[it], sparsity, beta[1],  exp, smoothness])

            #print(results_train)
            df_train = pd.DataFrame(results_train, columns = ["graph_type", "sparsity", "beta", "exp",  "smoothness1", "smoothness2",
                                                "smoothness_max", "smoothness_cor",
                                                "Method", "L","fold", "Error", "Prediction",
                                                "Bias", "Variance"])
            df_test = pd.DataFrame(results_test, columns = ["graph_type", "sparsity", "beta", "exp",  "smoothness1", "smoothness2",
                                                "smoothness_max", "smoothness_cor",
                                                "Method", "L","fold", "Error", "Prediction",
                                                "Bias", "Variance"])
            df_train.to_csv("~/Downloads/new_results4_train_simu_barbell_topo_" + str(args.add_node) + "_prediction.csv")
            df_test.to_csv("~/Downloads/new_results4_test_simu_barbell_topo_" + "_" + str(args.add_node) + "_prediction.csv")

