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
parser.add_argument('--dataset', required=True, type=str)  # Positional argument
parser.add_argument('--add_node', required=True,  type=int)  # Positional argument
args = parser.parse_args()

edges = pd.read_csv("/Users/cdonnat/Downloads/dataset_regression/" + args.dataset + "_edge_list.txt", header=None,
                   sep = "\t")
edges.columns = ['source', 'target']
nodes = pd.read_csv("/Users/cdonnat/Downloads/dataset_regression/" + args.dataset + "_labels.txt", header=None)
nodes.columns = ['Y']
edges['source'] = edges['source']-1
edges['target'] = edges['target']-1
G = nx.from_pandas_edgelist(edges)
G= G.to_undirected()
G.remove_edges_from(nx.selfloop_edges(G))
print(nx.number_of_nodes(G))
n_nodes_x =  nx.number_of_nodes(G)
D = incidence_matrix(G)

distances = dict(nx.all_pairs_shortest_path_length(G))
dist_matrix = np.zeros((n_nodes_x, n_nodes_x))

# Fill in the distance matrix
for i, node_i in enumerate(G.nodes()):
    for j, node_j in enumerate(G.nodes()):
        try:
            dist_matrix[i, j] = distances[node_i][node_j]
        except:
            dist_matrix[i, j] = 1e8


##### GNNs
results_train = []
results_test = []
### Create a latent space
np.random.seed(12345)
lambda_seq = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1., 2.5, 5., 10, 25, 50, 75, 100, 500]

y_true = nodes['Y']
Y = nodes['Y']
if args.add_node ==1:
    Atilde = nx.adjacency_matrix(G).todense() + np.eye(n_nodes_x)
else:
    Atilde = nx.adjacency_matrix(G).todense()
deg = Atilde.sum(1)
deg[np.where(deg==0)]=1
T = np.diag(1./deg).dot(Atilde)
S = np.diag(1./np.sqrt(deg)).dot(Atilde.dot(np.diag(1./np.sqrt(deg))))
smoothness = (np.abs(D@y_true)).mean()
smoothness2 = (np.square(D@y_true)).mean()
smoothness_max = (np.abs(D@y_true)).max()
smoothness_cor = (np.abs(D@(np.diag(1./np.sqrt(deg)).dot(y_true)))).max()
#### Run experiment

### Define train and test
for exp in np.arange(50):
    indices = np.arange(n_nodes_x)
    np.random.shuffle(indices)
    train = indices[0:500]
    test = indices[501:1000]

    for fold in np.arange(5):
        Y_copy = CV_signal(G, Y)
        Y_tilde = copy.deepcopy(Y)
        Y_tilde[list(test) + list(train)] = Y_copy[list(test) + list(train)]
        results_train += [[exp, smoothness, smoothness2,smoothness_max, smoothness_cor, "Benchmark",0,  0,np.square(Y_tilde[train] - y_true[train]).mean(), 
                       np.square(Y_tilde[train] - y_true[train]).mean() ]]
        results_test += [[exp, smoothness, smoothness2,smoothness_max, smoothness_cor, "Benchmark",0, 0,np.square(Y_tilde[test] - y_true[test]).mean(), 
                                       0]]
    
    
        for L in np.arange(1,10):
            pred_train = [average_signal_within_radius(node, dist_matrix, Y_tilde, L, include_node=(args.add_node==1)) for node in train]
            results_train += [[exp, smoothness, smoothness2,smoothness_max, smoothness_cor, "Local Average", L, fold, np.square(pred_train - y_true[train]).mean(), 
                               np.square(pred_train - Y[train]).mean()]]
            pred_test = [average_signal_within_radius(node, dist_matrix, Y, L, include_node=(args.add_node==1)) for node in test]
            results_test += [[exp, smoothness,smoothness2,smoothness_max, smoothness_cor, "Local Average", L, fold, np.square(pred_test - y_true[test]).mean(), 
                              np.square(pred_test - Y[test]).mean()]]
            #####
            S_k = np.linalg.matrix_power(S, L)
            T_k = np.linalg.matrix_power(T, L)
            Y_pred_GNN_S = S_k @ Y_tilde
            Y_pred_GNN_T = T_k @Y_tilde
            results_train += [[exp, smoothness,smoothness2,smoothness_max, smoothness_cor,  "GNN (S-convolution)", L, fold, np.square(Y_pred_GNN_S[train] - y_true[train]).mean(), 
                          np.square(Y_pred_GNN_S[train] - Y[train]).mean()]]
            results_train += [[ exp, smoothness, smoothness2,smoothness_max, smoothness_cor, "GNN (T-convolution)", L, fold, np.square(Y_pred_GNN_T[train] - y_true[train]).mean(), 
                              np.square(Y_pred_GNN_T[train] - Y[train]).mean()]]
            results_test += [[ exp, smoothness,smoothness2,smoothness_max, smoothness_cor,  "GNN (S-convolution)", L, fold, np.square(Y_pred_GNN_S[test] - y_true[test]).mean(), 
                              np.square(Y_pred_GNN_S[test] - Y[test]).mean()]]
            results_test += [[ exp, smoothness,smoothness2,smoothness_max, smoothness_cor,  "GNN (T-convolution)", L, fold, np.square(Y_pred_GNN_T[test] - y_true[test]).mean(), 
                              np.square(Y_pred_GNN_T[test] - Y[test]).mean()]]
        if args.add_node ==1:
            for lambda_ in lambda_seq:
                    graph_trend_predictions = graph_trend_filtering(Y_tilde, G, D, lambda_=lambda_)
                    results_train += [[ exp, smoothness,smoothness2,smoothness_max, smoothness_cor,
                                         "Graph Trend Filtering", lambda_, fold, np.square(graph_trend_predictions[train] - y_true[train]).mean(), 
                                    np.square(graph_trend_predictions[train] - Y[train]).mean()]]
                    results_test += [[  exp, smoothness,smoothness2,smoothness_max, smoothness_cor,
                                       "Graph Trend Filtering", lambda_, fold, np.square(graph_trend_predictions[test] - y_true[test]).mean(), 
                                    np.square(graph_trend_predictions[test] - Y[test]).mean()]]
                    
                    graph_trend_predictions = graph_trend_filtering_l2(Y_tilde, G, D, lambda_=lambda_)
                    results_train += [[ exp, smoothness,smoothness2,smoothness_max, smoothness_cor,
                                         "Graph Smoothing (norm)", lambda_, fold, np.square(graph_trend_predictions[train] - y_true[train]).mean(), 
                                    np.square(graph_trend_predictions[train] - Y[train]).mean()]]
                    results_test += [[ exp, smoothness,smoothness2,smoothness_max, smoothness_cor,
                                        "Graph Smoothing (norm)", lambda_, fold, np.square(graph_trend_predictions[test] - y_true[test]).mean(), 
                                    np.square(graph_trend_predictions[test] - Y[test]).mean()]]
                    
                    graph_trend_predictions = graph_trend_filtering_l22(Y_tilde, G, D, lambda_=lambda_)
                    results_train += [[ exp, smoothness,smoothness2,smoothness_max, smoothness_cor,
                                         "Graph Smoothing", lambda_, fold, np.square(graph_trend_predictions[train] - y_true[train]).mean(), 
                                    np.square(graph_trend_predictions[train] - Y[train]).mean()]]
                    results_test += [[ exp, smoothness,smoothness2,smoothness_max, smoothness_cor,
                                        "Graph Smoothing", lambda_, fold, np.square(graph_trend_predictions[test] - y_true[test]).mean(), 
                                    np.square(graph_trend_predictions[test] - Y[test]).mean()]]


        print([exp, smoothness])
        df_train = pd.DataFrame(results_train, columns = [ "exp", "smoothness1", "smoothness2", "smoothness_max", "smoothness_cor",
                                                  "Method", "L","fold", "Error", "Prediction"])
        df_test = pd.DataFrame(results_test, columns = ["exp", "smoothness1", "smoothness2","smoothness_max", "smoothness_cor",
                                                "Method", "L","fold", "Error", "Prediction"])
        df_train.to_csv("~/Downloads/new_results_train_" + args.dataset  + "_" + str(args.add_node) + "_prediction.csv")
        df_test.to_csv("~/Downloads/new_results_test_" + args.dataset  + "_" + str(args.add_node) + "_prediction.csv")
