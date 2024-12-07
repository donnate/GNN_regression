import argparse
import copy
import cvxpy as cp
import numpy as np
import pandas as pd
import scipy as sc
import networkx as nx
import sys, os
import random
from copy import deepcopy

from torch_geometric.nn import GCNConv, SAGEConv
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Data
from scipy.sparse import coo_matrix



def average_signal_within_radius(node, dist_matrix, Y, L, include_node=False):
    """
    Extracts the neighbors of a given node within a radius L and computes the average of their signal Y.

    Parameters:
    node (int): The index of the node.
    dist_matrix (numpy.ndarray): The distance matrix where dist_matrix[i, j] is the distance between node i and node j.
    Y (numpy.ndarray): The signal values on the nodes (same length as number of nodes).
    L (float): The radius within which to consider neighbors.

    Returns:
    float: The average of the signal Y of the neighbors within radius L. If no neighbors are found, return np.nan.
    """
    # Extract the distances from the given node to all other nodes
    distances = dist_matrix[node, :]
    
    # Find indices of neighbors within the radius L (excluding the node itself)
    if include_node:
        neighbors_indices = np.where((distances <= L))[0]  # Include the node itself
    else:
        neighbors_indices = np.where((distances <= L) & (distances > 0))[0]  # Exclude the node itself
    
    # Check if there are any neighbors
    if len(neighbors_indices) == 0:
        return np.nan  # If no neighbors are found, return NaN
    
    # Extract the signal values of the neighbors
    neighbor_signals = Y[neighbors_indices]
    
    # Compute the average of the neighbor signals
    average_signal = np.mean(neighbor_signals)
    
    return average_signal


def incidence_matrix(G):
    edges = list(G.edges())
    n_edges = len(edges)
    n_nodes = G.number_of_nodes()
    D = np.zeros((n_edges, n_nodes))
    
    for idx, (u, v) in enumerate(edges):
        D[idx, u] = 1
        D[idx, v] = -1
    return D


def graph_trend_filtering(Y, G, D, lambda_=0.1):
    # Regularization parameter
    n_nodes = nx.number_of_nodes(G)
    n_edges = nx.number_of_edges(G)
    # Variable to optimize
    x = cp.Variable(n_nodes)

    # Objective function
    objective = cp.Minimize(0.5/n_nodes * cp.sum_squares(x - Y) + lambda_/n_edges * cp.norm1(D @ x))

    # Problem setup
    prob = cp.Problem(objective)
    prob.solve()

    # Check the status
    if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
        x_estimated = x.value
    else:
        raise ValueError("Optimization did not converge!")
    return(x_estimated)


def graph_trend_filtering_l2(Y, G, D, lambda_=0.1):
    # Regularization parameter
    n_nodes = nx.number_of_nodes(G)
    n_edges = nx.number_of_edges(G)
    # Variable to optimize
    x = cp.Variable(n_nodes)

    # Objective function
    objective = cp.Minimize(0.5/n_nodes * cp.sum_squares(x - Y) + lambda_/n_edges * cp.norm2(D @ x))

    # Problem setup
    prob = cp.Problem(objective)
    prob.solve()

    # Check the status
    if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
        x_estimated = x.value
    else:
        raise ValueError("Optimization did not converge!")
    return(x_estimated)
          
def graph_trend_filtering_l22(Y, G, D, lambda_=0.1):
    # Regularization parameter
    n_nodes = nx.number_of_nodes(G)
    n_edges = nx.number_of_edges(G)
    # Variable to optimize
    x = cp.Variable(n_nodes)

    # Objective function
    objective = cp.Minimize(0.5/n_nodes * cp.sum_squares(x - Y) + lambda_/n_edges * cp.sum_squares(D @ x))

    # Problem setup
    prob = cp.Problem(objective)
    prob.solve()

    # Check the status
    if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
        x_estimated = x.value
    else:
        raise ValueError("Optimization did not converge!")
    return(x_estimated)

def CV_signal(G, Y):
    """
    Creates a copy of the input graph and replaces each node's label with
    the label of one of its direct neighbors.

    Parameters:
    - G: networkx.Graph
        The input graph.
    - Y: np.array 
        The signal

    Returns:
    - Y_new: np.array 
        A new graph signal with updated node labels.
    """
    # Create a deep copy of the graph to avoid modifying the original
    Y_new = np.zeros(Y.shape)
    
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if neighbors:
            # Randomly select one of the neighbors
            chosen_neighbor = random.choice(neighbors)
            # Replace the node's label with the neighbor's label
            Y_new[node] = Y[chosen_neighbor]
        else:
            # Handle isolated nodes (no neighbors)
            # Optionally, you can set a default value or leave the label unchanged
            pass  # Labels of isolated nodes remain unchanged
    
    return Y_new
    
    
    
import numpy as np

def relu(x):
    return np.maximum(0, x)


import torch
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv  # Replace GCNConv with another layer if needed

import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nb_layers):
        super().__init__()
        torch.manual_seed(1234567)

        # Dynamically create layers
        self.layers = nn.ModuleList()
        if nb_layers > 1:
            self.layers.append(GCNConv(input_dim, hidden_dim))  # Input layer
        else:
            self.layers.append(GCNConv(input_dim, output_dim))  # Input layer
        
        if nb_layers > 1:
            for _ in range(nb_layers - 2):  # Hidden layers
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
        
            self.layers.append(GCNConv(hidden_dim, output_dim))  # Output layer


    def forward(self, x, edge_index):
        if len(self.layers) > 1:
            for i, conv in enumerate(self.layers):
                x = conv(x, edge_index)
                if i < len(self.layers) - 1:  # No activation on the output layer
                    x = F.relu(x)
                    x = F.dropout(x, p=0.5, training=self.training)
        else:
            for i, conv in enumerate(self.layers):
                x = conv(x, edge_index)
        return x



class SAGEGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nb_layers):
        super().__init__()
        torch.manual_seed(1234567)

        # Dynamically create layers
        self.layers = nn.ModuleList()
        if nb_layers > 1:
            self.layers.append(SAGEConv(input_dim, hidden_dim))  # Input layer
        else:
            self.layers.append(SAGEConv(input_dim, output_dim))  # Input layer
        
        if nb_layers > 1:
            for _ in range(nb_layers - 2):  # Hidden layers
                self.layers.append(SAGEConv(hidden_dim, hidden_dim))
        
            self.layers.append(SAGEConv(hidden_dim, output_dim))  # Output layer


    def forward(self, x, edge_index):
        if len(self.layers) > 1:
            for i, conv in enumerate(self.layers):
                x = conv(x, edge_index)
                if i < len(self.layers) - 1:  # No activation on the output layer
                    x = F.relu(x)
                    x = F.dropout(x, p=0.5, training=self.training)
        else:
            for i, conv in enumerate(self.layers):
                x = conv(x, edge_index)
        return x


def fit_GNN(data, data_GT, GNN_type="GCN", L=1):
    #### Train a neural network
    if GNN_type == "GCN":
        model = GCN(1, 1, 1, L)
    else:
        model = SAGEGCN(1, 1, 1, L)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()

    def train():
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x, data.edge_index)  # Perform a single forward pass.
        loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss

    for epoch in range(1, 1001):
        loss = train()
        #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    # Assuming `data` is your PyTorch Geometric data object and `model` is your trained GCN
    model.eval()  # Set model to evaluation mode

    # Disable gradient computations for testing
    with torch.no_grad():
        # Forward pass
        out = model(data.x, data.edge_index)  # Predictions for all nodes
        # Extract predictions for test nodes
        test_pred = out[data.test_mask]
        test_labels = data.y[data.test_mask]
        train_pred = out[data.train_mask]

        out_GT = model(data_GT.x, data_GT.edge_index)  # Predictions for all nodes
        # Extract predictions for test nodes
        test_pred_GT = out_GT[data_GT.test_mask]
        train_pred_GT = out_GT[data_GT.train_mask]
        # For regression: Compute Mean Squared Error (MSE)
        mse_loss = torch.nn.functional.mse_loss(test_pred, test_labels)
        print(f"Test MSE: {mse_loss:.4f}")
        return(train_pred, test_pred, train_pred_GT, test_pred_GT)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Convert adjacency matrix to edge index
def adjacency_to_edge_index(adjacency_matrix):
    sparse_matrix = coo_matrix(adjacency_matrix)
    edge_index = torch.tensor(np.vstack((sparse_matrix.row, sparse_matrix.col)), dtype=torch.long)
    return edge_index