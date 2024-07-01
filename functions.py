import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from networkx import grid_graph
import scipy as sc

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected





def smooth_sobolev_function(L, r, alpha, M):  
    n = L.shape[0]
    eigvalues, eigvectors = sc.linalg.eigh(L.astype(float).toarray())
    #x = np.array([np.log(i/900) for i in np.arange(100, 700)])
    #y = np.log(eigvalues)[100:700]
    #slope, intercept, r_value, p_value, std_err = linregress(x, y)
    #### generate coefficients:
    ###determine growth
    #r = slope
    a_i = np.random.uniform(low=-1, high=1, size=L.shape[0])
    a_i = M * a_i/np.linalg.norm(a_i)
    D = np.diag(1/(1. + n**(2.0/r) * eigvalues)**(alpha/2))
    c_j = D @ a_i
    return(eigvectors @ c_j, eigvectors)


class GCN(torch.nn.Module):
    def __init__(self, input_dim, nb_convolutions=1, output_dim=1):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(input_dim, output_dim)
        #self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        #x = x.relu()
        #x = F.dropout(x, p=0.5, training=self.training)
        #x = self.conv2(x, edge_index)
        return x
        
    def train_data(self, data, train_mask):
          optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
          self.train()
          optimizer.zero_grad()  # Clear gradients.
          out = self.forward(data.x, data.edge_index)  # Perform a single forward pass.
          loss = criterion(out[train_mask], data.y[train_mask])  # Compute the loss solely based on the training nodes.
          loss.backward()  # Derive gradients.
          optimizer.step()  # Update parameters based on gradients.
          return loss

    def test_data(self, data, f, scale):
          new_Y = np.random.normal(f, scale=scale)
          self.eval()
          pred = self.forward(data.x, data.edge_index)
          test_mse = criterion(pred, torch.from_numpy(new_Y).float())  # Check against ground-truth labels.
          return test_mse, pred


def generate_mixture_of_gaussians(X, p, K, limit_mu = 5, limit_sigma_high = 3, limit_sigma_low= 0.5):
    # Define the 2D Gaussian function
    dim_ambiant = X.shape[1]
    def gaussian(X, mu, sigma):
        dim_ambiant = X.shape[1]
        log_exp = 0
        for i in np.arange(dim_ambiant):
            log_exp += (X[:,i] - mu[i])**2 / (2 * sigma[i]**2)
            
        return np.exp(-log_exp)

    Z = np.zeros((X.shape[0], p))
    for i in np.arange(p):
        ### generate mixure weights:
        w = np.random.uniform(low=0., high=1, size=K)
        w = w/np.sum(w)
        z_temp = np.zeros((X.shape[0],))
        for k in np.arange(K):
            mu = np.array(np.random.uniform(low=-limit_mu, high=limit_mu, size=dim_ambiant))
            sigma = np.array(np.random.uniform(low=limit_sigma_low, high=limit_sigma_high, size=dim_ambiant))
            z_temp += w[i] * gaussian(X, mu, sigma)
        Z[:,i] = z_temp
    return(Z)


def generate_mixture_of_cosines(X, p, K, low_freq=0.4, high_freq=5):
    # Define the 2D Gaussian function
    dim_ambiant = X.shape[1]
    def sine_function(X, frequency):
        dim_ambiant = X.shape[1]
        res = 1.
        for i in np.arange(dim_ambiant):
            u = np.random.uniform(size=1)
            #print([i,u])
            if u >0.5:
                res *= np.cos(frequency[i] * X[:,i])
            else:
                res *= np.sin(frequency[i] * X[:,i])
        return(res)

    Z = np.zeros((X.shape[0], p))
    for i in np.arange(p):
        ### generate mixure weights:
        w = np.random.uniform(low=0., high=1, size=K)
        w = w/np.sum(w)
        z_temp = np.zeros((X.shape[0],))
        for k in np.arange(K):
            frequency = np.array(np.random.uniform(low=low_freq, high=high_freq, size=dim_ambiant))
            z_temp += w[i] * sine_function(X, frequency)
        Z[:,i] = z_temp
    return(Z)