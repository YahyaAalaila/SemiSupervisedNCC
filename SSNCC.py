# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 02:18:46 2023

@author: Yahya

# Implementation of the SSNCC (Semi-Supervised Normalized Cut Clustering) algorithm in Python.
# Author: Yahya Aalaila, Sami Bamansour, Mouad Elhamdi and Diego Peluffo.
# Year: 2023
# Title: Semi-supervised NCC: Introducing Prior Knowledge to Normalized Cut Clustering
# Published as a preprint in Techrxiv
# DOI: 10.36227/techrxiv.24290146
"""
import numpy as np
from scipy.linalg import fractional_matrix_power
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.cluster._spectral import discretize

class SSNCC: 
    """
    A class representing the Semi-Supervised Normalized Cut Clustering algorithm.

    Parameters:
    -----------
    X : array-like
        The input data matrix.

    num_clusters : int
        The number of clusters to form.

    known_labels : array-like
        The vector of known labels.
        
    prior_idx : array-like
        The vector containing the indexes of known labels in the original data.

    gamma : float, optional
        The kernel bandwidth parameter for the RBF kernel. Defaults to 1.0.
    """
    def __init__(self, num_clusters, gamma):
        """
       Initialize the SSNCC class with the provided parameters.

       Parameters:
       -----------
       num_clusters : int
           The number of clusters to form.

       gamma : float, optional
           The kernel bandwidth parameter for the RBF kernel. Defaults to 1.0.
       """
        if gamma<=0:
            raise ValueError("The RBF hyperparameter must be a non-negative value.")
        elif num_clusters<2:
            raise ValueError("Number of Clusters must be greater or equal to 2.")   
            
        self.K = num_clusters
        self.gamma = gamma
        
        self.N = None
        self.P = None
        self.D = None
        self.A = None
        self.cluster_assignments = None
        
    def Adjacency_matrix(self, X):
        self.N = X.shape[0]
        W = pairwise_kernels(X, metric = 'rbf', gamma = self.gamma)  
        np.fill_diagonal(W, 0)
        self.D = np.diag(W.sum(1))
        sqrt_invD = fractional_matrix_power(self.D, -0.5)
        P = sqrt_invD @ (W @ sqrt_invD)
        return P
        
    def _encode_PriorKnowledge(self, known_labels):
        M_p = np.zeros((len(known_labels), self.K))

        for i, cluster in enumerate(known_labels):
            M_p[i][cluster] = 1
            
        A = np.zeros((self.N, self.K))
        A[:M_p.shape[0],:] = M_p
        return A
    
    def organizer(self, X, prior_idx):

        if self.P is None:
            self.P = self.Adjacency_matrix(X)
            
        unknwon_idx = [j for j in range(self.N) if j not in prior_idx]
        
        P_org, D_org = np.zeros((self.N, self.N)), np.zeros((self.N, self.N))
        for organized, original in zip([P_org, D_org],[self.P, self.D]):
            organized[len(prior_idx):, len(prior_idx):] = original[np.ix_(unknwon_idx, unknwon_idx)]
            organized[len(prior_idx):, :len(prior_idx)] = original[np.ix_(unknwon_idx, prior_idx)]
            organized[:len(prior_idx), len(prior_idx):] = original[np.ix_(prior_idx, unknwon_idx)]
            organized[:len(prior_idx), :len(prior_idx)] = original[np.ix_(prior_idx, prior_idx)]
        return P_org, D_org
        
        
    def compute_clusters(self, X, known_labels, prior_knowledge_indexes):
        
        if len(known_labels) != len(prior_knowledge_indexes):
            raise ValueError("The known labels and their indexes must be of the same length.")
        
        P, D = self.organizer(X, prior_knowledge_indexes)
        sqrt_D, sqrt_invD = fractional_matrix_power(D, 0.5), fractional_matrix_power(D, -0.5)
        
        A = self._encode_PriorKnowledge(known_labels) 
        f = abs(np.identity(self.K) - (A.T @ A)).sum()
        B = (f/self.K) * P
        PA = P @ (sqrt_D @ A)
        b = np.sqrt(f) * PA
        # Solving the inhomogeneous eigenvalue problem.
        E = np.block([
              [np.zeros((self.N, self.N)), np.identity(self.N)],
              [b.dot(b.T) - B.dot(B.T), B + B.T]
          ])
        
        eigvalues = np.linalg.eigvals(E)
        eigvals = eigvalues.real
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        inh_lam = eigvals[-self.K:]
        
        eigvec = np.zeros((self.N, self.K))
        for j in range(self.K): # calculating for each \lambda_j separately.
          B_lam = P - inh_lam[j] * np.identity(self.N)
          eigvec[:,j]  = np.linalg.solve(B_lam, PA[:,j])
        
        cont_assignemnts = np.sqrt(f) * sqrt_invD @ eigvec
        self.cluster_assignments = discretize(cont_assignemnts[len(known_labels):,:], max_svd_restarts = 1000,  random_state = True , n_iter_max= 1000)
    

    
        
        
    
    
