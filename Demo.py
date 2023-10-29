# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:32:31 2023

@author: Yahya
"""
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from SSNCC import SSNCC

path = 'moons.csv'
data = pd.read_csv(path)
X = np.array(data.iloc[:, :2])
label = np.array(data.iloc[:, -1])
N = X.shape[0]
number_of_clusters = len(np.unique(label))
    

model = SSNCC(num_clusters = number_of_clusters, gamma = 200)

percentage = 2
idx_p = np.random.choice(np.arange(0, N), int((percentage * N)/100), replace=False).tolist()
idx_c = [j for j in range(X.shape[0]) if j not in idx_p]

model.compute_clusters(X, label[idx_p], idx_p)
clusters = model.cluster_assignments

plt.scatter(X[idx_c, 0], X[idx_c, 1], c = clusters)
