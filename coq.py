import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

plotp = lambda x,col: plt.scatter(x[0,:], x[1,:], s=200, edgecolors="k", c=col, linewidths=2)

def distmat(x,y):
    return np.sum(x**2,0)[:,None] + np.sum(y**2,0)[None,:] - 2*x.transpose().dot(y)

def compute_d(costs,n):
    d = np.ones((n+1,n)) * np.inf
    d[0,0] = 0

    # is there a way to vectorize this further to reach 
    # better performance thanks to numpy ?
    for k in range(0,n):
        d[k+1,:] = (d[k,:] + costs.T).min(axis=1)

    return d

def compute_costs(X, Y):
    n = X.shape[1]
    tmp = X.T @ Y # (x_i | y_j)
    costs = (tmp.diagonal() - tmp.T).T
    del(tmp)
    costs[ np.eye(n)==1 ] = np.inf
    return costs

def compute_params(X, Y):
    n = X.shape[1]
    
    costs = compute_costs(X, Y)

    d = compute_d(costs,n)
    
    epsilon_star = ( ( ((d[n,:] - d)[:-1,:]).T / (n-np.arange(0,n)) ).T ).max(axis=0).min()

    d_bis = compute_d(costs - epsilon_star, n)
    shortest_paths = d_bis[:-1,:].min(axis=0)
    psi = - shortest_paths

    return epsilon_star, psi

def check_optimality_cdtn(costs, psi, epsilon_star, eps):
    n = costs.shape[0]
    tmp = np.zeros((n,n))
    spread = np.zeros((n,n))
    for i in range(0,n):
        for j in range(0,n):
            spread[i,j] = ( costs[i,j] - (psi[i] - psi[j] + epsilon_star) )
            tmp[i,j] = 1 if spread[i,j] >= - eps else 0
    return spread, tmp

def check_optimality_cdtn_bis(X, Y, psi):
    n = X.shape[1]
    tmp_bis = np.zeros(n)
    spread = np.zeros(n)
    for i in range(0,n):
        x = X[:,i]
        y = Y[:,i]
        spread[i] = ( (np.dot( x, Y) - psi).max() - (np.dot( x, y) - psi[i]) )
        tmp_bis[i] = 1 if spread[i] == 0 else 0
    return spread, tmp_bis

def smooth_T(x, Y, epsilon, psi, norm_max, lr=None, steps=None):
    psi = torch.tensor(psi,requires_grad=False)
    Y = torch.tensor(Y,requires_grad=False)

    x = torch.tensor(x / norm_max, dtype=torch.float64, requires_grad=False)
    #x = torch.tensor(x, dtype=torch.float64, requires_grad=False)

    y_0 = torch.tensor( x.clone().detach(), dtype=torch.float64, requires_grad=True)
    if lr is not None:
        optimizer = torch.optim.SGD([y_0], lr=lr)
    else:
        optimizer = torch.optim.SGD([y_0], lr=epsilon)
    steps = Y.shape[1] if steps is None else steps
    for i in range(steps):
        optimizer.zero_grad()
        phi_y = (torch.matmul( y_0, Y) - psi).max()
        loss = (phi_y + (y_0-x).pow(2).sum() / (2*epsilon))
        loss.backward()
        optimizer.step()

    return (norm_max * (x-y_0) / epsilon).detach().numpy()
    #return ( (x-y_0) / epsilon).detach().numpy()