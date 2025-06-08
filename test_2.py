import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

import MLExtreme as mlx

from utils import *

np.random.seed(16092025)

#####################################################
#####################################################
############## Simulation des données ###############
#####################################################
#####################################################

# Example RV-dirimix  data generation p = 2
n = 10000  # Number of samples
p = 2  # Dimensionality of the simplex (2D in this case)
k = 2  # Number of components in the Dirichlet mixture
alpha = 2.5  # Shape parameter of the Pareto distribution

# Mixture means (Mu), log scale (lnu), and weights (wei)
Mu = np.array([[0.95,0.05], [0.05,0.95]])  # k* p matrix of means
lnu = np.log(20) * np.ones(k)  # log(10) for both components
wei = np.array([1,1])  # weights for the mixture components
Mu, wei = mlx.normalize_param_dirimix(Mu, wei)
# inspect the angular density
Mu_wei = wei  @ Mu
# Display the result
print(Mu_wei)
mlx.plot_pdf_dirimix_2D(Mu, wei, lnu)

# Generate the dataset using the gen_rv_dirimix function.
# As an adversarial (bulk) angular parameter, use the  center of the simplex
X = mlx.gen_rv_dirimix(alpha, Mu, wei, lnu, index_weight_noise=3,
                       Mu_bulk=(np.ones(p)/p).reshape(1, p),
                       wei_bulk=np.ones(1),
                       lnu_bulk=np.ones(1) * np.log(2),
                       size=n)

# Plotting the dataset (2D plot), rescale for easier visualization
X_disp = X**(alpha/4)
idex = np.sum(X**alpha, axis=1) > 10
fig, axs = plt.subplots(1, 2, figsize=(8, 4)) 
axs[0].scatter(X_disp[:, 0], X_disp[:, 1], color='grey', alpha=0.5)
axs[0].set_xlabel('X1')
axs[0].set_ylabel('X2')
max_range = max(np.max(X_disp[:, 0]), np.max(X_disp[:, 1]))
axs[0].set_xlim(0, max_range)
axs[0].set_ylim(0, max_range)
axs[0].set_title('2D Scatter Plot of Generated Points')
axs[1].scatter(X_disp[idex, 0], X_disp[idex, 1], color='grey', alpha=0.5)
axs[1].set_xlabel('X1')
axs[1].set_ylabel('X2')
max_range = max(np.max(X_disp[idex, 0]), np.max(X_disp[idex, 1]))
axs[1].set_xlim(0, max_range)
axs[1].set_ylim(0, max_range)
axs[1].set_title('2D Scatter Plot of Extreme Generated Points')
plt.show()

Y = X.copy().T

## données simulées sous la distribution de référence
X = np.zeros((2,n))

np.random.seed(2025)

Theta = np.random.uniform(0, np.pi/2 ,size=n)
tmp = np.vstack((np.cos(Theta), np.sin(Theta).T)).T
rad = np.sqrt( np.sum(Y**2,axis=0) ) 
X = rad * tmp.T

# store original data
X_real, Y_real = X.copy(), Y.copy()

# normalize data
norm_max = np.sqrt( (Y **2).sum(axis=0) ).max()
X, Y = X/norm_max, Y/norm_max

plt.figure(figsize=(10,10))
plotp(Y, 'r')
plotp(X, 'b')
#plt.axis("off")
plt.grid(True)
#plt.axis([-80, 80, -80, 80])
plt.show()

#####################################################
#####################################################
########### Solve OT and build estimator ############
#####################################################
#####################################################

# solve discrete OT problem
C = distmat(X,Y)
LSA = linear_sum_assignment(C)
T = LSA[1]

# reorder Y so that Y[:,i] is associated to X[:,i]
Y = Y[:,T]
Y_real = Y_real[:,T]

# compute the parameters of the estimator
epsilon_star, psi = compute_params(X,Y)

T = lambda x : smooth_T(x, Y, epsilon_star, psi, norm_max)
T = lambda x : smooth_T(x, Y, epsilon_star, psi, norm_max, lr=epsilon_star/5, steps=3000)

#####################################################
#####################################################
################ Check the solution #################
#####################################################
#####################################################

costs = compute_costs(X, Y)


# check conditions
spread, tmp = check_optimality_cdtn(costs, psi, epsilon_star, epsilon_star)
print( f"Check condition 1 {n*n - tmp.sum()} (should be zero)") # should be zero
np.sort( spread.flatten() )[:20]

spread_bis, tmp_bis = check_optimality_cdtn_bis(X, Y, psi)
print( f"Check condition 2 {n - tmp_bis.sum()} (should be zero)") # should be zero too


# check if the value of the smooth estimator is close to the true value
k = int(2500/5)
T(X_real[:,k]), Y_real[:,k] 
smooth_T(X[:,k], Y, epsilon_star, psi, 1), Y[:,k] 


#####################################################
#####################################################
########### Compute the quantile contour ############
#####################################################
#####################################################

# try to use only the extreme points to compute the smooth interpolation
share = 0.20 # we use share percent of the points to compute the smooth interpolation
norm = np.sqrt(np.sum((Y_real.T)**2, axis=1))
index = (-norm).argsort()

T = lambda x : smooth_T(x, Y[:,index[:int(share*n)]], epsilon_star, psi[index[:int(share*n)]], norm_max, lr=epsilon_star/5, steps=3000)

i = 100
q = 0.999
# arbitrairement on considère p_t (en pourcent) des données comme extrêmes
p_t = 0.1
#pour ce modèle on a 
p = 0 # càd qu'il y a des extrêmes dans toutes les directions
beta = 1 - (1-q)/( p_t * (1-p) )



# new estimator
### estimateur de b(t_n) : \|X\|(k) ###
norm = np.sqrt(np.sum((X_real.T)**2, axis=1))
tmpb = norm.copy()
tmpb.sort()
extr = 0.05#0.08#0.02#0.02#0.07#0.1#p_t
b = tmpb[int(n*(1-extr))]

Theta = np.linspace(0,np.pi/2,i)
tmp = np.vstack((np.cos(Theta), np.sin(Theta).T)).T
rad = b
ref_contour = rad * tmp.T

# compute the quantile contour in the distribution of interest
int_contour = np.zeros((2,i))
for k in range(0,i):
    int_contour[:,k] = T(ref_contour[:,k]) / b

# Johan's estimator
k = int(0.05 * n) 
l = 1 - p_t #0.90
L = 0.01
b_m, b_p = tmpb[n-2*k], tmpb[n-k]
index_ref = np.where(np.logical_and(norm>=b_m, norm<=b_p))[0]
index_target = index_ref
theta_spt_rob = Y_real[:,index_target] / np.sqrt(np.sum((X_real[:,index_ref].T)**2, axis=1))

plt.figure(figsize=(10,10))
plt.scatter(theta_spt_rob[0,:], theta_spt_rob[1,:], s=200, edgecolors="k", c="y", linewidths=2, label = "Johan idea")
plt.plot(int_contour[0,:], int_contour[1,:], 'r', label="Hallin ")
#plt.axis("off")
plt.grid(True)
plt.show()