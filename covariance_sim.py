import numpy as np

seed = 42
np.random.seed(seed)

"""
p: dimension of X
d: dimension of the latent space
sigma_z: standard deviation of noise added to latent positions
alpha: latent interaction baseline parameter
sigma_theta: standard deviation of noise added to Theta
"""

# Generate latent variables according to Gaussian distribution with specified variance
# z_j ~ N(0,sigma_z^2 I_d)  Z: shape = (p, d)
def generate_latent_Z(p, d, sigma_z):
    Z = np.random.normal(0, sigma_z, size=(p, d))
    return Z

# Generate Theta matrix with entries distributed according to Gaussian distribution with specified
# variance centered at value determined by pairwise distance between latent positions
# theta_jk = N(alpha - |z_j-z_k|^2,sigma_theta^2)   Theta: shape = (p, p)
def generate_theta(Z, alpha, sigma_theta):
    p = Z.shape[0]
    Theta = np.zeros((p, p))
    for j in range(p):
        for k in range(p):
            Theta[j, k] = np.random.normal(alpha - np.sum((Z[j] - Z[k])**2), sigma_theta)
    return Theta

# Ensure covariance matrix is symmetric and positive semidefinite
def generate_cov(Theta):
    p = Theta.shape[0]
    # Symmetrize matrix
    for j in range(p):
        for k in range(j):
            Theta[j, k] = Theta[k, j]
    # Replace diagonal entries to ensure positive definiteness
    for j in range(p):
        Theta[j, j] = np.max(np.abs(Theta[j]))
    return Theta

# Generate samples from multivariate Gaussian with specified covariance matrix
# X = N(0, Theta)
def generate_cov_samples(n, Theta):
    p = Theta.shape[0]
    return np.random.multivariate_normal(mean=np.zeros(p), cov=Theta, size=n)