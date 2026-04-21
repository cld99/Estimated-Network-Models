# %%
import numpy as np
from scipy.linalg import toeplitz

seed = 42
np.random.seed(seed)

"""
p: dimension of X
d: dimension of the latent space
alpha: latent interaction baseline parameter
sigma_x: standard deviation for generating X
sigma_theta: standard deviation of noise added to Theta
sigma_y: standard deviation of noise added to y
cov_type: covariance type for X generation
rho: correlation parameter for AR(1) covariance
"""

# z_j ~ N(0,sigma_z^2 I_d)  Z: shape = (p, d)
def generate_latent_Z(p, d, sigma_z):
    Z = np.random.normal(0, sigma_z, size=(p, d))
    return Z

# theta_jk = alpha - |z_j-z_k|^2 + N(0,sigma_theta^2)   Theta: shape = (p, p)
def generate_theta(Z, alpha, sigma_theta):
    p = Z.shape[0]
    Theta = np.zeros((p, p))
    for j in range(p):
        for k in range(j+1, p):
            epsilon_jk = np.random.normal(0, sigma_theta)
            Theta[j, k] = alpha - np.sum((Z[j] - Z[k])**2) + epsilon_jk
            Theta[k, j] = Theta[j, k]
    
    np.fill_diagonal(Theta, 0)
    return Theta

# x_i ~ N(0,sigma_x^2 I_p)   X: shape = (n, p)
def generate_X(n, p, sigma_x, cov_type='iid', rho=0.5):
    if cov_type == 'iid':
        # The covariance between features is 0, they are independent of each other
        # Sigma = sigma_x^2 I_p
        cov = (sigma_x ** 2) * np.eye(p)

    elif cov_type == 'ar1':
        # The diagonal is still sigma_x^2, Adjacent features have a stronger correlation
        # Sigma = sigma_x^2 I_p rho^{|j-k|}
        corr = toeplitz(rho ** np.arange(p))
        cov = (sigma_x ** 2) * corr

    else:
        raise ValueError("cov_type must be 'iid' or 'ar1'")

    mean = np.zeros(p)
    X = np.random.multivariate_normal(mean=mean, cov=cov, size=n)
    return X

# y_i ~ N(beta_0 + sum beta_j x_ij + sum theta_jk x_ij x_ik,sigma_y^2)
def generate_y(X, Theta, beta0, beta, sigma_y):
    n, p = X.shape
    y = np.zeros(n)
    for i in range(n):
        mean_y = beta0 + np.sum(beta * X[i])
        for j in range(p):
            for k in range(j+1,p):
                mean_y += Theta[j, k] * X[i, j] * X[i, k]
        y[i] = np.random.normal(loc=mean_y, scale=sigma_y)
    return y

def simulate_data(n, p, d, alpha, sigma_z, sigma_theta, sigma_x, sigma_y, beta0, beta,
                  cov_type='iid', rho=0.5):
    Z = generate_latent_Z(p, d, sigma_z)
    Theta = generate_theta(Z, alpha, sigma_theta)
    X = generate_X(n, p, sigma_x, cov_type, rho)
    y = generate_y(X, Theta, beta0, beta, sigma_y)
    
    return Z, Theta, X, y
