import numpy as np
from scipy import optimize
from MultivariateHawkesProcess import *
import pandas as pd

"""generate hawkes data"""
# model parameters
np.random.seed(0)
mu = [0.1, 0.1] # background intensity
alpha = [[0.01, 0.025],
         [0.02, 0.015]] # mutual excitation matrix; alpha_12 means that 1 is excited by 2
beta = 1 # decay; assume beta is the same for all variables
time = 400 # time
num_params = len(alpha) # used for flattening/reshaping alpha matrix

timestamps, _ = simulation_by_cluster_representation(mu, alpha, beta, time) # hawkes

results = []
for i in range(1,21):
    """generate latent variables and theta"""
    np.random.seed(0)
    p = len(mu) # number of latent nodes to generate
    d = 2 # dimension of latent space
    sigma_z = 1 # variance of latent space generation
    alph = 1 # constant for theta generation; we call this alpha but i don't want to confuse it with the hawkes alpha
    """sigma_theta is changed every iteration"""
    sigma_theta = i * 0.1 # variance for theta generation

    Z = generate_latent_Z(p, d, sigma_z)
    theta = generate_theta(Z, alph, sigma_theta)
    theta_tilde = logistic(theta)
    t, _ = simulation_by_cluster_representation(mu, theta_tilde, beta, time) # hawkes process using theta_tilde

    """optimizing theta"""
    theta = np.array(theta).flatten() # flatten theta_tilde for optimize.fmin_l_bfgs_b
    args = (p, d, Z, alph, sigma_theta, theta_tilde, t, mu, beta, time) # gathers variables

    # for comparison
    ll = negative_complete_data_log_likelihood_of_theta(theta, *args)
    # print("\nNegative complete-data log-likelihood:")
    # print(ll)

    initial_guess = [0.5 for _ in theta] # arbitrary initial guess

    # do the optimization, with auto gradient
    # print('\nOptimize with scipy-calculated gradient:')
    opt_auto_grad = optimize.fmin_l_bfgs_b(func=negative_complete_data_log_likelihood_of_theta,
                                        args=args,
                                        x0=initial_guess,
                                        approx_grad=True) # theta doesn't have bounds; theta_tilde does
    # for res in opt_auto_grad:
        # print(res)

    # print('\nActual theta values:')
    # print(theta)

    theta = theta.reshape((p,p)) # reshape theta_tilde

    estimated_theta = opt_auto_grad[0].reshape((p,p))

    err_fro = np.linalg.norm(theta-estimated_theta, ord='fro') # frobenius norm
    err_rmse = rmse(theta, opt_auto_grad[0])
    # print("\nRMSE:")
    # print(err)

    results.append({'frobenius error':err_fro, "rmse error":err_rmse, 'misspecification':sigma_theta})

df = pd.DataFrame(results)
df.to_csv('results/MHP_model_misspecification.csv')