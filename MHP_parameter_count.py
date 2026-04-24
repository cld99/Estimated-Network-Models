import numpy as np
from scipy import optimize
from MultivariateHawkesProcess import *
import pandas as pd

results = []
for i in range(2,11):
    """generate hawkes data"""
    # model parameters
    np.random.seed(0)

    """number of parameters is changed every iteration"""
    mu = [0.1 for _ in range(i)] # background intensity
    alpha = [[0.4 / i for _ in range(i)] for _ in range(i)] # mutual excitation matrix; alpha_12 means that 1 is excited by 2
    beta = 1 # decay; assume beta is the same for all variables
    time = 400 # time
    num_params = len(alpha) # used for flattening/reshaping alpha matrix

    timestamps, _ = simulation_by_cluster_representation(mu, alpha, beta, time) # hawkes

    """generate latent variables and theta"""
    np.random.seed(0)
    p = len(mu) # number of latent nodes to generate
    d = 2 # dimension of latent space
    sigma_z = 1 # variance of latent space generation
    alph = -5 # constant for theta generation; we call this alpha but i don't want to confuse it with the hawkes alpha
    sigma_theta = 0.01 # variance for theta generation

    Z = generate_latent_Z(p, d, sigma_z)
    theta = generate_theta(Z, alph, sigma_theta)
    theta_tilde = logistic(theta)

    valid_hawkes = True
    for row in theta_tilde:
        if sum(row) >= 1:
            print("row sums to >= 1")
            valid_hawkes = False
    if not valid_hawkes:
        break

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

    results.append({'frobenius error':err_fro, "rmse error":err_rmse, 'parameter count':i})

df = pd.DataFrame(results)
df.to_csv('results/MHP_parameter_count.csv')