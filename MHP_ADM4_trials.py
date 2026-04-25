from MultivariateHawkesProcess import *
import numpy as np

# negative log likelihood of the adm4 model
def adm4_nll(alpha, *args):
    ll = negative_log_likelihood(alpha, *args[2:])
    num_params = args[6]

    lam1, lam2 = args[:2]
    alpha = alpha.reshape((num_params,num_params)) # reshape alpha
    ll -= lam1 * np.linalg.norm(alpha, ord='nuc')
    ll -= lam2 * np.linalg.norm(alpha, ord=1)
    return ll

def negative_complete_data_log_likelihood_for_adm4(theta, *args):
    lam1, lam2, p, d, z, alph, sigma_theta, theta_tilde, timestamps, mu, beta, time = args # same args as negative_complete_data_log_likelihood_of_theta, but with lam1 and lam2
    theta = theta.reshape(p,p) # theta requres reshaping

    args = (lam1, lam2, timestamps, mu, beta, time, p) # for hawkes
    return -(-adm4_nll(theta_tilde, *args) + log_p_theta_given_z(sigma_theta, theta, alph, z, p))

if __name__ == "__main__":
    """generate hawkes data"""
    # model parameters
    np.random.seed(0)
    mu = [0.4, 0.4] # background intensity
    alpha = [[0.3, 0.4],
            [0.35, 0.45]] # mutual excitation matrix; alpha_12 means that 1 is excited by 2
    beta = 1 # decay; assume beta is the same for all variables
    time = 400 # time
    num_params = len(alpha) # used for flattening/reshaping alpha matrix

    timestamps, _ = simulation_by_cluster_representation(mu, alpha, beta, time) # hawkes

    """optimize alpha with adm4"""
    lam1, lam2 = 0.02, 0.6
    alpha = np.array(alpha).flatten() # lbfgs requires flattening; will be unflattened later
    args = (lam1, lam2, timestamps, mu, beta, time, num_params) # gathers variables

    # for comparison
    likelihood = negative_log_likelihood(alpha, *args[2:])
    print("\nNegative complete-data log-likelihood:")
    print(likelihood)

    initial_guess = [0.5 for _ in alpha] # arbitrary initialization
    bounds = []
    for i in range(len(alpha)):
        bounds.append((0.0001, 1-0.0001)) # bounds is (0,1), exclusive
    bounds = np.array(bounds)

    # do the optimization, with auto gradient
    print('\noptimize with scipy-calculated gradient:')
    opt_auto_grad = optimize.fmin_l_bfgs_b(func=adm4_nll,
                                            args=args,
                                            x0=initial_guess,
                                            bounds=bounds,
                                            approx_grad=True) # theta doesn't have bounds; theta_tilde does
    # opt_auto_grad = optimize.fmin_l_bfgs_b(func=negative_log_likelihood,
                                            # args=args[2:],
                                            # x0=initial_guess,
                                            # bounds=bounds,
                                            # approx_grad=True) # theta doesn't have bounds; theta_tilde does

    for res in opt_auto_grad:
        print(res)

    print('\nActual alpha values:')
    print(alpha)

    alpha = alpha.reshape((num_params,num_params)) # reshape alpha for future use
    estimated_alpha = opt_auto_grad[0].reshape((num_params, num_params))

    """print errors"""
    err_fro = np.linalg.norm(alpha-estimated_alpha, ord='fro') # frobenius norm
    err_rmse = rmse(alpha, opt_auto_grad[0])
    print("frobenius:", err_fro)
    print("rmse err :", err_rmse)