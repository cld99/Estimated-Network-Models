import numpy as np
from scipy import optimize
from MultivariateHawkesProcess import *
from MHP_ADM4_trials import *
import pandas as pd

np.random.seed(0)

"""hawkes parameters"""
mu = [0.1 for _ in range(2)] # background intensity
# alpha = generate_alpha_matrix(len(mu))
beta = 1 # decay; assume beta is the same for all variables
time = 400 # time
num_params = len(mu) # used for flattening/reshaping alpha matrix
# timestamps, _ = simulation_by_cluster_representation(mu, alpha, beta, time) # hawkes

results = []
for i in range(1,15): # range(1,19)

    err_fro_totals = []
    err_rmse_totals = []
    err_fro_adm4_totals = []
    err_rmse_adm4_totals = []
    for _ in range(10): # repeats and takes mean ± st dev
        """generate latent variables and theta"""
        p = len(mu) # number of latent nodes to generate
        d = 2 # dimension of latent space
        sigma_z = 1 # variance of latent space generation
        alph = -5 # constant for theta generation; we call this alpha but i don't want to confuse it with the hawkes alpha
        """sigma_theta is changed every iteration"""
        sigma_theta = i * 0.4 # variance for theta generation

        Z = generate_latent_Z(p, d, sigma_z)
        theta = generate_theta(Z, alph, sigma_theta)
        theta_tilde = logistic(theta)

        if not is_valid_alpha_matrix(theta_tilde):
            print("not valid theta_tilde matrix")
            break
        else:
            should_break = False
            for row in theta_tilde:
                if sum(row) >= 0.8:
                    print("theta_tilde too large, skipped")
                    should_break = True
            if should_break: continue

        
        t, _ = simulation_by_cluster_representation(mu, theta_tilde, beta, time) # hawkes process using theta_tilde

        """optimizing theta"""
        theta = np.array(theta).flatten() # flatten theta_tilde for optimize.fmin_l_bfgs_b
        args = (p, d, Z, alph, sigma_theta, theta_tilde, t, mu, beta, time) # gathers variables
        args_for_adm4 = (0.095, 0.047, t, mu, beta, time, p)

        # for comparison
        # ll = negative_complete_data_log_likelihood_of_theta(theta, *args)
        # print("\nNegative complete-data log-likelihood:")
        # print(ll)

        initial_guess = [0.5 for _ in theta] # arbitrary initial guess
        bounds = []
        for _ in range(len(theta)):
            bounds.append((0.0001, 1-0.0001)) # bounds is (0,1), exclusive
        bounds = np.array(bounds)

        # do the optimization, with auto gradient
        # print('\nOptimize with scipy-calculated gradient:')
        opt_auto_grad = optimize.fmin_l_bfgs_b(func=negative_complete_data_log_likelihood_of_theta,
                                            args=args,
                                            x0=initial_guess,
                                            approx_grad=True) # theta doesn't have bounds; theta_tilde does
        adm4_optimize = optimize.fmin_l_bfgs_b(func=adm4_nll,
                                            args=args_for_adm4,
                                            x0=initial_guess,
                                            bounds=bounds, # estimating alphas, so need bounds
                                            approx_grad=True)
        # for res in opt_auto_grad:
            # print(res)

        # print('\nActual theta values:')
        # print(theta)

        """get results"""
        theta = theta.reshape((p,p)) # reshape theta_tilde

        estimated_theta = opt_auto_grad[0].reshape((p,p))
        estimated_alpha = logistic(estimated_theta) # feed estimated_theta through logistic again for fairer comparison to adm4
        estimated_adm4 = adm4_optimize[0].reshape((p,p))

        err_fro = np.linalg.norm(theta_tilde-estimated_alpha, ord='fro') # frobenius norm
        err_rmse = rmse(np.array(theta_tilde), estimated_alpha)

        err_fro_adm4 = np.linalg.norm(theta_tilde-estimated_adm4, ord='fro') # frobenius norm
        err_rmse_adm4 = rmse(np.array(theta_tilde), estimated_adm4)
        # print("\nRMSE:")
        # print(err)

        """for mean ± st dev"""
        err_fro_totals.append(err_fro)
        err_rmse_totals.append(err_rmse)
        err_fro_adm4_totals.append(err_fro_adm4)
        err_rmse_adm4_totals.append(err_rmse_adm4)

    results.append({'frobenius error mean':np.mean(err_fro_totals),
                    'frobenius error std':np.std(err_fro_totals),
                    "rmse error mean":np.mean(err_rmse_totals),
                    "rmse error std":np.std(err_rmse_totals),
                    'frobenius adm4 mean':np.mean(err_fro_adm4_totals),
                    'frobenius adm4 std':np.std(err_fro_adm4_totals),
                    'rmse adm4 mean':np.mean(err_rmse_adm4_totals),
                    'rmse adm4 std':np.std(err_rmse_adm4_totals),
                    'misspecification':sigma_theta})

    print(f"Finished {i} iterations")

df = pd.DataFrame(results)
df.to_csv('results/MHP_model_misspecification.csv')