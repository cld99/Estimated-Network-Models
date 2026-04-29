import numpy as np
import torch
import pandas as pd
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.metrics import root_mean_squared_error
from covariance_model import CovarianceModel

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Set datatype to float64 to reduce errors due to precision
torch.set_default_dtype(torch.float64)

params = {
    'n_features': 100,
    'n_samples': 50,
    'embedding_dim': 100,
    'sigma_theta': 1.0,
    'sigma_Z': 1.0
}

# Latent dimensions to test: 5, 10, 25, 50, 100 -> Done

# Feature counts to test: 5, 10, 25, 50, 100, 500 -> Done

# Sample counts to test: 5, 10, 25, 50, 100, 500 -> Done

# Sigma_theta values to test: 0, 0.1, 1, 10 -> Done

# Sigma_Z values to test: 0.1, 1, 10 -> Done

def evaluate(target_matrix, learned_matrix, print_error=False):
    sse = np.linalg.norm(target_matrix - learned_matrix, 'fro') ** 2
    rmse = root_mean_squared_error(target_matrix.flatten(), learned_matrix.flatten())
    if print_error:
        print(f'SSE of Covariance Values: {sse}')
        print(f'RMSE of Covariance Values: {rmse}')
    return sse, rmse

results_df = pd.read_csv('./covariance_data/covariance_sim_results.csv')

for _ in range(10):
    # Create target model and generate data
    target_model = CovarianceModel(params['n_features'], params['embedding_dim'], sigma_theta=params['sigma_theta'], sigma_Z=params['sigma_Z'])
    target_cov = target_model.get_cov().detach().numpy()
    data = target_model.gen_samples(params['n_samples'])

    # Fit and evaluate custom covariance model
    trained_model = CovarianceModel(params['n_features'], params['embedding_dim'], sigma_theta=params['sigma_theta'], sigma_Z=params['sigma_Z'])
    trained_model.optimize(data, 5000, lr=0.01)
    model_cov = trained_model.get_cov().detach().numpy()
    print('New Model Covariance Matrix:')
    model_sse, model_rmse = evaluate(target_cov, model_cov, print_error=True)

    # Fit and evaluate sample covariance matrix
    sample_cov = EmpiricalCovariance().fit(data).covariance_
    print('Sample Covariance Matrix:')
    sample_cov_sse, sample_cov_rmse = evaluate(target_cov, sample_cov, print_error=True)

    # Fit and evaluate Ledoit-Wolf covariance matrix
    lw_shrunk_cov = LedoitWolf().fit(data).covariance_
    print('Ledoit-Wolf Shrunk Covariance Matrix:')
    ledoit_wolf_sse, ledoit_wolf_rmse = evaluate(target_cov, lw_shrunk_cov, print_error=True)

    results_df.loc[len(results_df)] = [params['n_features'], params['n_samples'], params['embedding_dim'], params['sigma_theta'], params['sigma_Z'],
                                       sample_cov_sse, sample_cov_rmse, ledoit_wolf_sse, ledoit_wolf_rmse, model_sse, model_rmse]

results_df = results_df.astype({'n_features': int, 'n_samples': int, 'embedding_dim': int})
results_df.to_csv('./covariance_data/covariance_sim_results.csv', index=False)