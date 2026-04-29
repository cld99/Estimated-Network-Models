import numpy as np
import torch
import pandas as pd
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.metrics import root_mean_squared_error
from scipy.stats import multivariate_normal
from covariance_model import CovarianceModel

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Set datatype to float64 to reduce errors due to precision
torch.set_default_dtype(torch.float64)

# Read data from Cam Nugent S&P 500 dataset (https://www.kaggle.com/datasets/camnugent/sandp500)
raw_data = pd.read_csv('./covariance_data/all_stocks_5yr.csv')
# Pivot so that each stock represents its own feature
prices = raw_data.pivot(index='date', columns='Name', values='close')
# Convert raw stock prices to daily returns
returns = prices.pct_change(fill_method=None)
# Drop oldest row (cannot compute return)
returns = returns.iloc[1:]
# Drop stock tickers with missing data (only 35 of 505)
returns = returns.dropna(axis=1)

params = {
    'n_features': returns.shape[1],
    'embedding_dim': 10,
    'sigma_theta': 10.0,
    'sigma_Z': 1.0,
    'train_split': 0.8
}

def evaluate(data, mean, cov, print_llk=False):
    llk = np.mean(multivariate_normal.logpdf(data, mean=mean, cov=cov, allow_singular=True))
    if print_llk:
        print(llk)
    return llk

results_df = pd.read_csv('./covariance_data/covariance_real_data_results.csv')

train_data = returns.iloc[:int(returns.shape[0] * params['train_split'])]
test_data = returns.iloc[int(returns.shape[0] * params['train_split']):]

# Fit and evaluate custom covariance model
trained_model = CovarianceModel(params['n_features'], params['embedding_dim'], sigma_theta=params['sigma_theta'], sigma_Z=params['sigma_Z'])
trained_model.optimize(torch.tensor(train_data.to_numpy()), 5000, lr=0.01)
model_cov = trained_model.get_cov().detach().numpy()
print('New Model Covariance Matrix:')
model_llk = evaluate(test_data.to_numpy(), test_data.mean().values, model_cov, print_llk=True)

# Fit and evaluate sample covariance matrix
sample_cov = EmpiricalCovariance().fit(train_data).covariance_
print('Sample Covariance Matrix:')
sample_cov_llk = evaluate(test_data.to_numpy(), test_data.mean().values, sample_cov, print_llk=True)

# Fit and evaluate Ledoit-Wolf covariance matrix
lw_shrunk_cov = LedoitWolf().fit(train_data).covariance_
print('Ledoit-Wolf Shrunk Covariance Matrix:')
ledoit_wolf_llk = evaluate(test_data.to_numpy(), test_data.mean().values, lw_shrunk_cov, print_llk=True)

results_df.loc[len(results_df)] = [params['train_split'], sample_cov_llk, ledoit_wolf_llk, model_llk]

results_df.to_csv('./covariance_data/covariance_real_data_results.csv', index=False)