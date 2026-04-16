import numpy as np
import torch
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from covariance_model import CovarianceModel

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Set datatype to float64 to reduce errors due to precision
torch.set_default_dtype(torch.float64)

params = {
    "n_features": 500,
    "n_samples": 50,
    "embedding_dim": 100,
    "df": 1000,
    "sigma_Z": 1
}

def evaluate(target_matrix, learned_matrix):
    frobenius_dist = np.linalg.norm(target_matrix - learned_matrix, 'fro')
    rmse = np.sqrt(np.mean((target_matrix.flatten() - learned_matrix.flatten()) ** 2))
    print(f"Frobenius Distance to True Cov. Matrix: {frobenius_dist}")
    print(f"RMSE of Covariance Values: {rmse}")

# Create target model and generate data
target_model = CovarianceModel(params["n_features"], params["embedding_dim"], params["df"], params["sigma_Z"])
target_cov = target_model.get_cov().numpy()
data = target_model.gen_samples(params["n_samples"])

# Fit and evaluate sample covariance matrix
sample_cov = EmpiricalCovariance().fit(data)
print("Sample Covariance Matrix:")
evaluate(target_cov, sample_cov)

# Fit and evaluate Ledoit-Wolf covariance matrix
lw_shrunk_cov = LedoitWolf().fit(data)
print("Ledoit-Wolf Shrunk Covariance Matrix:")
evaluate(target_cov, lw_shrunk_cov)

# Fit and evaluate custom covariance model
model_cov = CovarianceModel(params["n_features"], params["embedding_dim"], params["df"], params["sigma_Z"])
print("New Model Covariance Matrix:")
evaluate(target_cov, model_cov)