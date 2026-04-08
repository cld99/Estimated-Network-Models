# %%
import numpy as np
import torch
from numpy.linalg import norm

from feature_interaction import FeatureInteractionModel
from feature_interaction_simu import simulate_data

# %%
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


n = 10000
p = 8
d = 2

alpha = 1.0
sigma_z = 1.0
sigma_theta = 0.5
sigma_x = 1.0
sigma_y = 1.0

beta0 = 0.5
beta = np.random.normal(0, 1, size=p)

cov_type = 'ar1'  
rho = 0.5


# simulate data 
Z_true, Theta_true, X_true, y_true = simulate_data(
    n=n,
    p=p,
    d=d,
    alpha=alpha,
    sigma_z=sigma_z,
    sigma_theta=sigma_theta,
    sigma_x=sigma_x,
    sigma_y=sigma_y,
    beta0=beta0,
    beta=beta,
    cov_type=cov_type,
    rho=rho
)

print("X shape:", X_true.shape)
print("y shape:", y_true.shape)
print("Theta_true shape:", Theta_true.shape)


# %%
# convert to torch
X = torch.tensor(X_true, dtype=torch.float32)
y = torch.tensor(y_true, dtype=torch.float32)


model = FeatureInteractionModel(p=p, d=d)
model.fit(X, y, lr=0.01, epochs=3000)
model.eval()

# %%
print("n =", X_true.shape[0])

print("\nTrue beta0:", beta0)
print("Estimated beta0:", model.beta0.item())

print("\nTrue beta:", beta)
print("Estimated beta:", model.beta.detach().cpu().numpy())

print("\nTrue alpha:", alpha)
print("Estimated alpha:", model.alpha.item())

print("\nTrue sigma_y:", sigma_y)
print("Estimated sigma_y:", np.exp(0.5 * model.log_sigma_y2.item()))

print("\nTrue sigma_theta:", sigma_theta)
print("Estimated sigma_theta:", np.exp(0.5 * model.log_sigma_theta2.item()))

print("\nTrue sigma_z:", sigma_z)
print("Estimated sigma_z:", np.exp(0.5 * model.log_sigma_z2.item()))

Theta_hat_model = np.zeros((p, p))
idx = 0
for j in range(p):
    for k in range(j + 1, p):
        Theta_hat_model[j, k] = model.theta_upper[idx].item()
        Theta_hat_model[k, j] = Theta_hat_model[j, k]
        idx += 1

np.fill_diagonal(Theta_hat_model, 0.0)

Theta_hat_cov = np.cov(X_true, rowvar=False)
np.fill_diagonal(Theta_hat_cov, 0.0)

error_model = np.sum((Theta_true - Theta_hat_model) ** 2)
error_cov = np.sum((Theta_true - Theta_hat_cov) ** 2)

print("Squared Frobenius norm error (model) =", error_model)
print("Squared Frobenius norm error (sample covariance of X) =", error_cov)


