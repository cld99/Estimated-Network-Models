# %%
import feature_interaction
from importlib import reload

reload(feature_interaction)
from feature_interaction import FeatureInteractionModel

# %%
import numpy as np
import torch
import pandas as pd

from feature_interaction import FeatureInteractionModel
from feature_interaction_simu import simulate_data


# %%
from sklearn.cluster import SpectralClustering

def get_group_labels_from_theta(Theta, K):
    A = np.abs(Theta).copy()
    np.fill_diagonal(A, 0)
    clustering = SpectralClustering(
        n_clusters=K,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=0
    )
    return clustering.fit_predict(A)

# %%

# settings
base_seed = 100
epochs = 2000
lr = 0.01
n_rep = 10

# fixed parameters
n = 2000
p = 20
d = 2
rho = 0.5

alpha = 1.0
sigma_z = 0.2
sigma_theta = 0.2
sigma_x = 1.0
sigma_y = 10.0
sigma_beta = 10.0
beta0 = 0.5
cov_type = "ar1"


K = 2


def get_theta_hat(model, p):
    Theta_hat = np.zeros((p, p))

    idx = 0
    for j in range(p):
        for k in range(j + 1, p):
            Theta_hat[j, k] = model.theta_upper[idx].item()
            Theta_hat[k, j] = Theta_hat[j, k]
            idx += 1

    np.fill_diagonal(Theta_hat, 0.0)
    return Theta_hat


def generate_one_dataset(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    beta = np.random.normal(0, sigma_beta, size=p)

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

    X = torch.tensor(X_true, dtype=torch.float32)
    y = torch.tensor(y_true, dtype=torch.float32)

    return X, y, Theta_true


def fit_one_model(X, y, use_latent, fit_seed, use_sbm, group_labels=None,K=None):
    torch.manual_seed(fit_seed)
    np.random.seed(fit_seed)

    model = FeatureInteractionModel(
        p=p,
        d=d,
        sigma_y=sigma_y,
        sigma_theta=sigma_theta,
        sigma_z=sigma_z,
        use_latent_theta=use_latent,
        use_sbm = use_sbm,
        group_labels=group_labels,
        K = K
    )

    model.fit(X, y, lr=lr, epochs=epochs)
    model.eval()

    return model


def run_one_replicate(rep,K):
    replicate_seed = base_seed + rep

    X, y, Theta_true = generate_one_dataset(seed=replicate_seed)

    model_with = fit_one_model(
        X=X,
        y=y,
        use_latent=True,
        use_sbm=False,
        fit_seed=replicate_seed + 10000
    )

    model_without = fit_one_model(
        X=X,
        y=y,
        use_latent=False,
        use_sbm=False,
        fit_seed=replicate_seed + 20000
    )
    
    Theta_hat_with = get_theta_hat(model_with, p)
    Theta_hat_without = get_theta_hat(model_without, p)
    
    group_labels_from_latent = get_group_labels_from_theta(Theta_hat_with, K=K)
    
    model_sbm = fit_one_model(
        X=X,
        y=y,
        use_latent=True,
        use_sbm=True,
        fit_seed=replicate_seed + 30000,
        group_labels=group_labels_from_latent,
        K = K
    )

    Theta_hat_sbm = get_theta_hat(model_sbm, p)

    err_with = np.linalg.norm(Theta_true - Theta_hat_with, ord="fro")
    err_without = np.linalg.norm(Theta_true - Theta_hat_without, ord="fro")
    err_sbm = np.linalg.norm(Theta_true - Theta_hat_sbm, ord="fro")

    return err_with, err_without, err_sbm


# run experiment: compare different K  compare no/latent/SBM
K_list = [2, 3, 4, 5]

results = []

for K in K_list:
    print(f"\n===== Running K = {K} =====")

    for rep in range(n_rep):
        err_with, err_without, err_sbm = run_one_replicate(rep, K=K)

        common_info = {
            "rep": rep,
            "K": K,
            "n": n,
            "p": p,
            "d": d,
            "rho": rho,
            "sigma_z": sigma_z,
            "sigma_theta": sigma_theta,
            "sigma_y": sigma_y,
            "sigma_beta": sigma_beta
        }

        results.append({
            **common_info,
            "model": "with latent",
            "theta_error": err_with
        })

        results.append({
            **common_info,
            "model": "without latent",
            "theta_error": err_without
        })

        results.append({
            **common_info,
            "model": "with latent + SBM",
            "theta_error": err_sbm
        })

        print(
            f"K={K} | rep={rep:02d} | "
            f"with latent={err_with:.4f} | "
            f"without latent={err_without:.4f} | "
            f"with latent + SBM={err_sbm:.4f}"
        )

# save results
df_results = pd.DataFrame(results)

df_summary = (
    df_results
    .groupby(["K", "model"])["theta_error"]
    .agg(["mean", "std"])
    .reset_index()
)

print(df_summary)

df_results.to_csv("theta_results_vary_K.csv", index=False)
df_summary.to_csv("theta_summary_vary_K.csv", index=False)

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

for model_name in df_summary["model"].unique():
    df_plot = df_summary[df_summary["model"] == model_name].sort_values("K")

    plt.errorbar(
        df_plot["K"],
        df_plot["mean"],
        yerr=df_plot["std"],
        marker="o",
        capsize=4,
        label=model_name
    )

plt.xlabel("K")
plt.ylabel("Theta Frobenius Error")
plt.title("Theta Error vs K")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


