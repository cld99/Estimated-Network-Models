# %%
import feature_interaction
from importlib import reload

reload(feature_interaction)
from feature_interaction import FeatureInteractionModel

# %%
# %%
import numpy as np
import torch
from numpy.linalg import norm
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import plotly.express as px
from feature_interaction import FeatureInteractionModel
from feature_interaction_simu import simulate_data

# %%
seed = 42

epochs = 3000

n = 1000
p = 8
d = 2
rho = 0.5

alpha = 1.0
sigma_z = 1.0
sigma_theta = 0.5
sigma_x = 1.0
sigma_y = 1.0
beta0 = 0.5

cov_type = 'ar1'  

n_list = list(range(1000, 10001, 500))
p_list = list(range(2, 10, 1))
rho_list = [0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
d_list = [2]

results = []

for rho in rho_list:
    for d in d_list:
        for p in p_list:
            print(f"\nRunning rho={rho}, p={p}, d={d}")
            np.random.seed(seed)
            torch.manual_seed(seed)
            beta = np.random.normal(0, 1, size=p)
            
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

            # print("X shape:", X_true.shape)
            # print("y shape:", y_true.shape)
            # print("Theta_true shape:", Theta_true.shape)

            # convert to torch
            X = torch.tensor(X_true, dtype=torch.float32)
            y = torch.tensor(y_true, dtype=torch.float32)

            # fit model
            model = FeatureInteractionModel(p=p, d=d, sigma_y=sigma_y, sigma_theta=sigma_theta, sigma_z=sigma_z)
            model.fit(X, y, lr=0.01, epochs=epochs)
            model.eval()
            
            # estimated Theta
            Theta_hat_model = np.zeros((p, p))
            idx = 0
            for j in range(p):
                for k in range(j + 1, p):
                    Theta_hat_model[j, k] = model.theta_upper[idx].item()
                    Theta_hat_model[k, j] = Theta_hat_model[j, k]
                    idx += 1
            np.fill_diagonal(Theta_hat_model, 0.0)

            # theta error
            theta_error = np.linalg.norm(Theta_true - Theta_hat_model, ord="fro")

            # z distance error
            D_true = squareform(pdist(Z_true, metric="euclidean"))
            Z_hat = model.Z.detach().cpu().numpy()
            D_hat = squareform(pdist(Z_hat, metric="euclidean"))
            z_dist_error = np.linalg.norm(D_true - D_hat, ord="fro")  # Frobenius norm

            print("theta error =", theta_error)
            print("z distance error =", z_dist_error)

            results.append({
                "d": d,
                "p": p,
                "rho":rho,
                "theta_error": theta_error,
                "z_dist_error": z_dist_error
            })
            

df_results = pd.DataFrame(results)
print(df_results)

# %%
df_results = pd.DataFrame(results)
print(df_results)

# %%
fixed_d = 2

df_d = df_results[df_results["d"] == fixed_d]


info_text = (
    f" n={n}, d={fixed_d}, "
    f" sigma_z={sigma_z}, sigma_theta={sigma_theta}, "
    f"sigma_x={sigma_x}, sigma_y={sigma_y}, cov_type={cov_type}"
)

z_table = df_d.pivot(index="rho", columns="p", values="z_dist_error")
z_table = z_table.reindex(index=rho_list, columns=p_list)

fig2 = px.imshow(
    z_table,
    text_auto=True,
    aspect="auto",
    title=f"Z-Pairwise Frobenius Norm Error Heatmap<br><sup>{info_text}</sup>",
    labels={"x": "p", "y": "rho", "color": "z distance error"}
)

fig2.update_xaxes(
    tickmode="array",
    tickvals=list(z_table.columns),
    ticktext=[str(x) for x in z_table.columns],
    side="bottom"
)

fig2.update_yaxes(
    tickmode="array",
    tickvals=list(z_table.index),
    ticktext=[str(y) for y in z_table.index]
)
fig2.show()