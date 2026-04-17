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
sigma_beta = 1.0
beta0 = 0.5

cov_type = 'ar1'  

n_list = list(range(1000, 10001, 500))
p_list = list(range(2, 10, 1))
rho_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
d_list = [2]

with_latent_results = []

for rho in rho_list:
    for d in d_list:
        for p in p_list:
            print(f"\nRunning rho={rho}, p={p}, d={d}")
            np.random.seed(seed)
            torch.manual_seed(seed)
            beta = np.random.normal(0, sigma_beta, size=p)  
            
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
            model = FeatureInteractionModel(p=p, d=d, sigma_y=sigma_y, sigma_theta=sigma_theta, sigma_z=sigma_z, use_latent_theta=True)
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

            with_latent_results.append({
                "d": d,
                "p": p,
                "rho":rho,
                "theta_error": theta_error,
                "z_dist_error": z_dist_error
            })
            

df_with_latent_results = pd.DataFrame(with_latent_results)
print(df_with_latent_results)

# %%
# Z-distance recovery plot

fixed_d = 2

df_d = df_with_latent_results[df_with_latent_results["d"] == fixed_d]

info_text = (
    f"use_latent_theta=True"
    f"<br>n={n}, d={fixed_d}, "
    f" sigma_z={sigma_z}, sigma_theta={sigma_theta}, "
    f"sigma_x={sigma_x}, sigma_y={sigma_y}, cov_type={cov_type}"
)

z_table = df_d.pivot(index="rho", columns="p", values="z_dist_error")
z_table = z_table.reindex(index=rho_list, columns=p_list)

fig2 = px.imshow(
    z_table,
    text_auto=True,
    aspect="auto",
    title=rf"Z-Pairwise Frobenius Norm Error Heatmap"
          rf"||Z_pairwise_distance_true - Z_pairwise_distance_hat||_F"
          rf"<br><sup>{info_text}</sup>",
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

# %%
# Theta recovery plot

fixed_d = 2

df_d = df_with_latent_results[df_with_latent_results["d"] == fixed_d]

info_text = (
    f"use_latent_theta=True"
    f"<br>n={n}, d={fixed_d}, "
    f" sigma_z={sigma_z}, sigma_theta={sigma_theta}, "
    f"sigma_x={sigma_x}, sigma_y={sigma_y}, cov_type={cov_type}"
)

theta_table = df_d.pivot(index="rho", columns="p", values="theta_error")
theta_table = theta_table.reindex(index=rho_list, columns=p_list)

fig1 = px.imshow(
    theta_table,
    text_auto=True,
    aspect="auto",
    title=rf"Theta Frobenius Norm Error Heatmap: "
          rf"||Theta_true - Theta_hat||_F"
          rf"<br><sup>{info_text}</sup>",
    labels={"x": "p", "y": "rho", "color": "theta error"}
)

fig1.update_xaxes(
    tickmode="array",
    tickvals=list(theta_table.columns),
    ticktext=[str(x) for x in theta_table.columns],
    side="bottom"
)

fig1.update_yaxes(
    tickmode="array",
    tickvals=list(theta_table.index),
    ticktext=[str(y) for y in theta_table.index]
)

fig1.show()

# %%
fixed_d = 2
fixed_rho = 0.5

df_line = df_with_latent_results[
    (df_with_latent_results["d"] == fixed_d) &
    (df_with_latent_results["rho"] == fixed_rho)
].copy()

df_line = df_line.sort_values("p")

info_text = (
    f"use_latent_theta=True"
    f"<br>n={n}, rho={fixed_rho}, d={fixed_d}, "
    f"sigma_z={sigma_z}, sigma_theta={sigma_theta}, "
    f"sigma_x={sigma_x}, sigma_y={sigma_y}, cov_type={cov_type}"
)

fig = px.line(
    df_line,
    x="p",
    y=["theta_error", "z_dist_error"],
    markers=True,
    title=rf"Theta Error and Z-Distance Error vs p"
          rf"<br><sup>{info_text}</sup>",
    labels={
        "value": "Error",
        "p": "p",
        "variable": "Metric"
    }
)

for trace in fig.data:
    if trace.name == "theta_error":
        trace.line.dash = "solid"
    elif trace.name == "z_dist_error":
        trace.line.dash = "dash"

fig.update_layout(
    legend_title_text="Metric"
)

fig.show()

# %%
# %% baseline

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
sigma_beta = 1.0
beta0 = 0.5

cov_type = 'ar1'  

n_list = list(range(1000, 10001, 500))
p_list = list(range(2, 10, 1))
rho_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
d_list = [2]

without_latent_results = []

for rho in rho_list:
    for d in d_list:
        for p in p_list:
            print(f"\nRunning rho={rho}, p={p}, d={d}")
            np.random.seed(seed)
            torch.manual_seed(seed)
            beta = np.random.normal(0, sigma_beta, size=p)  
            
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
            model = FeatureInteractionModel(p=p, d=d, sigma_y=sigma_y, sigma_theta=sigma_theta, sigma_z=sigma_z, use_latent_theta=False)
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
            # theta_error = np.linalg.norm(Theta_true - Theta_hat_model, ord="fro") / np.linalg.norm(Theta_true, ord="fro")
            theta_error = np.linalg.norm(Theta_true - Theta_hat_model, ord="fro") 

            # z distance error
            D_true = squareform(pdist(Z_true, metric="euclidean"))
            if model.use_latent_theta:
                Z_hat = model.Z.detach().cpu().numpy()
                D_hat = squareform(pdist(Z_hat, metric="euclidean"))
                z_dist_error = np.linalg.norm(D_true - D_hat, ord="fro")
            else:
                z_dist_error = np.nan
            
            print("theta error =", theta_error)
            print("z distance error =", z_dist_error)

            without_latent_results.append({
                "d": d,
                "p": p,
                "rho":rho,
                "theta_error": theta_error,
                "z_dist_error": z_dist_error
            })
            

df_without_latent_results = pd.DataFrame(without_latent_results)
print(df_without_latent_results)

# %%
# Theta recovery plot

fixed_d = 2

df_d = df_without_latent_results[df_without_latent_results["d"] == fixed_d]

info_text = (
    f"use_latent_theta=False,"
    f"<br>n={n}, d={fixed_d}, "
    f" sigma_z={sigma_z}, sigma_theta={sigma_theta}, "
    f"sigma_x={sigma_x}, sigma_y={sigma_y}, cov_type={cov_type}"
)

theta_table = df_d.pivot(index="rho", columns="p", values="theta_error")
theta_table = theta_table.reindex(index=rho_list, columns=p_list)

fig1 = px.imshow(
    theta_table,
    text_auto=True,
    aspect="auto",
    title=rf"Theta Frobenius Norm Error Heatmap: "
          rf"||Theta_true - Theta_hat||_F"
          rf"<br><sup>{info_text}</sup>",
    labels={"x": "p", "y": "rho", "color": "theta error"}
)

fig1.update_xaxes(
    tickmode="array",
    tickvals=list(theta_table.columns),
    ticktext=[str(x) for x in theta_table.columns],
    side="bottom"
)

fig1.update_yaxes(
    tickmode="array",
    tickvals=list(theta_table.index),
    ticktext=[str(y) for y in theta_table.index]
)

fig1.show()

# %%
fixed_d = 2

df_no = df_without_latent_results[df_without_latent_results["d"] == fixed_d]
df_yes = df_with_latent_results[df_with_latent_results["d"] == fixed_d]

theta_no = df_no.pivot(index="rho", columns="p", values="theta_error")
theta_yes = df_yes.pivot(index="rho", columns="p", values="theta_error")

theta_no = theta_no.reindex(index=rho_list, columns=p_list)
theta_yes = theta_yes.reindex(index=rho_list, columns=p_list)

theta_diff = theta_no - theta_yes

info_text = (
    f"n={n}, d={fixed_d}, "
    f"sigma_z={sigma_z}, sigma_theta={sigma_theta}, "
    f"sigma_x={sigma_x}, sigma_y={sigma_y}, cov_type={cov_type}"
)

fig_diff = px.imshow(
    theta_diff,
    text_auto=True,
    aspect="auto",
    title=rf"Theta Error Improvement Heatmap"
          rf"<br>without latent - with latent"
          rf"<br><sup>{info_text}</sup>",
    labels={"x": "p", "y": "rho", "color": "error diff"}
)

fig_diff.update_xaxes(
    tickmode="array",
    tickvals=list(theta_diff.columns),
    ticktext=[str(x) for x in theta_diff.columns],
    side="bottom"
)

fig_diff.update_yaxes(
    tickmode="array",
    tickvals=list(theta_diff.index),
    ticktext=[str(y) for y in theta_diff.index]
)

fig_diff.show()


# %%
import plotly.graph_objects as go

fixed_d = 2
fixed_rho = 0.5

df_with_line = df_with_latent_results[
    (df_with_latent_results["d"] == fixed_d) &
    (df_with_latent_results["rho"] == fixed_rho)
].copy().sort_values("p")

df_without_line = df_without_latent_results[
    (df_without_latent_results["d"] == fixed_d) &
    (df_without_latent_results["rho"] == fixed_rho)
].copy().sort_values("p")

info_text = (
    f"n={n}, rho={fixed_rho}, d={fixed_d}, "
    f"sigma_z={sigma_z}, sigma_theta={sigma_theta}, "
    f"sigma_x={sigma_x}, sigma_y={sigma_y}, cov_type={cov_type}"
)

fig_compare = go.Figure()

fig_compare.add_trace(go.Scatter(
    x=df_with_line["p"],
    y=df_with_line["theta_error"],
    mode="lines+markers",
    name="Theta error (with latent theta)",
    line=dict(dash="solid")
))


fig_compare.add_trace(go.Scatter(
    x=df_without_line["p"],
    y=df_without_line["theta_error"],
    mode="lines+markers",
    name="Theta error (without latent theta)",
    line=dict(dash="solid")
))

fig_compare.update_layout(
    title=rf"Theta Errors vs p"
          rf"<br><sup>{info_text}</sup>",
    xaxis_title="p",
    yaxis_title="Error",
    legend_title="Metric"
)

fig_compare.show()

# %%



