
import feature_interaction
from importlib import reload

reload(feature_interaction)
from feature_interaction import FeatureInteractionModel

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from feature_interaction import FeatureInteractionModel
from feature_interaction_simu import simulate_data

# %%
seed = 100
epochs = 500
lr = 0.01

# fixed parameters
alpha = 1.0
sigma_z_default = 0.2
sigma_theta = 0.2
sigma_x = 1.0
sigma_y = 10.0
sigma_beta_default = 10.0
beta0 = 0.5
cov_type = "ar1"

# %%
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


def run_one_setting(n, p, d, rho, use_latent, sigma_z, sigma_beta):
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

    torch.manual_seed(seed)
    model = FeatureInteractionModel(
        p=p,
        d=d,
        sigma_y=sigma_y,
        sigma_theta=sigma_theta,
        sigma_z=sigma_z,
        use_latent_theta=use_latent
    )

    model.fit(X, y, lr=lr, epochs=epochs)
    model.eval()

    Theta_hat = get_theta_hat(model, p)
    theta_error = np.linalg.norm(Theta_true - Theta_hat, ord="fro")

    return theta_error


# %%
results = []

# 1) vary n
n_list = list(range(500, 5000, 500))
p_fixed = 8
d_fixed = 2
rho_fixed = 0.5
sigma_z_fixed = sigma_z_default
sigma_beta_fixed = sigma_beta_default

for n in n_list:
    print(f"Running vary n: n={n}")

    err_with = run_one_setting(
        n=n, p=p_fixed, d=d_fixed, rho=rho_fixed,
        use_latent=True,
        sigma_z=sigma_z_fixed,
        sigma_beta=sigma_beta_fixed
    )
    err_without = run_one_setting(
        n=n, p=p_fixed, d=d_fixed, rho=rho_fixed,
        use_latent=False,
        sigma_z=sigma_z_fixed,
        sigma_beta=sigma_beta_fixed
    )

    results.append({
        "vary": "n",
        "x_value": n,
        "n": n,
        "p": p_fixed,
        "d": d_fixed,
        "rho": rho_fixed,
        "sigma_z": sigma_z_fixed,
        "sigma_beta": sigma_beta_fixed,
        "model": "with latent",
        "theta_error": err_with
    })
    results.append({
        "vary": "n",
        "x_value": n,
        "n": n,
        "p": p_fixed,
        "d": d_fixed,
        "rho": rho_fixed,
        "sigma_z": sigma_z_fixed,
        "sigma_beta": sigma_beta_fixed,
        "model": "without latent",
        "theta_error": err_without
    })

    print(f"  with latent    : {err_with:.4f}")
    print(f"  without latent : {err_without:.4f}")


# 2) vary p
p_list = list(range(2, 20, 1))
n_fixed = 500
d_fixed = 2
rho_fixed = 0.5
sigma_z_fixed = sigma_z_default
sigma_beta_fixed = sigma_beta_default

for p in p_list:
    print(f"Running vary p: p={p}")

    err_with = run_one_setting(
        n=n_fixed, p=p, d=d_fixed, rho=rho_fixed,
        use_latent=True,
        sigma_z=sigma_z_fixed,
        sigma_beta=sigma_beta_fixed
    )
    err_without = run_one_setting(
        n=n_fixed, p=p, d=d_fixed, rho=rho_fixed,
        use_latent=False,
        sigma_z=sigma_z_fixed,
        sigma_beta=sigma_beta_fixed
    )

    results.append({
        "vary": "p",
        "x_value": p,
        "n": n_fixed,
        "p": p,
        "d": d_fixed,
        "rho": rho_fixed,
        "sigma_z": sigma_z_fixed,
        "sigma_beta": sigma_beta_fixed,
        "model": "with latent",
        "theta_error": err_with
    })
    results.append({
        "vary": "p",
        "x_value": p,
        "n": n_fixed,
        "p": p,
        "d": d_fixed,
        "rho": rho_fixed,
        "sigma_z": sigma_z_fixed,
        "sigma_beta": sigma_beta_fixed,
        "model": "without latent",
        "theta_error": err_without
    })

    print(f"  with latent    : {err_with:.4f}")
    print(f"  without latent : {err_without:.4f}")


# 3) vary rho
rho_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
n_fixed = 500
p_fixed = 8
d_fixed = 2
sigma_z_fixed = sigma_z_default
sigma_beta_fixed = sigma_beta_default

for rho in rho_list:
    print(f"Running vary rho: rho={rho}")

    err_with = run_one_setting(
        n=n_fixed, p=p_fixed, d=d_fixed, rho=rho,
        use_latent=True,
        sigma_z=sigma_z_fixed,
        sigma_beta=sigma_beta_fixed
    )
    err_without = run_one_setting(
        n=n_fixed, p=p_fixed, d=d_fixed, rho=rho,
        use_latent=False,
        sigma_z=sigma_z_fixed,
        sigma_beta=sigma_beta_fixed
    )

    results.append({
        "vary": "rho",
        "x_value": rho,
        "n": n_fixed,
        "p": p_fixed,
        "d": d_fixed,
        "rho": rho,
        "sigma_z": sigma_z_fixed,
        "sigma_beta": sigma_beta_fixed,
        "model": "with latent",
        "theta_error": err_with
    })
    results.append({
        "vary": "rho",
        "x_value": rho,
        "n": n_fixed,
        "p": p_fixed,
        "d": d_fixed,
        "rho": rho,
        "sigma_z": sigma_z_fixed,
        "sigma_beta": sigma_beta_fixed,
        "model": "without latent",
        "theta_error": err_without
    })

    print(f"  with latent    : {err_with:.4f}")
    print(f"  without latent : {err_without:.4f}")


# 4) vary d
d_list = list(range(2, 20, 1))
n_fixed = 500
p_fixed = 8
rho_fixed = 0.5
sigma_z_fixed = sigma_z_default
sigma_beta_fixed = sigma_beta_default

for d in d_list:
    print(f"Running vary d: d={d}")

    err_with = run_one_setting(
        n=n_fixed, p=p_fixed, d=d, rho=rho_fixed,
        use_latent=True,
        sigma_z=sigma_z_fixed,
        sigma_beta=sigma_beta_fixed
    )
    err_without = run_one_setting(
        n=n_fixed, p=p_fixed, d=d, rho=rho_fixed,
        use_latent=False,
        sigma_z=sigma_z_fixed,
        sigma_beta=sigma_beta_fixed
    )

    results.append({
        "vary": "d",
        "x_value": d,
        "n": n_fixed,
        "p": p_fixed,
        "d": d,
        "rho": rho_fixed,
        "sigma_z": sigma_z_fixed,
        "sigma_beta": sigma_beta_fixed,
        "model": "with latent",
        "theta_error": err_with
    })
    results.append({
        "vary": "d",
        "x_value": d,
        "n": n_fixed,
        "p": p_fixed,
        "d": d,
        "rho": rho_fixed,
        "sigma_z": sigma_z_fixed,
        "sigma_beta": sigma_beta_fixed,
        "model": "without latent",
        "theta_error": err_without
    })

    print(f"  with latent    : {err_with:.4f}")
    print(f"  without latent : {err_without:.4f}")


# 5) vary sigma_z
sigma_z_list = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
n_fixed = 500
p_fixed = 8
d_fixed = 2
rho_fixed = 0.5
sigma_beta_fixed = sigma_beta_default

for sigma_z in sigma_z_list:
    print(f"Running vary sigma_z: sigma_z={sigma_z}")

    err_with = run_one_setting(
        n=n_fixed, p=p_fixed, d=d_fixed, rho=rho_fixed,
        use_latent=True,
        sigma_z=sigma_z,
        sigma_beta=sigma_beta_fixed
    )
    err_without = run_one_setting(
        n=n_fixed, p=p_fixed, d=d_fixed, rho=rho_fixed,
        use_latent=False,
        sigma_z=sigma_z,
        sigma_beta=sigma_beta_fixed
    )

    results.append({
        "vary": "sigma_z",
        "x_value": sigma_z,
        "n": n_fixed,
        "p": p_fixed,
        "d": d_fixed,
        "rho": rho_fixed,
        "sigma_z": sigma_z,
        "sigma_beta": sigma_beta_fixed,
        "model": "with latent",
        "theta_error": err_with
    })
    results.append({
        "vary": "sigma_z",
        "x_value": sigma_z,
        "n": n_fixed,
        "p": p_fixed,
        "d": d_fixed,
        "rho": rho_fixed,
        "sigma_z": sigma_z,
        "sigma_beta": sigma_beta_fixed,
        "model": "without latent",
        "theta_error": err_without
    })

    print(f"  with latent    : {err_with:.4f}")
    print(f"  without latent : {err_without:.4f}")


# 6) vary sigma_beta
sigma_beta_list = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
n_fixed = 500
p_fixed = 8
d_fixed = 2
rho_fixed = 0.5
sigma_z_fixed = sigma_z_default

for sigma_beta in sigma_beta_list:
    print(f"Running vary sigma_beta: sigma_beta={sigma_beta}")

    err_with = run_one_setting(
        n=n_fixed, p=p_fixed, d=d_fixed, rho=rho_fixed,
        use_latent=True,
        sigma_z=sigma_z_fixed,
        sigma_beta=sigma_beta
    )
    err_without = run_one_setting(
        n=n_fixed, p=p_fixed, d=d_fixed, rho=rho_fixed,
        use_latent=False,
        sigma_z=sigma_z_fixed,
        sigma_beta=sigma_beta
    )

    results.append({
        "vary": "sigma_beta",
        "x_value": sigma_beta,
        "n": n_fixed,
        "p": p_fixed,
        "d": d_fixed,
        "rho": rho_fixed,
        "sigma_z": sigma_z_fixed,
        "sigma_beta": sigma_beta,
        "model": "with latent",
        "theta_error": err_with
    })
    results.append({
        "vary": "sigma_beta",
        "x_value": sigma_beta,
        "n": n_fixed,
        "p": p_fixed,
        "d": d_fixed,
        "rho": rho_fixed,
        "sigma_z": sigma_z_fixed,
        "sigma_beta": sigma_beta,
        "model": "without latent",
        "theta_error": err_without
    })

    print(f"  with latent    : {err_with:.4f}")
    print(f"  without latent : {err_without:.4f}")


# %%
df_results = pd.DataFrame(results)
print(df_results)

# %%
# %%
for vary_name in ["n", "p", "rho", "d", "sigma_z", "sigma_beta"]:
    df_plot = df_results[df_results["vary"] == vary_name]

    df_with = df_plot[df_plot["model"] == "with latent"].sort_values("x_value")
    df_without = df_plot[df_plot["model"] == "without latent"].sort_values("x_value")

    row0 = df_plot.iloc[0]

    info_text = (
        f"fixed: n={row0['n']}, p={row0['p']}, d={row0['d']}, rho={row0['rho']}, "
        f"sigma_z={row0['sigma_z']}, sigma_beta={row0['sigma_beta']}\n"
        f"alpha={alpha}, sigma_theta={sigma_theta}, sigma_x={sigma_x}, sigma_y={sigma_y}, "
        f"beta0={beta0}, cov_type={cov_type}, epochs={epochs}, lr={lr}, seed={seed}"
    )

    plt.figure(figsize=(9, 6))
    plt.plot(df_with["x_value"], df_with["theta_error"], marker="o", label="with latent")
    plt.plot(df_without["x_value"], df_without["theta_error"], marker="o", label="without latent")

    plt.xlabel(vary_name)
    plt.ylabel("theta error")
    plt.title(f"Theta Error vs {vary_name}\n{info_text}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


