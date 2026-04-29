import numpy as np
import pandas as pd
import torch

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.cluster import SpectralClustering

from feature_interaction import FeatureInteractionModel


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


def get_group_labels_from_theta(Theta, K=2):
    A = np.abs(Theta).copy()
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(A, 0.0)

    if np.allclose(A, 0):
        return np.arange(A.shape[0]) % K

    clustering = SpectralClustering(
        n_clusters=K,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=0
    )

    return clustering.fit_predict(A)


def evaluate_model(model, X_test_t, y_test_np, y_scaler):
    model.eval()

    with torch.no_grad():
        y_pred_std = model.predict(X_test_t).squeeze().detach().cpu().numpy()

    y_pred = y_scaler.inverse_transform(
        y_pred_std.reshape(-1, 1)
    ).ravel()

    mse = mean_squared_error(y_test_np, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_np, y_pred)
    mae = np.abs(y_test_np - y_pred).mean()

    return mse, rmse, r2, mae


def add_result(results, run, seed, K, d, model_name, mse, rmse, r2, mae):
    results.append({
        "run": run,
        "seed": seed,
        "K": K,
        "d": d,
        "model": model_name,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAE": mae
    })


# settings
base_seed = 42
n_runs = 5
lr = 0.0005
epochs = 4000

d = 2
K_list = [2, 3, 4, 5, 6]


results = []


# load real data
dataset = fetch_openml(
    data_id=44959,  # Cancer_Drug_Response
    as_frame="auto",
    parser="auto"
)

X = dataset.data.astype(np.float32)
y = dataset.target.astype(np.float32)


# run experiment
for run in range(n_runs):
    seed = base_seed + run
    print(f"\nRun {run + 1}/{n_runs}, seed={seed}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed
    )

    # standardize X
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train_std, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_std, dtype=torch.float32)

    # standardize y for neural model
    y_train_np = y_train.to_numpy(dtype=np.float32)
    y_test_np = y_test.to_numpy(dtype=np.float32)

    y_scaler = StandardScaler()
    y_train_std = y_scaler.fit_transform(
        y_train_np.reshape(-1, 1)
    ).ravel()

    y_train_t = torch.tensor(y_train_std, dtype=torch.float32)

    n, p = X_train_t.shape

    # linear regression baseline
    print("  Running linear regression")

    linreg = LinearRegression()
    linreg.fit(X_train_std, y_train_np)

    y_pred = linreg.predict(X_test_std)

    mse = mean_squared_error(y_test_np, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_np, y_pred)
    mae = np.abs(y_test_np - y_pred).mean()

    print(f"    MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

    add_result(
        results=results,
        run=run,
        seed=seed,
        K="NA",
        d="NA",
        model_name="linear regression",
        mse=mse,
        rmse=rmse,
        r2=r2,
        mae=mae
    )

    # without latent model
    print("  Running without latent")

    torch.manual_seed(seed + 10000)

    model_without = FeatureInteractionModel(
        p=p,
        d=d,
        use_latent_theta=False,
        use_sbm=False
    )

    model_without.fit(
        X_train_t,
        y_train_t,
        epochs=epochs,
        lr=lr
    )

    mse, rmse, r2, mae = evaluate_model(
        model_without,
        X_test_t,
        y_test_np,
        y_scaler
    )

    print(f"    MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

    add_result(
        results=results,
        run=run,
        seed=seed,
        K="NA",
        d=d,
        model_name="without latent",
        mse=mse,
        rmse=rmse,
        r2=r2,
        mae=mae
    )

    # # use this theta to get SBM group labels
    # Theta_hat = get_theta_hat(model_with, p)

    # with latent model
    print("  Running with latent")

    torch.manual_seed(seed + 20000)

    model_with = FeatureInteractionModel(
        p=p,
        d=d,
        use_latent_theta=True,
        use_sbm=False
    )

    model_with.fit(
        X_train_t,
        y_train_t,
        epochs=epochs,
        lr=lr
    )

    mse, rmse, r2, mae = evaluate_model(
        model_with,
        X_test_t,
        y_test_np,
        y_scaler
    )

    print(f"    MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

    add_result(
        results=results,
        run=run,
        seed=seed,
        K="NA",
        d=d,
        model_name="with latent",
        mse=mse,
        rmse=rmse,
        r2=r2,
        mae=mae
    )
    
    # use this theta to get SBM group labels
    Theta_hat = get_theta_hat(model_with, p)
    # SBM models with multiple K
    
    for K in K_list:
        print(f"  Running with latent + SBM, K={K}")

        group_labels_hat = get_group_labels_from_theta(
            Theta_hat,
            K=K
        )

        torch.manual_seed(seed + 30000 + K)

        model_sbm = FeatureInteractionModel(
            p=p,
            d=d,
            use_latent_theta=True,
            use_sbm=True,
            group_labels=group_labels_hat,
            K=K
        )

        model_sbm.fit(
            X_train_t,
            y_train_t,
            epochs=epochs,
            lr=lr
        )

        mse, rmse, r2, mae = evaluate_model(
            model_sbm,
            X_test_t,
            y_test_np,
            y_scaler
        )

        print(f"    K={K}, MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

        add_result(
            results=results,
            run=run,
            seed=seed,
            K=K,
            d=d,
            model_name="with latent + SBM",
            mse=mse,
            rmse=rmse,
            r2=r2,
            mae=mae
        )


# save results
df_results = pd.DataFrame(results)

df_summary = (
    df_results
    .groupby(["K", "model"], as_index=False)
    .agg(
        mean_MSE=("MSE", "mean"),
        std_MSE=("MSE", "std"),
        mean_RMSE=("RMSE", "mean"),
        std_RMSE=("RMSE", "std"),
        mean_R2=("R2", "mean"),
        std_R2=("R2", "std"),
        mean_MAE=("MAE", "mean"),
        std_MAE=("MAE", "std")
    )
)
