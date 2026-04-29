import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_comparison(x, y1, y2, y3, y1_err, y2_err, y3_err, x_label, y_label, title, path, x_log=False, y_log=False, show=True):
    plt.errorbar(x, y1, yerr=y1_err, marker='.', capsize=4, label='Sample Covariance')
    plt.errorbar(x, y2, yerr=y2_err, marker='.', capsize=4, label='Ledoit-Wolf')
    plt.errorbar(x, y3, yerr=y3_err, marker='.', capsize=4, label='Latent Position Model')
    if x_log:
        plt.xscale('log')
    if y_log:
        plt.yscale('log')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig(path)
    if show:
        plt.show()

sim_df = pd.read_csv('./covariance_data/covariance_sim_results.csv')

# Aggregate results
agg_sim_df = sim_df.groupby(['n_features', 'n_samples', 'embedding_dim', 'sigma_theta', 'sigma_Z']).agg(
    sample_cov_SSE_mean=('sample_cov_SSE', 'mean'),
    sample_cov_SSE_std=('sample_cov_SSE', 'std'),
    sample_cov_RMSE_mean=('sample_cov_RMSE', 'mean'),
    sample_cov_RMSE_std=('sample_cov_RMSE', 'std'),
    ledoit_wolf_SSE_mean=('ledoit_wolf_SSE', 'mean'),
    ledoit_wolf_SSE_std=('ledoit_wolf_SSE', 'std'),
    ledoit_wolf_RMSE_mean=('ledoit_wolf_RMSE', 'mean'),
    ledoit_wolf_RMSE_std=('ledoit_wolf_RMSE', 'std'),
    latent_model_SSE_mean=('latent_model_SSE', 'mean'),
    latent_model_SSE_std=('latent_model_SSE', 'std'),
    latent_model_RMSE_mean=('latent_model_RMSE', 'mean'),
    latent_model_RMSE_std=('latent_model_RMSE', 'std'),
).reset_index()

# Plot 1: Embedding Dimension
filtered_data = agg_sim_df.loc[(agg_sim_df['n_features'] == 100) & (agg_sim_df['n_samples'] == 50) &
                               (agg_sim_df['sigma_theta'] == 1.0) & (agg_sim_df['sigma_Z'] == 1.0)].sort_values(by='embedding_dim')

plot_comparison(filtered_data['embedding_dim'],
                filtered_data['sample_cov_SSE_mean'],
                filtered_data['ledoit_wolf_SSE_mean'],
                filtered_data['latent_model_SSE_mean'],
                filtered_data['sample_cov_SSE_std'],
                filtered_data['ledoit_wolf_SSE_std'],
                filtered_data['latent_model_SSE_std'],
                'Embedding Dimension',
                'SSE vs True Covariance Matrix',
                'SSE vs Embedding Dimension',
                './covariance_plots/embedding_sse.png')

plot_comparison(filtered_data['embedding_dim'],
                filtered_data['sample_cov_RMSE_mean'],
                filtered_data['ledoit_wolf_RMSE_mean'],
                filtered_data['latent_model_RMSE_mean'],
                filtered_data['sample_cov_RMSE_std'],
                filtered_data['ledoit_wolf_RMSE_std'],
                filtered_data['latent_model_RMSE_std'],
                'Embedding Dimension',
                'RMSE vs True Covariance Matrix',
                'RMSE vs Embedding Dimension',
                './covariance_plots/embedding_rmse.png')

# Plot 2: Feature Count
filtered_data = agg_sim_df.loc[(agg_sim_df['n_samples'] == 50) & (agg_sim_df['embedding_dim'] == 10) &
                               (agg_sim_df['sigma_theta'] == 1.0) & (agg_sim_df['sigma_Z'] == 1.0)].sort_values(by='n_features')

plot_comparison(filtered_data['n_features'],
                filtered_data['sample_cov_SSE_mean'],
                filtered_data['ledoit_wolf_SSE_mean'],
                filtered_data['latent_model_SSE_mean'],
                filtered_data['sample_cov_SSE_std'],
                filtered_data['ledoit_wolf_SSE_std'],
                filtered_data['latent_model_SSE_std'],
                'Feature Count',
                'SSE vs True Covariance Matrix',
                'SSE vs Feature Count',
                './covariance_plots/features_sse.png',
                x_log=True, y_log=True)

plot_comparison(filtered_data['n_features'],
                filtered_data['sample_cov_RMSE_mean'],
                filtered_data['ledoit_wolf_RMSE_mean'],
                filtered_data['latent_model_RMSE_mean'],
                filtered_data['sample_cov_RMSE_std'],
                filtered_data['ledoit_wolf_RMSE_std'],
                filtered_data['latent_model_RMSE_std'],
                'Feature Count',
                'RMSE vs True Covariance Matrix',
                'RMSE vs Feature Count',
                './covariance_plots/features_rmse.png',
                x_log=True)

# Plot 3: Sample Count
filtered_data = agg_sim_df.loc[(agg_sim_df['n_features'] == 100) & (agg_sim_df['embedding_dim'] == 10) &
                               (agg_sim_df['sigma_theta'] == 1.0) & (agg_sim_df['sigma_Z'] == 1.0)].sort_values(by='n_samples')

plot_comparison(filtered_data['n_samples'],
                filtered_data['sample_cov_SSE_mean'],
                filtered_data['ledoit_wolf_SSE_mean'],
                filtered_data['latent_model_SSE_mean'],
                filtered_data['sample_cov_SSE_std'],
                filtered_data['ledoit_wolf_SSE_std'],
                filtered_data['latent_model_SSE_std'],
                'Sample Count',
                'SSE vs True Covariance Matrix',
                'SSE vs Sample Count',
                './covariance_plots/samples_sse.png',
                x_log=True, y_log=True)

plot_comparison(filtered_data['n_samples'],
                filtered_data['sample_cov_RMSE_mean'],
                filtered_data['ledoit_wolf_RMSE_mean'],
                filtered_data['latent_model_RMSE_mean'],
                filtered_data['sample_cov_RMSE_std'],
                filtered_data['ledoit_wolf_RMSE_std'],
                filtered_data['latent_model_RMSE_std'],
                'Sample Count',
                'RMSE vs True Covariance Matrix',
                'RMSE vs Sample Count',
                './covariance_plots/samples_rmse.png',
                x_log=True)

# Plot 4: Model Misspecification
filtered_data = agg_sim_df.loc[(agg_sim_df['n_features'] == 100) & (agg_sim_df['n_samples'] == 50) &
                               (agg_sim_df['embedding_dim'] == 10) & (agg_sim_df['sigma_Z'] == 1.0)].sort_values(by='sigma_theta')

plot_comparison(filtered_data['sigma_theta'],
                filtered_data['sample_cov_SSE_mean'],
                filtered_data['ledoit_wolf_SSE_mean'],
                filtered_data['latent_model_SSE_mean'],
                filtered_data['sample_cov_SSE_std'],
                filtered_data['ledoit_wolf_SSE_std'],
                filtered_data['latent_model_SSE_std'],
                'Sigma_theta',
                'SSE vs True Covariance Matrix',
                'SSE vs Model Misspecification',
                './covariance_plots/theta_sse.png',
                y_log=True)

plot_comparison(filtered_data['sigma_theta'],
                filtered_data['sample_cov_RMSE_mean'],
                filtered_data['ledoit_wolf_RMSE_mean'],
                filtered_data['latent_model_RMSE_mean'],
                filtered_data['sample_cov_RMSE_std'],
                filtered_data['ledoit_wolf_RMSE_std'],
                filtered_data['latent_model_RMSE_std'],
                'Sigma_theta',
                'RMSE vs True Covariance Matrix',
                'RMSE vs Model Misspecification',
                './covariance_plots/theta_rmse.png')

# Plot 5: Latent Position Variability
filtered_data = agg_sim_df.loc[(agg_sim_df['n_features'] == 100) & (agg_sim_df['n_samples'] == 50) &
                               (agg_sim_df['embedding_dim'] == 10) & (agg_sim_df['sigma_theta'] == 1.0)].sort_values(by='sigma_Z')

plot_comparison(filtered_data['sigma_Z'],
                filtered_data['sample_cov_SSE_mean'],
                filtered_data['ledoit_wolf_SSE_mean'],
                filtered_data['latent_model_SSE_mean'],
                filtered_data['sample_cov_SSE_std'],
                filtered_data['ledoit_wolf_SSE_std'],
                filtered_data['latent_model_SSE_std'],
                'Sigma_Z',
                'SSE vs True Covariance Matrix',
                'SSE vs Latent Position Variability',
                './covariance_plots/z_sse.png',
                x_log=True, y_log=True)

plot_comparison(filtered_data['sigma_Z'],
                filtered_data['sample_cov_RMSE_mean'],
                filtered_data['ledoit_wolf_RMSE_mean'],
                filtered_data['latent_model_RMSE_mean'],
                filtered_data['sample_cov_RMSE_std'],
                filtered_data['ledoit_wolf_RMSE_std'],
                filtered_data['latent_model_RMSE_std'],
                'Sigma_Z',
                'RMSE vs True Covariance Matrix',
                'RMSE vs Latent Position Variability',
                './covariance_plots/z_rmse.png',
                x_log=True, y_log=True)