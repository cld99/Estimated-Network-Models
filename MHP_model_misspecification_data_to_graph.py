import pandas as pd
import plotly.express as px
from MHP_parameter_count_data_to_graph import create_fig

df = pd.read_csv('results/MHP_model_misspecification.csv')
df = df.rename({"frobenius error mean":"LSM Frobenius error",
                    "frobenius error std":"LSM Frobenius error std",
                    "rmse error mean": "LSM RMSE",
                    "rmse error std": "LSM RMSE std",
                    "frobenius adm4 mean":"ADM4 Frobenius error",
                    "frobenius adm4 std":"ADM4 Frobenius error std",
                    "rmse adm4 mean": "ADM4 RMSE",
                    "rmse adm4 std": "ADM4 RMSE std"
                },
                axis=1)
model_parameters = 'p=2, mu=0.1, beta=1, time=400, d=2, σ_z=1'

fig_rmse = create_fig(df, 'misspecification', ['LSM RMSE', "ADM4 RMSE"], ['LSM RMSE std', "ADM4 RMSE std"], "RMSE vs σ_θ", "σ_θ (model misspecification)", "RMSE",model_parameters)
fig_rmse.write_image("results/MHP_model_misspecification_RMSE.pdf")

fig_frob = create_fig(df, 'misspecification', ['LSM Frobenius error', "ADM4 Frobenius error"], ['LSM Frobenius error std', "ADM4 Frobenius error std"], "Frobenius error vs σ_θ", "σ_θ (model misspecification)", "Frobenius error", model_parameters)
fig_frob.write_image("results/MHP_model_misspecification_Frobenius.pdf")
