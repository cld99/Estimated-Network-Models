import pandas as pd
import plotly.express as px
from MHP_paramter_count_data_to_graph import create_fig

df = pd.read_csv('results/MHP_model_misspecification.csv')
df = df.rename({"frobenius error":"LSM Frobenius error",
                "rmse error": "LSM RMSE",
                "frobenius adm4":"ADM4 Frobenius error",
                "rmse adm4": "ADM4 RMSE"
                },
                axis=1)
model_parameters = 'p=2, mu=0.1, beta=1, time=400, d=2, σ_z=1'

fig_rmse = create_fig(df, 'misspecification', ['LSM RMSE', "ADM4 RMSE"], "RMSE vs σ_θ", "σ_θ (model misspecification)", "RMSE",model_parameters)
fig_rmse.write_image("results/MHP_model_misspecification_RMSE.pdf")

fig_frob = create_fig(df, 'misspecification', ['LSM Frobenius error', "ADM4 Frobenius error"], "Frobenius error vs σ_θ", "σ_θ (model misspecification)", "Frobenius error", model_parameters)
fig_frob.write_image("results/MHP_model_misspecification_Frobenius.pdf")