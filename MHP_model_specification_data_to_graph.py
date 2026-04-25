import pandas as pd
import plotly.express as px

df = pd.read_csv('results/MHP_model_misspecification.csv')

df = df.rename({"frobenius error":"Model Frobenius error",
                "rmse error": "Model RMSE",
                "frobenius adm4":"ADM4 Frobenius error",
                "rmse adm4": "ADM4 RMSE"
                },
                axis=1)

fig = px.line(
    df, x='misspecification', y=['Model Frobenius error', 'Model RMSE', 'ADM4 Frobenius error', "ADM4 RMSE"], title="Error vs misspecification"
)

fig.update_xaxes(
    title_text="Model misspecification: σ_θ",
    # tickmode="array",
    tickvals=list(df['misspecification']),
    # ticktext=[str(x) for x in df.columns],
    # side="bottom"
)

fig.update_yaxes(
    title_text="Errors",
    # tickmode="array",
    # tickvals=list(df['misspecification']),
    # ticktext=[str(y) for y in df.index]
)

fig.write_image("results/MHP_model_misspecification.pdf")