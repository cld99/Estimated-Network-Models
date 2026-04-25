import pandas as pd
import plotly.express as px

df = pd.read_csv('results/MHP_parameter_count.csv')

df = df.rename({"frobenius error":"Model Frobenius error",
                "rmse error": "Model RMSE",
                "frobenius adm4":"ADM4 Frobenius error",
                "rmse adm4": "ADM4 RMSE"
                },
                axis=1)

fig = px.line(
    df, x='parameter count', y=['Model Frobenius error', 'Model RMSE', 'ADM4 Frobenius error', "ADM4 RMSE"], title="Error vs parameter count"
)

fig.update_xaxes(
    title_text="Parameter count: number of variables",
    # tickmode="array",
    tickvals=list(df['parameter count']),
    # ticktext=[str(x) for x in df.columns],
    # side="bottom"
)

fig.update_yaxes(
    title_text="Errors",
    # tickmode="array",
    # tickvals=list(df.index),
    # ticktext=[str(y) for y in df.index]
)

fig.write_image("results/MHP_parameter_count.pdf")