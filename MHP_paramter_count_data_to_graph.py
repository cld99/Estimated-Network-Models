import pandas as pd
import plotly.express as px

df = pd.read_csv('results/MHP_parameter_count.csv')

df = df.rename({"frobenius error": "Frobenius error", "rmse error": "RMSE error"}, axis=1)

fig = px.line(
    df, x='parameter count', y=['Frobenius error', 'RMSE error'], title="Error vs parameter count"
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