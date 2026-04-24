import pandas as pd
import plotly.express as px

df = pd.read_csv('results/MHP_model_misspecification.csv')

df = df.rename({"frobenius error": "Frobenius error", "rmse error": "RMSE error"}, axis=1)

fig = px.line(
    df, x='misspecification', y=['Frobenius error', 'RMSE error'], title="Error vs misspecification"
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