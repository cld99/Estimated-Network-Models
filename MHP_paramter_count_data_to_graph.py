import pandas as pd
import plotly.express as px

def create_fig(df, x_col, y_cols, title, x_label, y_label, model_parameters):
    fig = px.line(
        df,
        x=x_col,
        y=y_cols,
        markers=True,
        title=title,
        subtitle=model_parameters,
        labels={"variable":"Metric"},
        color_discrete_sequence=['dodgerblue', 'darkorange'],
    )

    fig.update_xaxes(
        title_text=x_label,
        tickvals=list(df[x_col]),
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey',
    )

    fig.update_yaxes(
        title_text=y_label,
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )

    fig.update_layout(legend=dict(x=0.75,y=1.25), plot_bgcolor='white')
    return fig

if __name__ == "__main__":
    df = pd.read_csv('results/MHP_parameter_count.csv')
    df = df.rename({"frobenius error":"LSM Frobenius error",
                    "rmse error": "LSM RMSE",
                    "frobenius adm4":"ADM4 Frobenius error",
                    "rmse adm4": "ADM4 RMSE"
                    },
                    axis=1)
    model_parameters = 'mu=0.2/p, beta=1, time=400, d=2, σ_z=1, σ_θ=1.5'

    fig_rmse = create_fig(df, 'parameter count', ['LSM RMSE', "ADM4 RMSE"], "RMSE vs p", "p (parameter count)", "RMSE", model_parameters)
    fig_rmse.write_image("results/MHP_parameter_count_RMSE.pdf")

    fig_frob = create_fig(df, 'parameter count', ['LSM Frobenius error', "ADM4 Frobenius error"], "Frobenius error vs p", "p (parameter count)", "Frobenius error", model_parameters)
    fig_frob.write_image("results/MHP_parameter_count_Frobenius.pdf")