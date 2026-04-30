import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def create_fig(df, x_col, y_cols, y_stds, title, x_label, y_label, model_parameters):
    fig1 = individual_fig(df, x_col, y_cols[0], y_stds[0], title, x_label, y_label, model_parameters, 'dodgerblue')
    fig2 = individual_fig(df, x_col, y_cols[1], y_stds[1], title, x_label, y_label, model_parameters, 'darkorange')
    fig = go.Figure(data=fig1.data+fig2.data)

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

    fig.update_layout(legend=dict(x=0.75,y=1.25),
                      plot_bgcolor='white',
                      title=go.layout.Title(text=f"{title}<br><sup>{model_parameters}</sup>"),
                    #   labels={"variable":"Metric"},
                      )

    return fig

def individual_fig(df, x_col, y_col, y_std, title, x_label, y_label, model_parameters, color):
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        error_y=y_std,
        markers=True,
        # title=title,
        # subtitle=model_parameters,
        # labels={"variable":"Metric"},
        color_discrete_sequence=[color],
    )

    fig.for_each_trace(lambda t: t.update(name=y_col))
    fig.update_traces(showlegend=True)

    # fig.update_xaxes(
    #     title_text=x_label,
    #     tickvals=list(df[x_col]),
    #     mirror=True,
    #     ticks='outside',
    #     showline=True,
    #     linecolor='black',
    #     gridcolor='lightgrey',
    # )

    # fig.update_yaxes(
    #     title_text=y_label,
    #     mirror=True,
    #     ticks='outside',
    #     showline=True,
    #     linecolor='black',
    #     gridcolor='lightgrey'
    # )

    # fig.update_layout(legend=dict(x=0.75,y=1.25), plot_bgcolor='white')
    # fig.show()
    return fig

if __name__ == "__main__":
    df = pd.read_csv('results/MHP_parameter_count.csv')
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
    model_parameters = 'mu=0.2/p, beta=1, time=400, d=2, σ_z=1, σ_θ=1.5'

    fig_rmse = create_fig(df, 'parameter count', ['LSM RMSE', "ADM4 RMSE"], ['LSM RMSE std', "ADM4 RMSE std"], "RMSE vs p", "p (parameter count)", "RMSE", model_parameters)
    fig_rmse.write_image("results/MHP_parameter_count_RMSE.pdf")

    fig_frob = create_fig(df, 'parameter count', ['LSM Frobenius error', "ADM4 Frobenius error"], ['LSM Frobenius error std', "ADM4 Frobenius error std"], "Frobenius error vs p", "p (parameter count)", "Frobenius error", model_parameters)
    fig_frob.write_image("results/MHP_parameter_count_Frobenius.pdf")
