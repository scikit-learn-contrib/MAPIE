from typing import Optional

import numpy as np
import plotly.graph_objects as go


font_dict = dict(family="Computer Modern", size=18, color="#7f7f7f")


def plot_regression(
    X: np.ndarray,
    y: np.ndarray,
    y_mesh: np.ndarray = None,
    name_mesh: Optional[str] = None,
    title: Optional[str] = None,
    xaxis_title: str = "x",
    yaxis_title: str = "y"
) -> go.Figure:

    fig = go.Figure()

    # data
    fig.add_trace(
        go.Scatter(
            x=X.ravel(),
            y=y,
            mode="markers",
            marker=dict(color="#1f77b4", size=2),
            name="data",
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=X.ravel(),
            y=y_mesh,
            mode="lines",
            line=dict(color="#ff7f0e", dash="solid"),
            name=name_mesh,
        )
    )
    
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        font=font_dict,
        hovermode="x",
    )
    return fig


def plot_uncertainties(
    X: np.ndarray,
    y: np.ndarray,
    y_preds: np.ndarray,
    y_pis: np.ndarray,
    title: Optional[str] = None,
    xaxis_title: str = "x",
    yaxis_title: str = "y"
) -> go.Figure:
    fig = go.Figure()

    # lower/upper bounds
    fig.add_trace(
        go.Scatter(
            x=X.ravel(),
            y=y_pis[:, 0, 0],
            mode="lines",
            line=dict(color="#ff7f0e", dash="solid"),
            name="lower bound",
            showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=X.ravel(),
            y=y_pis[:, 1, 0],
            mode="lines",
            fill="tonexty",
            line=dict(color="#ff7f0e", dash="solid"),
            name="upper bound",
            showlegend=False
        )
    )

    # predictions
    fig.add_trace(go.Scatter(
        name="predictions",
        x=X.ravel(),
        y=y_preds,
        mode="lines",
        line=dict(color="#008000", dash="solid")
    ))

    # data
    fig.add_trace(go.Scatter(
        name="data",
        x=X.ravel(),
        y=y,
        mode="markers",
        marker=dict(color="#1f77b4", size=2)
    ))

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        font=font_dict,
        hovermode="x",
    )
    return fig


def plot_two_uncertainties(
    X: np.ndarray,
    y: np.ndarray,
    y_preds: np.ndarray,
    y_pis_conformal: np.ndarray,
    y_pis_naive: np.ndarray,
    title: Optional[str] = None,
    xaxis_title: str = "x",
    yaxis_title: str = "y"
) -> go.Figure:
    fig = go.Figure()

    # conformal lower/upper bounds
    fig.add_trace(
        go.Scatter(
            x=X.ravel(),
            y=y_pis_conformal[:, 0, 0],
            mode="lines",
            line=dict(color="#ff7f0e", dash="solid"),
            name="Conformal interval",
            legendgroup="conformal_interval",
            showlegend=True
        )
    )
    fig.add_trace(
        go.Scatter(
            x=X.ravel(),
            y=y_pis_conformal[:, 1, 0],
            mode="lines",
            fill="tonexty",
            line=dict(color="#ff7f0e", dash="solid"),
            name="conformal upper bound",
            legendgroup="conformal_interval",
            showlegend=False
        )
    )

    # naive lower/upper bounds (single legend entry)
    fig.add_trace(
        go.Scatter(
            x=X.ravel(),
            y=y_pis_naive[:, 0, 0],
            mode="lines",
            line=dict(color="#FFD700", dash="dash"),
            name="Naive Interval",               # name shown in legend
            legendgroup="naive_interval",        # group ID
            showlegend=True                      # show this one in legend
        )
    )
    fig.add_trace(
        go.Scatter(
            x=X.ravel(),
            y=y_pis_naive[:, 1, 0],
            mode="lines",
            fill="tonexty",
            line=dict(color="#FFD700", dash="dash"),
            name="Naive Interval",               # same name (optional)
            legendgroup="naive_interval",        # same group ID
            showlegend=False                     # only show one in legend
        )
    )

    # predictions
    fig.add_trace(go.Scatter(
        name="predictions",
        x=X.ravel(),
        y=y_preds,
        mode="lines",
        line=dict(color="#008000", dash="solid")
    ))

    # data
    fig.add_trace(go.Scatter(
        name="data",
        x=X.ravel(),
        y=y,
        mode="markers",
        marker=dict(color="#1f77b4", size=2)
    ))

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        font=font_dict,
        hovermode="x",
    )
    return fig


def plot_prediction_interval_width(
    X_test: np.ndarray,
    y_pis: np.ndarray,
    title: Optional[str] = None,
    xaxis_title: str = "x",
    yaxis_title: str = "y"
):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=X_test.ravel(),
            y=abs(y_pis[:, 1, 0] - y_pis[:, 0, 0])
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        font=font_dict,
        hovermode="x",
    )
    return fig


def plot_probability_curve(X, y):
    fig = go.Figure()

    for name, y_score in y.items():
        fig.add_trace(
            go.Scatter(
                x=X,
                y=y_score,
                name=name
            )
        )

    fig.update_layout(
        title="Probability curves",
        xaxis_title="X",
        yaxis_title="P(Y=1|X)",
        font=font_dict,
        hovermode="x",
    )
    
    return fig


def plot_calibration_curve(X, y):
    fig = go.Figure()

    for name, curve in y.items():
        fig.add_trace(
            go.Scatter(
                x=curve[1],
                y=curve[0],
                mode="lines+markers",
                name=name
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line_dash="dash",
            line_color="black",
            name="perfect calibration"
        )
    )

    fig.update_layout(
        title="Standard calibration curves",
        xaxis_title="Score",
        yaxis_title="Positive rate",
        font=font_dict,
        hovermode="x",
    )
    
    return fig


def plot_cumulative_calibration_curve(cum_diffs):
    fig = go.Figure()

    for name, cum_diff in cum_diffs.items():
        k = np.arange(len(cum_diff))/len(cum_diff)

        fig.add_trace(
            go.Scatter(
                x=k,
                y=cum_diff,
                name=name,
                legendgroup=name
            )
        )

    fig.update_layout(
        title="Cumulative calibration curves",
        xaxis_title="Proportion of scores considered",
        yaxis_title="Normalized cumulative differences with the ground truth",
        font=font_dict,
        hovermode="x",
    )

    return fig
