import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import streamlit as st

@st.cache_resource
def plot_predictions(df, target_variable):
    """
    Create an interactive plot with True Values and Model Predictions
    """
    
    fig = go.Figure()

    # True values
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df[target_variable], 
        mode="lines", 
        name="True", 
        line=dict(color="blue")
    ))

    # Model predictions
    if "Original Prediction" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df["Original Prediction"], 
            mode="lines", 
            name="Original Prediction", 
            line=dict(color="red")
        ))


    fig.update_layout(
        title="Actual vs Predicted Values",
        xaxis=dict(title="Index", gridwidth=1, showgrid=True),
        yaxis=dict(title="Target Value", gridwidth=1),
        paper_bgcolor='#0e1117',
        plot_bgcolor='#262730',
        width=None,
        font=dict(color='#FFFFFF', size=15),
        xaxis_rangeslider_visible=True,
    )

    return fig

def plot_updated_predictions(df, target_variable, xaxis_range):
    """
    Create an interactive plot with True Values, Model Predictions, and Modified Predictions
    """
    
    fig = go.Figure()

    # True
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df[target_variable], 
        mode="lines", 
        name="True", 
        line=dict(color="blue")
    ))

    # Original Prediction
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df["Original Prediction"], 
        mode="lines", 
        name="Original Prediction", 
        line=dict(color="red", dash="dot")
    ))

    # Modified Prediction
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df["Modified Prediction"], 
        mode="lines", 
        name="Modified Prediction", 
        line=dict(color="green", dash="dash")
    ))

    fig.update_layout(
        title="Updated Predictions (Zoom Applied)",
        xaxis=dict(
            title="Index", 
            gridwidth=1, 
            showgrid=True,
            range=xaxis_range  
        ),
        yaxis=dict(title="Target Value", gridwidth=1),
        width=None,
        paper_bgcolor='#0e1117',
        plot_bgcolor='#262730',
        font=dict(color='#FFFFFF', size=15),
        xaxis_rangeslider_visible=True,
    )

    return fig