"""
Visualization Utilities

This module provides reusable visualization components for the dashboard.
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from typing import List, Dict, Any, Tuple
import squarify

# Set default styling for matplotlib
plt.style.use('ggplot')
sns.set_style('whitegrid')

def create_metric_card(title: str, value: Any, delta: float = None, 
                     delta_text: str = None, help_text: str = None):
    """
    Create a metric card with an optional delta indicator.
    
    Args:
        title: The metric title
        value: The metric value
        delta: Optional percentage change
        delta_text: Optional text to display instead of percentage
        help_text: Optional help text
    """
    if delta is not None:
        st.metric(
            label=title,
            value=value,
            delta=delta_text if delta_text else f"{delta:.1f}%",
            help=help_text
        )
    else:
        st.metric(
            label=title,
            value=value,
            help=help_text
        )

def plot_time_series(df: pd.DataFrame, x_col: str, y_col: str, title: str, 
                    color: str = '#1f77b4', height: int = 400, 
                    show_events: bool = False, events_df: pd.DataFrame = None):
    """
    Create a time series plot with optional event markers.
    
    Args:
        df: DataFrame containing the data
        x_col: Column name for x-axis (time)
        y_col: Column name for y-axis (metric)
        title: Plot title
        color: Line color
        height: Plot height
        show_events: Whether to show event markers
        events_df: DataFrame containing event data
    """
    fig = px.line(df, x=x_col, y=y_col, title=title, height=height)
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        title_x=0.5,
        title_font_size=16
    )
    
    if show_events and events_df is not None:
        for _, event in events_df.iterrows():
            fig.add_vline(
                x=event['event_start'],
                line_width=1,
                line_dash="dash",
                line_color="red",
                annotation_text=event['name'] if 'name' in event else event['type'],
                annotation_position="top right"
            )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_heatmap(df: pd.DataFrame, x_col: str, y_col: str, value_col: str, 
                title: str, height: int = 500, color_scale: str = 'Greens'):
    """
    Create a heatmap.
    
    Args:
        df: DataFrame containing the data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        value_col: Column name for cell values
        title: Plot title
        height: Plot height
        color_scale: Color scale for the heatmap
    """
    pivot_table = df.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc='mean')
    
    fig = px.imshow(
        pivot_table,
        title=title,
        color_continuous_scale=color_scale,
        labels=dict(x=x_col, y=y_col, color=value_col),
        height=height
    )
    
    fig.update_layout(
        title_x=0.5,
        title_font_size=16
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, 
                 color: str = '#1f77b4', horizontal: bool = False, 
                 height: int = 400):
    """
    Create a bar chart.
    
    Args:
        df: DataFrame containing the data
        x_col: Column name for categories
        y_col: Column name for values
        title: Plot title
        color: Bar color
        horizontal: Whether to create a horizontal bar chart
        height: Plot height
    """
    if horizontal:
        fig = px.bar(df, y=x_col, x=y_col, title=title, orientation='h', height=height)
    else:
        fig = px.bar(df, x=x_col, y=y_col, title=title, height=height)
    
    fig.update_layout(
        title_x=0.5,
        title_font_size=16
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_treemap(df: pd.DataFrame, path_cols: List[str], values_col: str, 
               title: str, height: int = 500):
    """
    Create a treemap visualization.
    
    Args:
        df: DataFrame containing the data
        path_cols: List of column names for hierarchical paths
        values_col: Column name for cell values/sizes
        title: Plot title
        height: Plot height
    """
    fig = px.treemap(
        df,
        path=path_cols,
        values=values_col,
        title=title,
        height=height
    )
    
    fig.update_layout(
        title_x=0.5,
        title_font_size=16
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_pie_chart(df: pd.DataFrame, names_col: str, values_col: str, 
                 title: str, height: int = 400):
    """
    Create a pie chart.
    
    Args:
        df: DataFrame containing the data
        names_col: Column name for segment names
        values_col: Column name for segment values
        title: Plot title
        height: Plot height
    """
    fig = px.pie(
        df,
        names=names_col,
        values=values_col,
        title=title,
        height=height
    )
    
    fig.update_layout(
        title_x=0.5,
        title_font_size=16
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_gauge(value: float, min_val: float, max_val: float, title: str, 
              threshold: float = None, height: int = 300):
    """
    Create a gauge chart for a single value.
    
    Args:
        value: The value to display
        min_val: Minimum value on the gauge
        max_val: Maximum value on the gauge
        title: Plot title
        threshold: Optional threshold value to mark
        height: Plot height
    """
    # Create a half-circle gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "blue"},
            'steps': [
                {'range': [min_val, (max_val-min_val)/2], 'color': "lightgray"},
                {'range': [(max_val-min_val)/2, max_val], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold if threshold is not None else max_val
            }
        }
    ))
    
    fig.update_layout(
        height=height,
        title_x=0.5,
        title_font_size=16
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_map(df: pd.DataFrame, lat_col: str, lon_col: str, color_col: str = None, 
           size_col: str = None, hover_data: List[str] = None, title: str = "",
           height: int = 500):
    """
    Create a map visualization with data points.
    
    Args:
        df: DataFrame containing the data
        lat_col: Column name for latitude
        lon_col: Column name for longitude
        color_col: Optional column name for point colors
        size_col: Optional column name for point sizes
        hover_data: List of column names to show on hover
        title: Plot title
        height: Plot height
    """
    fig = px.scatter_mapbox(
        df,
        lat=lat_col,
        lon=lon_col,
        color=color_col,
        size=size_col,
        hover_name=hover_data[0] if hover_data else None,
        hover_data=hover_data,
        zoom=11,
        height=height,
        title=title
    )
    
    fig.update_layout(
        mapbox_style="open-street-map",
        title_x=0.5,
        title_font_size=16
    )
    
    st.plotly_chart(fig, use_container_width=True) 