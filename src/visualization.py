import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt


def plot_movies_latent_space(movie_embedded_df):
    alt.data_transformers.disable_max_rows()
    cluster_select = alt.selection_single(empty="all", fields=["cluster"], on="mouseover", bind="legend")

    movies_latent_chart = alt.Chart(
        movie_embedded_df
    ).mark_circle(

    ).encode(
        x=alt.X("x_cord:Q", title="dim 1"),
        y=alt.Y("y_cord:Q", title="dim 2"),
        color="cluster:N",
        tooltip=[
            alt.Tooltip("movie_name:N", title="movie"),
            alt.Tooltip("alternative_title:N", title="alt title"),
            alt.Tooltip("release_date:N", title="release year"),
            alt.Tooltip("watch_count:N", title="watched"),
            alt.Tooltip("genre:N", title="genre"),
            alt.Tooltip("overview:N", title="overview"),
        ],
        opacity=alt.condition(cluster_select, alt.value(0.7),alt.value(0.1)),
        size=alt.value(100)
    ).properties(
        width=1000, height=600,
        title="Movies' latent space"
    ).add_selection(
        cluster_select
    )
    return movies_latent_chart


def plot_user_latent_space(user_embedded_df):
    alt.data_transformers.disable_max_rows()
    cluster_select = alt.selection_single(empty="all", fields=["cluster"], on="mouseover", bind="legend")

    user_latent_chart = alt.Chart(
        user_embedded_df
    ).mark_circle(
    ).encode(
        x=alt.X("x_cord:Q", title="dim 1"),
        y=alt.Y("y_cord:Q", title="dim 2"),
        color="cluster:N",
        tooltip=[
            alt.Tooltip("customer_id:N", title="user id"),
            alt.Tooltip("genre:N", title="top genre"),
            alt.Tooltip("release_date:N", title="top release date"),
        ],
        opacity=alt.condition(cluster_select, alt.value(0.7),alt.value(0.1)),
        size=alt.value(100)
    ).properties(
        width=1000, height=600,
        title="Users' latent space"
    ).add_selection(
        cluster_select
    )
    return user_latent_chart


def plot_relative_prediction_error(df, column, algo, save=False, save_path=None):


    error_chart = alt.Chart(df).mark_bar().encode(
        x=alt.X(f"{column}:Q", bin=alt.Bin(extent=[-100, 100], step=10), title="Error percentage"),
        y=alt.Y('count()', title="Number of records")
    ).properties(
        width=1000, height=600,
        title=f"{algo}'s relative prediction errors"
    )
    if save:
        error_chart.save(save_path)
    return error_chart

