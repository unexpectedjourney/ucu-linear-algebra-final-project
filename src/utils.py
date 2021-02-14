import json

import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def movies_info_preproc(movies_info_df):
    movies_info_df = movies_info_df.copy()
    movies_info_df["genre"] = movies_info_df["info"] \
        .apply(lambda s: ", ".join([g["name"] for g in json.loads(s)["genres"]]))

    movies_info_df["overview"] = movies_info_df["info"].apply(lambda s: json.loads(s)["overview"])

    movies_info_df["poster_url"] = movies_info_df["info"].apply(
        lambda s: "https://image.tmdb.org/t/p/original" + json.loads(s)["poster_path"] \
                  if json.loads(s)["poster_path"] is not None else None
    )

    movies_info_df["language"] = movies_info_df["info"] \
        .apply(lambda s: json.loads(s)["original_language"])

    def get_alt_title(s):
        titles = json.loads(s)["titles"]
        for t in titles:
            if t["iso_3166_1"] in ["RU", "UA"]:
                return t["title"]
        return None

    movies_info_df["alternative_title"] = movies_info_df.alternative_titles.apply(get_alt_title)
    movies_info_df = movies_info_df.drop(columns=["info", "alternative_titles", "keywords"])
    return movies_info_df


# TODO rename
def get_movie_embedded_df(movie_emb, df, movies_info_df, n_clusters=10):
    movie_clusters = KMeans(n_clusters).fit_predict(movie_emb)
    np.unique(movie_clusters)
    movie_embedded = TSNE(n_components=2, random_state=1).fit_transform(movie_emb)

    movie_embedded_df = pd.DataFrame({
        "x_cord": movie_embedded[:, 0],
        "y_cord": movie_embedded[:, 1],
        "cluster": movie_clusters,
    })

    movie_embedded_df = pd.merge(
        movie_embedded_df,
        df.groupby(["movie_id", "movie_name", "release_date"]) \
          .size().reset_index() \
          .rename(columns={0: "watch_count"}),
        left_index=True,
        right_on="movie_id"
    )

    # Warning not all mvoies has meta info
    movie_embedded_df = pd.merge(
        movie_embedded_df,
        movies_info_df,
        on="movie_name",
    )
    return movie_embedded_df


# TODO rename
def get_user_embedded_df(user_emb, df, movies_info_df, n_clusters=10):
    user_df = pd.merge(
        df,
        movies_info_df,
        on="movie_name"
    )
    user_df = user_df.groupby("customer_id").agg({
        "genre": lambda x: x.value_counts().index[0],
        "release_date": lambda x: x.mean().round(),
    }).reset_index()


    user_clusters = KMeans(n_clusters).fit_predict(user_emb)
    np.unique(user_clusters)
    user_embedded = TSNE(n_components=2, random_state=1).fit_transform(user_emb)

    user_embedded_df = pd.DataFrame({
        "x_cord": user_embedded[:, 0],
        "y_cord": user_embedded[:, 1],
        "cluster": user_clusters,
    })

    user_embedded_df = pd.merge(
        user_embedded_df,
        user_df,
        left_index=True,
        right_on="customer_id"
    )
    return user_embedded_df
