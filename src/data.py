import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm


def get_netflix_dataframe(datapath, is_train=True):
    with open(datapath, "r") as file:
        lines = file.readlines()
        movie_id = -1

        movie_ids = []
        customer_ids = []
        ratings = []
        dates = []

        for line in tqdm(lines):
            line_elements = line.strip().split(",")
            if len(line_elements) == 1 and line_elements[0].endswith(":"):
                movie_id = int(line_elements[0][:-1])
                continue
            movie_ids.append(movie_id)
            customer_ids.append(line_elements[0])
            if is_train:
                ratings.append(line_elements[1])
                dates.append(line_elements[2])

        if is_train:
            df = pd.DataFrame({
                "movie_id": np.array(movie_ids, dtype=int),
                "customer_id": np.array(customer_ids, dtype=int),
                "rating": np.array(ratings, dtype=np.float32),
                "date": dates
            })
        else:
            df = pd.DataFrame({
                "movie_id": np.array(movie_ids, dtype=int),
                "customer_id": np.array(customer_ids, dtype=int),
            })

        return df


def combine_dataframes(df_list):
    result_df = pd.DataFrame()
    for df in df_list:
        result_df.append(df)

    return df


def generate_sparce_matrix(df):
    customer_ids = df.customer_id.tolist()

    movie_ids = df.movie_id.tolist()
    ratings = df.rating.tolist()

    coo_matrix = sparse.coo_matrix((ratings, (customer_ids, movie_ids)))

    return coo_matrix
