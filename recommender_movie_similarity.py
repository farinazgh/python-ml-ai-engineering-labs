import pandas as pd

# People who rated Star Wars (1977) highly — what other movies did they also tend to rate similarly?

# the recommendation is based mainly on this metric: Pearson correlation similarity with Star Wars ratings

# load user ratings
# ↓
# load movie titles
# ↓
# merge ratings with movie names
# ↓
# create a user × movie rating matrix
# ↓
# take the Star Wars rating column
# ↓
# compare every movie column with Star Wars
# ↓
# filter out unpopular movies
# ↓
# show movies most similar to Star Wars


def load_movielens_data():

    rating_columns = ["user_id", "movie_id", "rating"]
    ratings = pd.read_csv(
        "movie-data.data",
        sep="\t",
        names=rating_columns,
        usecols=range(3),  # Read only the first 3 columns.
        encoding="ISO-8859-1",  # movie titles with special characters.
    )

    movie_columns = ["movie_id", "title"]
    movies = pd.read_csv(
        "movie-data.item",  # movie metadata
        sep="|",
        names=movie_columns,
        usecols=range(2),
        encoding="ISO-8859-1",
    )

    return pd.merge(
        movies, ratings, on="movie_id"
    )  # merge based on the shared column movie_id


def build_user_movie_matrix(ratings):
    """
    Build a user-item matrix:
    one row per user
    one column per movie
        - rows are users
        - columns are movie titles
        - values are ratings
        - NaN: This user did not rate this movie.
    """

    return ratings.pivot_table(
        index="user_id",
        columns="title",
        values="rating",
    )


# how many people rated each movie, and what the average rating was.
# Group all rows by movie title.
# For each movie title, look at its rating column.
# Then calculate:
# - size = how many ratings this movie has
# - mean = average rating for this movie
# title                size    mean
# Star Wars (1977)    3       4.67
# Toy Story (1995)    2       4.50
def calculate_movie_statistics(ratings):

    movie_stats = ratings.groupby("title")["rating"].agg(["size", "mean"])
    movie_stats.columns = ["rating_count", "average_rating"]

    return movie_stats


# +1  = very similar rating pattern
#  0  = no clear relationship
# -1  = opposite rating pattern
def recommend_similar_movies(
    movie_ratings,
    movie_stats,
    target_movie,
    minimum_rating_count=100,
    number_of_recommendations=15,
):
    """
    Find movies with rating patterns similar to the target movie.
    """

    if target_movie not in movie_ratings.columns:
        raise ValueError(f"Movie not found: {target_movie}")

    target_movie_ratings = movie_ratings[target_movie]

    similarities = movie_ratings.corrwith(target_movie_ratings)
    similarities = similarities.dropna()

    similarity_df = pd.DataFrame(
        similarities,
        columns=["similarity"],
    )

    popular_movies = movie_stats["rating_count"] >= minimum_rating_count

    recommendations = movie_stats[popular_movies].join(similarity_df)

    recommendations = recommendations.dropna(subset=["similarity"])

    if target_movie in recommendations.index:
        recommendations = recommendations.drop(index=target_movie)

    recommendations = recommendations.sort_values(
        by="similarity",
        ascending=False,
    )

    return recommendations.head(number_of_recommendations)


def main():
    # -----------------------------
    # 1. Load data
    # -----------------------------

    ratings = load_movielens_data()

    print("Merged ratings data:")
    print(ratings.head())
    print()

    # -----------------------------
    # 2. Build user-movie matrix
    # -----------------------------

    movie_ratings = build_user_movie_matrix(ratings)

    print("User-movie rating matrix:")
    print(movie_ratings.head())
    print()

    # -----------------------------
    # 3. Calculate movie statistics
    # -----------------------------

    movie_stats = calculate_movie_statistics(ratings)

    print("Movie statistics:")
    print(movie_stats.head())
    print()

    # -----------------------------
    # 4. Recommend movies similar to target movie
    # -----------------------------

    target_movie = "Star Wars (1977)"

    recommendations = recommend_similar_movies(
        movie_ratings=movie_ratings,
        movie_stats=movie_stats,
        target_movie=target_movie,
        minimum_rating_count=100,
        number_of_recommendations=15,
    )
    print("******************************************************")
    print(f"Movies similar to {target_movie}:")
    print(recommendations)
    print("******************************************************")


if __name__ == "__main__":
    main()
# Pearson correlation measures:
#   how strongly two variables move together in a linear way.
#   It first looks at each value compared to its own average: “Is this rating above or below this movie’s average?”
#   Then it checks whether the two movies tend to be above-average and below-average for the same users at the same time.
#   if users who rate `Star Wars` higher than average also tend to rate `Empire Strikes Back` higher than average,
#   the Pearson correlation becomes positive and close to `+1`.
#   If there is no consistent pattern, it is close to `0`.
#   If users who rate one movie high tend to rate the other low, it becomes negative, close to `-1`.
#   So it does not simply ask “are both movies highly rated?”;
#   it asks:
#       “do their rating patterns rise and fall together across the same users?”
