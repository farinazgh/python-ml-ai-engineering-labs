import pandas as pd

# instead of:
#   People who liked Star Wars may also like X
#   Based on everything this specific user liked, recommend X


# load MovieLens ratings
# ↓
# load movie titles
# ↓
# merge ratings with titles
# ↓
# create user × movie rating matrix
# ↓
# calculate movie-to-movie similarity matrix
# ↓
# get one user's personal ratings
# ↓
# for every movie that user rated:
#     find similar movies
#     multiply similarity by user's rating
# ↓
# combine all recommendation scores
# ↓
# remove movies the user already rated
# ↓
# return top recommendations


def load_movielens_data():

    rating_columns = ["user_id", "movie_id", "rating"]
    ratings = pd.read_csv(
        "movie-data.data",
        sep="\t",
        names=rating_columns,
        usecols=range(3),
        encoding="ISO-8859-1",
    )

    movie_columns = ["movie_id", "title"]
    movies = pd.read_csv(
        "movie-data.item",
        sep="|",
        names=movie_columns,
        usecols=range(2),
        encoding="ISO-8859-1",
    )

    return pd.merge(movies, ratings, on="movie_id")


def build_user_movie_matrix(ratings):
    """
    Build a user-item matrix:
    - rows are users
    - columns are movies
    - values are ratings
    """

    return ratings.pivot_table(
        index="user_id",
        columns="title",
        values="rating",
    )


# Build the full movie-vs-movie using Pearson correlation matrix.
# This compares every movie with every other movie using user rating patterns.
# min_periods=100 keeps only correlations based on at least 100 shared ratings,
# which helps avoid unreliable similarities from too little data.
def build_movie_correlation_matrix(user_ratings):

    return user_ratings.corr(
        method="pearson",
        min_periods=100,
    )


# Get all ratings made by one specific user.
# This creates the user's personal taste profile: the movies they rated and the scores they gave.
# excluding movies they did not rate.
def get_user_ratings(user_ratings, user_id):

    if user_id not in user_ratings.index:
        raise ValueError(f"User ID {user_id} was not found in the ratings data.")

    return user_ratings.loc[user_id].dropna()


# Recommend movies for one user based on their whole rating history.
# For every movie the user rated, we find similar movies, multiply each similarity
# by the user's rating, then add all scores together.
# Finally, we remove movies the user has already rated.
def recommend_movies_for_user(correlation_matrix, user_movie_ratings):

    similarity_candidates = pd.Series(dtype=float)
    #
    for movie_title, user_rating in user_movie_ratings.items():
        print(f"Adding similarities for {movie_title}...")
        # removes missing values.
        similar_movies = correlation_matrix[movie_title].dropna()
        # This is the emotional intelligence of the recommender.
        #
        # similar_movies contains similarity scores.
        # But similarity alone is not enough. We also need to know:
        #
        # “How much did the user like the original movie?”
        #
        # If the user rated Star Wars as 5, then similar movies should get a strong boost.
        # The more the user liked a movie,
        # the more influence its similar movies should have.
        weighted_similarities = similar_movies.map(
            lambda similarity: similarity * user_rating
        )

        similarity_candidates = pd.concat(
            [similarity_candidates, weighted_similarities]
        )

    recommendation_scores = similarity_candidates.groupby(
        similarity_candidates.index
    ).sum()

    recommendation_scores.sort_values(
        inplace=True,
        ascending=False,
    )

    filtered_recommendations = recommendation_scores.drop(
        user_movie_ratings.index,
        errors="ignore",
    )

    return filtered_recommendations


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

    user_ratings = build_user_movie_matrix(ratings)

    print("User-movie rating matrix:")
    print(user_ratings.head())
    print()

    # -----------------------------
    # 3. Build movie correlation matrix
    # -----------------------------

    correlation_matrix = build_movie_correlation_matrix(user_ratings)

    print("Movie correlation matrix:")
    print(correlation_matrix.head())
    print()

    # -----------------------------
    # 4. Get one user's ratings
    # -----------------------------

    user_id = 0
    my_ratings = get_user_ratings(user_ratings, user_id)

    print(f"Ratings for user {user_id}:")
    print(my_ratings)
    print()

    # -----------------------------
    # 5. Generate recommendations
    # -----------------------------

    recommendations = recommend_movies_for_user(
        correlation_matrix=correlation_matrix,
        user_movie_ratings=my_ratings,
    )

    print("Top movie recommendations:")
    print(recommendations.head(10))


if __name__ == "__main__":
    main()
