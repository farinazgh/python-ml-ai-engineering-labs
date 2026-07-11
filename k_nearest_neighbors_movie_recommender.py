import numpy as np
import pandas as pd
from scipy import spatial

TARGET_MOVIE_ID = 1
NUMBER_OF_NEIGHBORS = 10


# Movies have properties
# ↓
# Compare movie properties
# ↓
# Find movies closest to a target movie

# Given one movie, find other movies that are similar by genre and popularity.


# load MovieLens ratings
# ↓
# calculate movie popularity and average rating
# ↓
# normalize movie popularity
# ↓
# load movie titles and genre vectors
# ↓
# create a movie dictionary
# ↓
# define a distance function between two movies
# ↓
# distance = genre distance + popularity distance
# ↓
# find the K nearest movies to a target movie
# ↓
# print those neighbors and their average ratings
# ↓
# calculate the average rating of the neighbors
def load_ratings():
    """
    Load MovieLens 100K ratings data.

    The returned DataFrame contains:
    - user_id
    - movie_id
    - rating
    """

    rating_columns = ["user_id", "movie_id", "rating"]

    return pd.read_csv(
        "movie-data.data",
        sep="\t",
        names=rating_columns,
        usecols=range(3),
        encoding="ISO-8859-1",
    )


def calculate_movie_properties(ratings):
    """
    rating_count tells us how popular a movie is: how many people rated it
    average_rating tells us how highly rated it is:  how much people liked it on average
    """
    movie_properties = ratings.groupby("movie_id")["rating"].agg(["size", "mean"])

    movie_properties.columns = [
        "rating_count",
        "average_rating",
    ]
    #   movie_id   rating size   rating mean
    #   -------:   ----------:   ----------:
    #          1           452          3.88
    #          2           131          3.21
    #          3            90          3.03
    return movie_properties


def normalize_rating_counts(movie_properties):
    """
    Normalize rating counts to the range 0 to 1 using min-max normalization.
    """
    rating_counts = movie_properties["rating_count"]

    minimum_count = rating_counts.min()
    maximum_count = rating_counts.max()

    if maximum_count == minimum_count:
        return pd.Series(
            data=0.0,
            index=rating_counts.index,
            name="normalized_rating_count",
        )

    normalized_rating_counts = (rating_counts - minimum_count) / (
        maximum_count - minimum_count
    )

    normalized_rating_counts.name = "normalized_rating_count"
    # By normalizing popularity to 0–1, we make it comparable to the genre distance.
    # This is a very important ML idea: When combining different features, their scales matter.
    # Because distance-based algorithms are sensitive to magnitude.
    return normalized_rating_counts


def load_movie_dictionary(
    movie_properties,
    normalized_rating_counts,
):
    """
    Each movie is represented as:
    - title
    - genre vector: [0, 0, 0, 1, 1, 0, 0, ...] movie can be rom-com and comedy at thesame time so 1,1
    - normalized rating count
    - average rating
     index 0 = title
     index 1 = genres
     index 2 = normalized popularity
     index 3 = average rating
    """

    movie_dict = {}

    with open("movie-data.item", encoding="ISO-8859-1") as file:
        for line in file:
            fields = line.rstrip("\n").split("|")

            movie_id = int(fields[0])
            title = fields[1]

            genre_values = fields[5:25]
            genres = np.array([int(value) for value in genre_values])

            movie_dict[movie_id] = {
                "title": title,
                "genres": genres,
                "normalized_rating_count": normalized_rating_counts.loc[movie_id],
                "average_rating": movie_properties.loc[movie_id, "average_rating"],
            }

    return movie_dict


def compute_distance(movie_a, movie_b):
    """
    Compute distance between two movies.

    Distance is based on:
    - genre cosine distance
    - normalized popularity distance

    Lower distance means the movies are more similar.
    """
    # This compares the genre vectors of two movies.
    # If the genre vectors are very similar, the cosine distance is small.
    #
    # If the genre vectors are very different, the cosine distance is larger.
    # cosine distance = 1 - cosine similarity
    genre_distance = spatial.distance.cosine(
        movie_a["genres"],
        movie_b["genres"],
    )

    popularity_distance = abs(
        movie_a["normalized_rating_count"] - movie_b["normalized_rating_count"]
    )

    return genre_distance + popularity_distance


def get_neighbors(movie_dict, target_movie_id, number_of_neighbors):
    """
    Find the nearest movies to a target movie.
    """

    if target_movie_id not in movie_dict:
        raise ValueError(f"Movie ID {target_movie_id} was not found.")

    distances = []

    target_movie = movie_dict[target_movie_id]

    for movie_id, candidate_movie in movie_dict.items():
        if movie_id == target_movie_id:
            continue

        distance = compute_distance(
            target_movie,
            candidate_movie,
        )

        distances.append((movie_id, distance))
    # sort by the second item in each tuple: distance
    distances.sort(key=lambda movie_distance: movie_distance[1])

    nearest_neighbors = [
        movie_id for movie_id, distance in distances[:number_of_neighbors]
    ]
    # Taking the top K neighbors
    return nearest_neighbors


def calculate_average_neighbor_rating(movie_dict, neighbors):
    """
    Calculate the average rating of the nearest neighbor movies.
    """

    neighbor_ratings = [
        movie_dict[movie_id]["average_rating"] for movie_id in neighbors
    ]

    return np.mean(neighbor_ratings)


def main():
    # -----------------------------
    # 1. Load ratings
    # -----------------------------

    ratings = load_ratings()

    print("Ratings data:")
    print(ratings.head())
    print()

    # -----------------------------
    # 2. Calculate movie statistics
    # -----------------------------

    movie_properties = calculate_movie_properties(ratings)

    print("Movie properties:")
    print(movie_properties.head())
    print()

    # -----------------------------
    # 3. Normalize popularity
    # -----------------------------

    normalized_rating_counts = normalize_rating_counts(movie_properties)

    print("Normalized rating counts:")
    print(normalized_rating_counts.head())
    print()

    # -----------------------------
    # 4. Load movie metadata
    # -----------------------------

    movie_dict = load_movie_dictionary(
        movie_properties=movie_properties,
        normalized_rating_counts=normalized_rating_counts,
    )

    # -----------------------------
    # 5. Find nearest neighbors
    # -----------------------------

    target_movie = movie_dict[TARGET_MOVIE_ID]

    print(f"Target movie: {target_movie['title']}")
    print()

    neighbors = get_neighbors(
        movie_dict=movie_dict,
        target_movie_id=TARGET_MOVIE_ID,
        number_of_neighbors=NUMBER_OF_NEIGHBORS,
    )

    print(f"Top {NUMBER_OF_NEIGHBORS} nearest movies:")
    print()

    for neighbor_id in neighbors:
        neighbor = movie_dict[neighbor_id]

        print(f"{neighbor['title']} " f"{neighbor['average_rating']:.2f}")

    # -----------------------------
    # 6. Average neighbor rating
    # -----------------------------

    average_neighbor_rating = calculate_average_neighbor_rating(
        movie_dict=movie_dict,
        neighbors=neighbors,
    )

    print()
    print(f"Average rating of nearest neighbors: " f"{average_neighbor_rating:.2f}")


if __name__ == "__main__":
    main()
