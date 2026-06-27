import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

# N = total number of people/data points
# k = number of clusters/groups

# create fake people with income and age
# ↓
# scale/normalize income and age
# ↓
# ask K-Means to find 5 groups
# ↓
# print the assigned cluster labels
# ↓
# plot the original data colored by cluster label


def create_clustered_data(number_of_points, number_of_clusters):

    np.random.seed(10)

    points_per_cluster = float(number_of_points) / number_of_clusters
    people_income_age = []

    for _ in range(number_of_clusters):
        income_centroid = np.random.uniform(20_000.0, 200_000.0)
        age_centroid = np.random.uniform(20.0, 70.0)
        # For every cluster, the code randomly chooses a center.
        # This creates the actual people inside each cluster.
        #
        # Since pointsPerCluster = 20, this loop creates 20 people per cluster.
        for _ in range(int(points_per_cluster)):
            income = np.random.normal(income_centroid, 10_000.0)
            age = np.random.normal(age_centroid, 2.0)

            people_income_age.append([income, age])

    return np.array(people_income_age)


def main():
    # -----------------------------
    # 1. Create fake clustered data
    # -----------------------------

    # Create 100 fake people distributed around 5 hidden groups.
    data = create_clustered_data(number_of_points=100, number_of_clusters=5)

    # -----------------------------
    # 2. Create K-Means model
    # -----------------------------

    model = KMeans(n_clusters=5, random_state=10)

    # -----------------------------
    # 3. Scale data and fit model
    # -----------------------------
    # normalizes the two columns: income and age.
    # Because income and age live on completely different scales: income: 20,000 to 200,000; age: 20 to 70
    #
    # Without scaling, income has much bigger numbers, so it would dominate the distance calculation.
    # K-Means uses distance. So without scaling, the algorithm may think:
    # income difference matters massively
    # age difference barely matters
    # After scaling, each feature is roughly transformed to:
    # mean ≈ 0
    # standard deviation ≈ 1
    # So income and age can both influence clustering. Let income and age speak in the same numeric volume.
    #
    scaled_data = scale(data)

    # trains the K-Means model which means: Find cluster centers and assign each point to one cluster.
    model.fit(scaled_data)

    # -----------------------------
    # 4. Print cluster labels
    # -----------------------------

    print("Cluster labels:")
    print(model.labels_)

    # -----------------------------
    # 5. Visualize clusters
    # -----------------------------

    plt.figure(figsize=(8, 6))
    plt.scatter(
        x=data[:, 0],
        y=data[:, 1],
        c=model.labels_.astype(float),
    )

    plt.title("K-Means Clustering: Income vs Age")
    plt.xlabel("Income")
    plt.ylabel("Age")
    plt.show()


# This empty list will store all fake people.
if __name__ == "__main__":
    main()
