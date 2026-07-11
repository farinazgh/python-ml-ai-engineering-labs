from itertools import cycle

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

NUMBER_OF_COMPONENTS = 2
USE_WHITENING = True


# Can we take 4 flower measurements and compress them into 2 dimensions so we can visualize them?
# load Iris dataset
# ↓
# check number of samples and features
# ↓
# extract the feature matrix X
# ↓
# fit PCA on the 4D flower measurements
# ↓
# transform the data from 4D to 2D
# ↓
# print the PCA components
# ↓
# print how much variance the 2 components explain
# ↓
# plot the flowers in 2D
# ↓
# color the points by Iris species
def load_iris_dataset():
    """
    Load the Iris dataset from scikit-learn.
    """

    return load_iris()


def print_dataset_summary(iris):
    """
    Print basic information about the Iris dataset.
    """

    number_of_samples, number_of_features = iris.data.shape

    print("Number of samples:")
    print(number_of_samples)
    print()

    print("Number of features:")
    print(number_of_features)
    print()

    print("Target names:")
    print(list(iris.target_names))
    print()


#   sepal length   sepal width   petal length   petal width   species
#   -----------:   ----------:   -----------:   ----------:   ----------
#            5.1           3.5            1.4           0.2   setosa
#            7.0           3.2            4.7           1.4   versicolor
#            6.3           3.3            6.0           2.5   virginica


# After PCA, make the component values more standardized.
def apply_pca(features, number_of_components=2, whiten=True):
    """
    Fit PCA and transform the original features into PCA space.

    fit:
        Learns the principal component directions.

    transform:
        Projects the original data onto those new directions.
    """

    pca = PCA(
        n_components=number_of_components,
        whiten=whiten,
    )
    # Learn the PCA directions from the data.
    # Along which direction are the flowers spread out the most?
    # That becomes principal component 1.
    #
    # Along which next direction is there the most remaining variation?
    # That becomes principal component 2.
    transformed_features = pca.fit_transform(features)

    return pca, transformed_features


def print_pca_summary(pca):
    """
    Print PCA components and explained variance information.
    """

    print("PCA components:")
    print(pca.components_)
    print()

    print("Explained variance ratio:")
    print(pca.explained_variance_ratio_)
    print()

    print("Total explained variance ratio:")
    print(sum(pca.explained_variance_ratio_))
    print()


def plot_pca_results(transformed_features, targets, target_names):
    """
    Plot the first two PCA components and color points by species.

    PCA itself is unsupervised.
    The target labels are used only for coloring the visualization.
    """

    colors = cycle(["r", "g", "b"])
    target_ids = range(len(target_names))

    plt.figure()

    for target_id, color, label in zip(target_ids, colors, target_names):
        plt.scatter(
            transformed_features[targets == target_id, 0],
            transformed_features[targets == target_id, 1],
            c=color,
            label=label,
        )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Iris Dataset Projected with PCA")
    plt.legend()
    plt.show()


def main():
    # -----------------------------
    # 1. Load data
    # -----------------------------

    iris = load_iris_dataset()

    print_dataset_summary(iris)

    # -----------------------------
    # 2. Apply PCA
    # -----------------------------

    features = iris.data

    pca, transformed_features = apply_pca(
        features=features,
        number_of_components=NUMBER_OF_COMPONENTS,
        whiten=USE_WHITENING,
    )

    # -----------------------------
    # 3. Print PCA information
    # -----------------------------

    print_pca_summary(pca)

    # -----------------------------
    # 4. Plot PCA result
    # -----------------------------

    plot_pca_results(
        transformed_features=transformed_features,
        targets=iris.target,
        target_names=iris.target_names,
    )


if __name__ == "__main__":
    main()
