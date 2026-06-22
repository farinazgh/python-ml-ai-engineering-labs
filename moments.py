import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def generate_normal_data(mean, std_dev, sample_size):
    return np.random.normal(mean, std_dev, sample_size)


def display_histogram(values):
    plt.hist(values, bins=50)
    plt.title(
        "Normal Distribution Histogram\nMost values near the average, extremes are rare"
    )
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()


def display_mean_marker(values):
    mean = np.mean(values)

    plt.hist(values, bins=50)
    plt.axvline(mean, linestyle="--", label=f"Mean = {mean:.4f}")
    plt.title("Mean of Normal Distribution\nCenter point of the data")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


def display_variance_example(values):
    mean = np.mean(values)
    std_dev = np.std(values)

    plt.hist(values, bins=50)
    plt.axvline(mean, linestyle="--", label=f"Mean = {mean:.4f}")
    plt.axvline(
        mean - std_dev, linestyle=":", label=f"-1 Std Dev = {mean - std_dev:.4f}"
    )
    plt.axvline(
        mean + std_dev, linestyle=":", label=f"+1 Std Dev = {mean + std_dev:.4f}"
    )

    plt.title("Variance / Standard Deviation\nHow much values spread around the mean")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


def print_statistics(values):
    print(f"Mean:      {np.mean(values):.4f}")
    print(f"Variance:  {np.var(values):.4f}")
    print(f"Skewness:  {stats.skew(values):.4f}")
    print(f"Kurtosis:  {stats.kurtosis(values):.4f}")


def main():
    mean = 0.0
    std_dev = 0.5
    sample_size = 10_000

    values = generate_normal_data(mean, std_dev, sample_size)

    display_histogram(values)
    display_mean_marker(values)
    display_variance_example(values)
    print_statistics(values)


if __name__ == "__main__":
    main()
