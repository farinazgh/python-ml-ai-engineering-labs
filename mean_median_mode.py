import numpy as np
import matplotlib.pyplot as plt


def calculate_mean(incomes):
    mean = np.mean(incomes)
    print(f"Mean: {mean:.2f}")


def calculate_median(incomes):
    median = np.median(incomes)
    print(f"Median: {median:.2f}")


def add_outlier(incomes):
    return np.append(incomes, [1_000_000_000])


def display_histogram(incomes, title):
    plt.hist(incomes, 50)
    plt.title(title)
    plt.show()


def main():

    incomes = np.random.normal(27_000, 15_000, 10_000)

    print("Before outlier:")
    calculate_mean(incomes)
    calculate_median(incomes)
    display_histogram(incomes, "Income distribution before outlier")

    incomes_with_outlier = add_outlier(incomes)

    print("After outlier:")
    calculate_mean(incomes_with_outlier)
    calculate_median(incomes_with_outlier)
    display_histogram(incomes_with_outlier, "Income distribution after outlier")


if __name__ == "__main__":
    main()
