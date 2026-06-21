import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def calculate_mean(incomes):
    mean = np.mean(incomes)
    print(f"Mean: {mean:.2f}")


def calculate_median(incomes):
    median = np.median(incomes)
    print(f"Median: {median:.2f}")


def calculate_mode(ages):
    mode_result = stats.mode(ages)

    mode_age = mode_result.mode
    mode_count = mode_result.count

    print(f"Mode age: {mode_age}")
    print(f"Appears: {mode_count} times")


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

    ages = np.random.randint(18, high=90, size=500)
    calculate_mode(ages)
    display_histogram(ages, "Age distribution")


if __name__ == "__main__":
    main()
# Practice notes:
# - Avoided hidden global state by passing `incomes` and `ages` explicitly into functions.
# - Fixed the outlier bug: `add_outlier()` now receives an array instead of using None.
# - Separated responsibilities: calculation functions calculate, display function plots.
# - SciPy `stats.mode()` returns a ModeResult object, not a plain number.
# - Improved readability with clearer names like `mean_income`, `mode_result`, and `incomes_with_outlier`.
