import numpy as np
import matplotlib.pyplot as plt


def generate_normal_data(mean, std_dev, sample_size):
    return np.random.normal(mean, std_dev, sample_size)


def display_percentile_markers(values):
    percentile_20 = np.percentile(values, 20)
    percentile_50 = np.percentile(values, 50)
    percentile_90 = np.percentile(values, 90)

    plt.hist(values, bins=50)
    plt.axvline(
        percentile_20, linestyle="--", label=f"20th percentile = {percentile_20:.4f}"
    )
    plt.axvline(
        percentile_50, linestyle="--", label=f"50th percentile = {percentile_50:.4f}"
    )
    plt.axvline(
        percentile_90, linestyle="--", label=f"90th percentile = {percentile_90:.4f}"
    )

    plt.title("Percentiles\nRelative position inside the data")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


def compare_percentage_and_percentile(values, value):
    minimum = np.min(values)
    maximum = np.max(values)

    percentage_of_range = ((value - minimum) / (maximum - minimum)) * 100
    percentile_rank = (np.sum(values <= value) / len(values)) * 100

    print(f"Chosen value: {value:.4f}")
    print()
    print(f"Percentage of range: {percentage_of_range:.2f}%")
    print("Meaning: where this value sits between the minimum and maximum.")
    print()
    print(f"Percentile rank:     {percentile_rank:.2f}%")
    print("Meaning: how many data points are below this value.")


def main():
    mean = 0.0
    std_dev = 0.5
    sample_size = 10_000

    values = generate_normal_data(mean, std_dev, sample_size)

    display_percentile_markers(values)

    chosen_value = np.percentile(values, 90)
    compare_percentage_and_percentile(values, chosen_value)


if __name__ == "__main__":
    main()
