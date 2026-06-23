import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Is there a relationship between page speed and purchase amount?
# Page Speed ↑
# Purchase Amount ↓
# Slower website → customers spend less money


def generate_sample_data():
    # Generate 1000 page load times
    page_speeds = np.random.normal(loc=3.0, scale=1.0, size=1000)
    # Purchase Amount = 100 - 3 × Page Speed
    purchase_amounts = 100 - (page_speeds + np.random.normal(0, 0.1, 1000)) * 3

    return page_speeds, purchase_amounts


def display_scatter_plot(page_speeds, purchase_amounts):

    plt.scatter(page_speeds, purchase_amounts)

    plt.title("Page Speed vs Purchase Amount\nRaw Data")

    plt.xlabel("Page Speed (seconds)")

    plt.ylabel("Purchase Amount ($)")

    plt.show()


def train_linear_regression(page_speeds, purchase_amounts):
    # find Best Fit Line
    return stats.linregress(page_speeds, purchase_amounts)


def predict(page_speed, slope, intercept):

    return slope * page_speed + intercept


def display_regression_line(page_speeds, purchase_amounts, slope, intercept):

    predicted_values = predict(page_speeds, slope, intercept)

    plt.scatter(page_speeds, purchase_amounts, label="Observed Data")

    plt.plot(page_speeds, predicted_values, label="Regression Line")

    plt.title("Linear Regression\nPage Speed Predicts Purchase Amount")

    plt.xlabel("Page Speed (seconds)")

    plt.ylabel("Purchase Amount ($)")

    plt.legend()

    plt.show()


def print_statistics(slope, intercept, r_value, p_value, std_err):

    print(f"Slope:      {slope:.4f}")
    print(f"Intercept:  {intercept:.4f}")
    print(f"R-value:    {r_value:.4f}")
    print(f"P-value:    {p_value:.10f}")
    print(f"Std Error:  {std_err:.4f}")


def main():

    page_speeds, purchase_amounts = generate_sample_data()

    display_scatter_plot(page_speeds, purchase_amounts)

    slope, intercept, r_value, p_value, std_err = train_linear_regression(
        page_speeds, purchase_amounts
    )
    # -1 → perfect negative relationship
    #  0 → no relationship
    # +1 → perfect positive relationship
    print_statistics(slope, intercept, r_value, p_value, std_err)

    display_regression_line(page_speeds, purchase_amounts, slope, intercept)


if __name__ == "__main__":
    main()
