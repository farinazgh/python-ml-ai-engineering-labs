import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Is there a relationship between page speed and purchase amount?
# Page Speed ↑
# Purchase Amount ↓
# Slower website → customers spend less money


def generate_sample_data():
    np.random.seed(2)
    # Generate 1000 page load times
    page_speeds = np.random.normal(loc=3.0, scale=1.0, size=1000)
    # customer budget / page speed
    # That means slower pages produce smaller purchases.
    # But because division is involved, the shape is curved
    purchase_amounts = np.random.normal(loc=50.0, scale=10.0, size=1000) / page_speeds

    return page_speeds, purchase_amounts


def display_scatter_plot(page_speeds, purchase_amounts):
    plt.scatter(page_speeds, purchase_amounts)

    plt.title("Page Speed vs Purchase Amount\nRaw Non-Linear Data")
    plt.xlabel("Page Speed (seconds)")
    plt.ylabel("Purchase Amount ($)")

    plt.show()


# Allow the line to bend more than a simple straight line.
#
# But be careful: A higher degree can fit the training data better, but may overfit.
#
# Linear regression draws a straight road. Polynomial regression allows the road to curve.


def train_polynomial_regression(x, y, degree):
    # creates a 4th-degree polynomial model
    # y = ax⁴ + bx³ + cx² + dx + e
    coefficients = np.polyfit(x, y, degree)
    model = np.poly1d(coefficients)

    return model


def display_polynomial_regression_line(x, y, model):
    xp = np.linspace(0, 7, 100)

    plt.scatter(x, y, label="Observed Data")
    plt.plot(xp, model(xp), label="Polynomial Regression Line")

    plt.title(
        "Polynomial Regression\nCurved relationship between page speed and purchase amount"
    )
    plt.xlabel("Page Speed (seconds)")
    plt.ylabel("Purchase Amount ($)")
    plt.legend()

    plt.show()


# How much of the variation in purchase amount is explained by the model?
# 1.0 = better fit
# 0.0 = weak fit
def print_model_score(y, predicted_y):
    r2 = r2_score(y, predicted_y)

    print(f"R-squared: {r2:.4f}")


def main():
    degree = 4

    page_speeds, purchase_amounts = generate_sample_data()

    x = np.array(page_speeds)
    y = np.array(purchase_amounts)

    display_scatter_plot(x, y)

    model = train_polynomial_regression(x, y, degree)

    predicted_y = model(x)

    display_polynomial_regression_line(x, y, model)

    print_model_score(y, predicted_y)


if __name__ == "__main__":
    main()
