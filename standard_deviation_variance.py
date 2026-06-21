import numpy as np
import matplotlib.pyplot as plt


def main():
    incomes = np.random.normal(100.0, 50.0, 10_000)

    plt.hist(incomes, bins=50)
    plt.title("Income Distribution")
    plt.xlabel("Income")
    plt.ylabel("Frequency")
    plt.show()

    print(f"Standard deviation: {incomes.std():.2f}")
    print(f"Variance: {incomes.var():.2f}")


if __name__ == "__main__":
    main()
