import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom, expon, norm, poisson


def plot_uniform():
    values = np.random.uniform(-10.0, 10.0, 100_000)

    plt.hist(values, bins=50)
    plt.title("Uniform Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()


def plot_normal():
    x = np.arange(-3, 3, 0.001)

    plt.plot(x, norm.pdf(x))
    plt.title("Normal / Gaussian Distribution")
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.show()


def plot_binomial():
    n = 10
    p = 0.5
    x = np.arange(0, n + 1)

    plt.bar(x, binom.pmf(x, n, p))
    plt.title("Binomial Distribution")
    plt.xlabel("Number of Successes")
    plt.ylabel("Probability")
    plt.show()


def plot_exponential():
    x = np.arange(0, 10, 0.001)

    plt.plot(x, expon.pdf(x))
    plt.title("Exponential Distribution")
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.show()


def plot_poisson():
    mu = 500
    x = np.arange(400, 601)

    plt.bar(x, poisson.pmf(x, mu))
    plt.title("Poisson Distribution")
    plt.xlabel("Number of Events")
    plt.ylabel("Probability")
    plt.show()


def plot_random_normal_histogram():
    mu = 5.0
    sigma = 2.0
    values = np.random.normal(mu, sigma, 10_000)

    plt.hist(values, bins=50)
    plt.title("Random Normal Values Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()


def main():
    plot_uniform()
    plot_normal()
    plot_binomial()
    plot_exponential()
    plot_poisson()
    plot_random_normal_histogram()


if __name__ == "__main__":
    main()
