import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom, poisson, expon, bernoulli, lognorm


# 1. Normal distribution
def plot_normal():
    x = np.linspace(-4, 4, 1000)
    y = norm.pdf(x, loc=0, scale=1)

    plt.plot(x, y)
    plt.title("Normal Distribution\nMost values near the average, extremes are rare")
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.show()


# Binomial distribution = number of successes in a fixed number of yes/no attempts.
def plot_binomial():
    n = 10
    p = 0.5
    x = np.arange(0, n + 1)
    y = binom.pmf(x, n, p)

    plt.bar(x, y)
    plt.title("Binomial Distribution\nNumber of successes in fixed yes/no attempts")
    plt.xlabel("Number of Successes")
    plt.ylabel("Probability")
    plt.show()


# 3. Poisson distribution
def plot_poisson():
    mu = 5
    x = np.arange(0, 16)
    y = poisson.pmf(x, mu)

    plt.bar(x, y)
    plt.title("Poisson Distribution\nNumber of events in a fixed time/window")
    plt.xlabel("Number of Events")
    plt.ylabel("Probability")
    plt.show()


# 4. Uniform distribution
def plot_uniform():
    values = np.random.uniform(0, 10, 100_000)

    plt.hist(values, bins=50)
    plt.title("Uniform Distribution\nEvery value has equal chance")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()


# 5. Exponential distribution
def plot_exponential():
    x = np.linspace(0, 10, 1000)
    y = expon.pdf(x, scale=1)

    plt.plot(x, y)
    plt.title("Exponential Distribution\nWaiting time until the next event")
    plt.xlabel("Waiting Time")
    plt.ylabel("Probability Density")
    plt.show()


# 6. Bernoulli distribution
def plot_bernoulli():
    p = 0.7
    x = [0, 1]
    y = bernoulli.pmf(x, p)

    plt.bar(x, y)
    plt.title("Bernoulli Distribution\nOne yes/no event")
    plt.xlabel("Outcome: 0 = No, 1 = Yes")
    plt.ylabel("Probability")
    plt.xticks([0, 1])
    plt.show()


# 7. Skewed distribution
def plot_skewed_income():
    values = np.random.exponential(scale=30_000, size=100_000)

    plt.hist(values, bins=80)
    plt.title("Right-Skewed Distribution\nMost values are small, few are very large")
    plt.xlabel("Income-like Value")
    plt.ylabel("Frequency")
    plt.show()


# 8. Log-normal distribution
def plot_lognormal():
    values = np.random.lognormal(mean=2.5, sigma=1.0, size=100_000)

    plt.hist(values, bins=100)
    plt.title("Log-Normal Distribution\nMany small values, few gigantic values")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()


# Bonus: compare normal vs log-normal
def compare_normal_and_lognormal():
    normal_values = np.random.normal(loc=50, scale=10, size=100_000)
    lognormal_values = np.random.lognormal(mean=3.5, sigma=0.7, size=100_000)

    plt.hist(normal_values, bins=80, alpha=0.6, label="Normal")
    plt.hist(lognormal_values, bins=80, alpha=0.6, label="Log-normal")
    plt.title("Normal vs Log-normal\nSymmetric world vs heavy-tailed world")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


def main():
    plot_normal()
    plot_binomial()
    plot_poisson()
    plot_uniform()
    plot_exponential()
    plot_bernoulli()
    plot_skewed_income()
    plot_lognormal()
    compare_normal_and_lognormal()


if __name__ == "__main__":
    main()
