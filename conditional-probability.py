from numpy import random


def simulate_customers(number_of_customers):
    print("simulate_customers")
    # The purpose of this line is to make sure that we get consistent results every time we run this code.
    random.seed(0)

    totals = {20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0}

    purchases = {20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0}

    total_purchases = 0
    # In Python, _ means: I don’t really care what that value is on each iteration.
    for _ in range(number_of_customers):

        age_decade = random.choice([20, 30, 40, 50, 60, 70])

        # Just for this example, we assume a bias: the older you are, the more likely you are to buy.
        purchase_probability = age_decade / 100.0

        totals[age_decade] += 1

        if random.random() < purchase_probability:
            purchases[age_decade] += 1
            total_purchases += 1

    return totals, purchases, total_purchases


def calculate_conditional_probability(purchases, totals, age_decade):
    return purchases[age_decade] / totals[age_decade]


def calculate_age_probability(totals, age_decade, population_size):
    return totals[age_decade] / population_size


def calculate_purchase_probability(total_purchases, population_size):
    return total_purchases / population_size


def calculate_joint_probability(purchases, age_decade, population_size):
    return purchases[age_decade] / population_size


def print_results(totals, purchases, total_purchases, population_size):

    age_decade = 30

    p_purchase_given_30s = calculate_conditional_probability(
        purchases, totals, age_decade
    )

    p_30s = calculate_age_probability(totals, age_decade, population_size)

    p_purchase = calculate_purchase_probability(total_purchases, population_size)

    # probability of something happening given something else happened
    # P(A, B)
    # P(B | A)
    # P(B | A) = P(A, B) / P(A)

    # P(E | F) ≈ P(E)
    p_30s_and_purchase = calculate_joint_probability(
        purchases, age_decade, population_size
    )

    print(f"P(Purchase | 30s): {p_purchase_given_30s:.4f}")
    print(f"P(30s): {p_30s:.4f}")
    print(f"P(Purchase): {p_purchase:.4f}")
    print(f"P(30s AND Purchase): {p_30s_and_purchase:.4f}")
    print(f"P(30s) × P(Purchase): {(p_30s * p_purchase):.4f}")


def main():

    population_size = 100_000

    totals, purchases, total_purchases = simulate_customers(population_size)

    print_results(totals, purchases, total_purchases, population_size)


if __name__ == "__main__":
    main()
