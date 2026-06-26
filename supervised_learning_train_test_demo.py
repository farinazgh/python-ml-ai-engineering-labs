import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


# Given page speed, can we predict purchase amount?
# X = pageSpeeds
# Y = purchaseAmount
# pageSpeeds is the input feature.
# purchaseAmount is the value we want to predict.
def main():

    # -----------------------------
    # 1. Generate fake data
    # faster page speed | lower load time  → higher purchase amount
    # slower page speed | higher load time → lower purchase amount
    # -----------------------------

    # makes the random numbers reproducible. always generate the same random numbers each time
    np.random.seed(2)

    #  creates 100 fake page speeds centered around 3.0.
    page_speeds = np.random.normal(3.0, 1.0, 100)

    # creates fake purchase amounts centered around 50, but divides by pageSpeeds.
    purchase_amounts = np.random.normal(50.0, 30.0, 100) / page_speeds

    # -----------------------------
    # 2. Plot all data
    # -----------------------------

    plt.scatter(page_speeds, purchase_amounts)
    plt.title("All Data")
    plt.xlabel("Page Speed")
    plt.ylabel("Purchase Amount")
    plt.show()

    # -----------------------------
    # 3. Split into train/test

    # -----------------------------

    # first 80 rows  → training data
    # last 20 rows   → testing data
    train_x = page_speeds[:80]
    test_x = page_speeds[80:]

    train_y = purchase_amounts[:80]
    test_y = purchase_amounts[80:]
    # We do not want a model that only memorizes the training data.
    # We want a model that generalizes to new data it has never seen.

    # -----------------------------
    # 4. Plot training data
    # -----------------------------

    plt.scatter(train_x, train_y)
    plt.title("Training Data")
    plt.xlabel("Page Speed")
    plt.ylabel("Purchase Amount")
    plt.show()

    # -----------------------------
    # 5. Plot test data
    # -----------------------------

    plt.scatter(test_x, test_y)
    plt.title("Test Data")
    plt.xlabel("Page Speed")
    plt.ylabel("Purchase Amount")
    plt.show()

    # -----------------------------
    # 6. Train polynomial regression model
    # -----------------------------

    # Fit an 8th-degree polynomial to the training data.
    # y = a₈x⁸ + a₇x⁷ + a₆x⁶ + a₅x⁵ + a₄x⁴ + a₃x³ + a₂x² + a₁x + a₀
    degree = 8

    model = np.poly1d(np.polyfit(train_x, train_y, degree))

    # -----------------------------
    # 7. Plot model against training data
    # -----------------------------

    xp = np.linspace(0, 7, 100)

    plt.scatter(train_x, train_y)
    plt.plot(xp, model(xp), color="red")
    plt.title("Polynomial Model on Training Data")
    plt.xlabel("Page Speed")
    plt.ylabel("Purchase Amount")
    plt.xlim(0, 7)
    plt.ylim(0, 200)
    plt.show()

    # -----------------------------
    # 8. Plot model against test data
    # -----------------------------

    plt.scatter(test_x, test_y)
    plt.plot(xp, model(xp), color="red")
    plt.title("Polynomial Model on Test Data")
    plt.xlabel("Page Speed")
    plt.ylabel("Purchase Amount")
    plt.xlim(0, 7)
    plt.ylim(0, 200)
    plt.show()

    # -----------------------------
    # 9. Evaluate model
    # -----------------------------

    # This checks performance on the data the model already learned from.
    # For this example, the test R² is approximately:

    # 0.30018
    # That means the model explains around 30% of the variation in the test data.

    # Training R² ≈ 0.64
    # Test R²     ≈ 0.30

    # This gap is important. It suggests:
    #
    # The model fits the training data better than the test data.
    #
    # That is normal, but if the gap becomes too large, we start suspecting overfitting.
    # training data = what the model studies
    # test data     = the exam
    # The model is not judged only by how well it remembers the training examples.
    # It is judged by how well it handles unseen examples.
    #
    test_r2 = r2_score(test_y, model(test_x))
    train_r2 = r2_score(train_y, model(train_x))

    print(f"Test R²: {test_r2}")
    print(f"Training R²: {train_r2}")


if __name__ == "__main__":
    main()

    # This is not unsupervised learning yet.
    #
    # This example is supervised learning because we have:
    #
    # input  → page speed
    # target → purchase amount
    #
    # The model is learning from known answers.
    #
    # Unsupervised learning would be more like:
    #
    # Here are customers. I will not tell you purchase labels. Find natural groups/clusters by yourself.
    #
    # But this notebook is definitely:
    #
    # supervised regression + train/test evaluation
    #
    # And the 8th-degree polynomial is intentionally a little dramatic — like giving the model a very flexible snake-like curve.
    # That is useful for teaching overfitting, because flexible models can sometimes look clever on training data but become unreliable on unseen data.
