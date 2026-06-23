import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


def load_car_data():
    return pd.read_excel("C:\\polaris\\code\\python-ml-ai-engineering-labs\\cars.xls")


# group by mileage range
# 0–10,000
# 10,000–20,000
# 20,000–30,000
# 30,000–40,000


def display_average_price_by_mileage(df):
    mileage_and_price = df[["Mileage", "Price"]]

    bins = np.arange(0, 50_000, 10_000)

    groups = mileage_and_price.groupby(
        pd.cut(mileage_and_price["Mileage"], bins)
    ).mean()

    print(groups.head())

    groups["Price"].plot.line()
    plt.title("Average Car Price by Mileage\nHigher mileage usually means lower price")
    plt.xlabel("Mileage Range")
    plt.ylabel("Average Price")
    plt.show()


# Price = b0 + b1(Mileage) + b2(Cylinder) + b3(Doors)
def prepare_regression_data(df):
    scaler = StandardScaler()

    X = df[["Mileage", "Cylinder", "Doors"]].copy()
    y = df["Price"]

    X[["Mileage", "Cylinder", "Doors"]] = scaler.fit_transform(
        X[["Mileage", "Cylinder", "Doors"]].values
    )

    X = sm.add_constant(X)

    return X, y, scaler


def train_regression_model(X, y):
    model = sm.OLS(y, X).fit()

    print(model.summary())

    return model


# e.g. What price do we predict for a car with:
# 45,000 mileage
# 8 cylinders
# 4 doors?
def predict_car_price(model, scaler, mileage, cylinders, doors):
    scaled_values = scaler.transform([[mileage, cylinders, doors]])

    scaled_values = np.insert(scaled_values[0], 0, 1)

    predicted_price = model.predict(scaled_values)

    return predicted_price[0]


def main():
    df = load_car_data()

    display_average_price_by_mileage(df)

    X, y, scaler = prepare_regression_data(df)

    model = train_regression_model(X, y)

    predicted_price = predict_car_price(
        model, scaler, mileage=45_000, cylinders=8, doors=4
    )

    print(f"Predicted price: ${predicted_price:.2f}")


if __name__ == "__main__":
    main()
