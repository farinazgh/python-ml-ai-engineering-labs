from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

## Given measurements of an iris flower, can we predict which iris species it belongs to? ##
# load Iris dataset
# ↓
# inspect number of samples, features, and classes
# ↓
# split data into train/test sets
# ↓
# convert data into XGBoost’s DMatrix format
# ↓
# define XGBoost model parameters
# ↓
# train the model
# ↓
# predict flower classes on test data
# ↓
# measure accuracy
def main():
    # -----------------------------
    # 1. Load Iris dataset
    # -----------------------------

    iris = load_iris()

    num_samples, num_features = iris.data.shape

    print(f"Number of samples: {num_samples}")
    print(f"Number of features: {num_features}")
    print(f"Target names: {list(iris.target_names)}")
    print()

    # -----------------------------
    # 2. Split data into train/test
    # -----------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=0,
    )

    # -----------------------------
    # 3. Convert data into XGBoost DMatrix format
    # -----------------------------

    train = xgb.DMatrix(X_train, label=y_train)
    test = xgb.DMatrix(X_test, label=y_test)

    # -----------------------------
    # 4. Define XGBoost parameters
    # -----------------------------
    # eta = step size
    #
    # Like walking down a mountain:
    #
    # small eta → small careful steps
    # large eta → bigger aggressive steps
    params = {
        "max_depth": 4,
        "eta": 0.3,
        "objective": "multi:softmax",
        "num_class": 3,
    }
    # Build 10 rounds of boosted trees.
    num_boost_round = 10

    # -----------------------------
    # 5. Train XGBoost model
    # -----------------------------

    model = xgb.train(params, train, num_boost_round)

    # -----------------------------
    # 6. Make predictions
    # -----------------------------

    predictions = model.predict(test)

    print("Predictions:")
    print(predictions)
    print()

    # -----------------------------
    # 7. Evaluate model accuracy
    # -----------------------------
    # accuracy_score compares:
    #
    # real labels → y_test
    # predicted labels → predictions
    #
    # Accuracy means:
    #
    # number of correct predictions divided by total predictions
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
