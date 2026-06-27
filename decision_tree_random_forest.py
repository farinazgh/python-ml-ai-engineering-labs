from io import StringIO

import pandas as pd
import pydot
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text

# supervised classification

# load hiring data
# ↓
# convert categorical text to numbers
# ↓
# separate features X from target y
# ↓
# train a decision tree
# ↓
# print the decision tree rules
# ↓
# train a random forest
# ↓
# predict hiring decisions for new candidates


def load_and_prepare_data(input_file):

    df = pd.read_csv(input_file, header=0)

    yes_no_mapping = {"Y": 1, "N": 0}

    df["Hired"] = df["Hired"].map(yes_no_mapping)
    df["Employed?"] = df["Employed?"].map(yes_no_mapping)
    df["Top-tier school"] = df["Top-tier school"].map(yes_no_mapping)
    df["Interned"] = df["Interned"].map(yes_no_mapping)

    education_mapping = {"BS": 0, "MS": 1, "PhD": 2}
    df["Level of Education"] = df["Level of Education"].map(education_mapping)

    return df


def main():
    # -----------------------------
    # 1. Load and prepare data
    # -----------------------------

    input_file = "PastHires.csv"
    df = load_and_prepare_data(input_file)

    print("Prepared data:")
    print(df.head())
    print()

    # -----------------------------
    # 2. Select features and target
    # -----------------------------

    # This takes the first 6 columns of the DataFrame and uses them as input features. hired, employed, etc.
    features = list(df.columns[:6])
    # This is the classic supervised learning split:
    # X = input features
    #   experience
    #   employment status
    #   previous employers
    #   education
    #   school type
    #   internship
    # y = target/output
    #   hired or not hired
    X = df[features]
    y = df["Hired"]

    print("Features:")
    print(features)
    print()

    # -----------------------------
    # 3. Train Decision Tree model
    # -----------------------------

    # creates an empty decision tree
    decision_tree = tree.DecisionTreeClassifier(random_state=42)

    # Learning happens here.
    # fit: The decision tree tries to find the best first question:
    #      If I split the dataset using this question, do the resulting groups become cleaner?
    #      A “clean” group means most rows inside that group have the same label.
    decision_tree.fit(X, y)

    # -----------------------------
    # 4. Print Decision Tree rules
    # -----------------------------

    print("Decision Tree rules:")
    print(export_text(decision_tree, feature_names=features))

    # -----------------------------
    # 5. Export Decision Tree as PNG
    # -----------------------------

    dot_data = StringIO()

    tree.export_graphviz(
        decision_tree,
        out_file=dot_data,
        feature_names=features,
        class_names=["Not Hired", "Hired"],
        filled=True,
        rounded=True,
        special_characters=True,
    )

    graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]
    graph.write_png("decision_tree.png")

    print("Decision tree visualization saved to decision_tree.png")

    # -----------------------------
    # 6. Train Random Forest model
    # -----------------------------
    # A random forest is an ensemble model.
    random_forest = RandomForestClassifier(
        n_estimators=10,  # build 10 decision trees: Each tree gives a prediction, then the forest takes the majority vote.
        random_state=42,
    )

    random_forest.fit(X, y)

    # -----------------------------
    # 7. Predict new candidates
    # -----------------------------

    employed_candidate = pd.DataFrame(
        [[10, 1, 4, 0, 0, 0]],
        columns=features,
    )

    unemployed_candidate = pd.DataFrame(
        [[10, 0, 4, 0, 0, 0]],
        columns=features,
    )

    employed_prediction = random_forest.predict(employed_candidate)
    unemployed_prediction = random_forest.predict(unemployed_candidate)

    print("Prediction for employed 10-year veteran:")
    print(employed_prediction)

    print("Prediction for unemployed 10-year veteran:")
    print(unemployed_prediction)


if __name__ == "__main__":
    main()
