import io
import os

import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Text
# ↓
# Vectorizer
# ↓
# Numbers
# ↓
# Classifier
# ↓
# Prediction

# Text       → email body
# Vectorizer → CountVectorizer
# Numbers    → word counts
# Classifier → MultinomialNB
# Prediction → spam or ham


def read_files(directory_path):

    for root, dirnames, filenames in os.walk(directory_path):
        for filename in filenames:
            file_path = os.path.join(root, filename)

            in_body = False
            lines = []

            with io.open(file_path, "r", encoding="latin1") as file:
                for line in file:
                    if in_body:
                        lines.append(line)
                    elif line == "\n":
                        in_body = True

            message = "\n".join(lines)
            yield file_path, message


def dataframe_from_directory(directory_path, classification):

    rows = []
    index = []

    for filename, message in read_files(directory_path):
        rows.append({"message": message, "class": classification})
        index.append(filename)

    return DataFrame(rows, index=index)


def main():
    # -----------------------------
    # 1. Load email data
    # -----------------------------

    data = DataFrame({"message": [], "class": []})

    spam_data = dataframe_from_directory("emails/spam", "spam")
    ham_data = dataframe_from_directory("emails/ham", "ham")

    data = pd.concat([data, spam_data])
    data = pd.concat([data, ham_data])

    print(f"Loaded {len(data)} emails.")
    print(data["class"].value_counts())

    # -----------------------------
    # 2. Convert text into word counts
    # -----------------------------

    vectorizer = CountVectorizer()
    counts = vectorizer.fit_transform(data["message"].values)

    # -----------------------------
    # 3. Train Naive Bayes classifier
    # -----------------------------

    classifier = MultinomialNB()
    targets = data["class"].values

    classifier.fit(counts, targets)

    # -----------------------------
    # 4. Predict new examples
    # -----------------------------

    examples = [
        "Free Viagra now!!!",
        "Hi Bob, how about a game of golf tomorrow?",
    ]

    example_counts = vectorizer.transform(examples)
    predictions = classifier.predict(example_counts)

    # -----------------------------
    # 5. Print predictions
    # -----------------------------

    for example, prediction in zip(examples, predictions):
        print(f"Message: {example}")
        print(f"Prediction: {prediction}")
        print()


if __name__ == "__main__":
    main()
