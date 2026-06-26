import io
import os

import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Given the text of an email, can we predict whether it is spam or ham?
# raw emails
# ↓
# extract message body
# ↓
# create DataFrame with message + label
# ↓
# convert text to word-count vectors
# ↓
# train Naive Bayes classifier
# ↓
# predict spam/ham for new messages


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

    # This creates an empty DataFrame with two columns:
    # message
    # class
    data = DataFrame({"message": [], "class": []})

    spam_data = dataframe_from_directory("emails/spam", "spam")
    ham_data = dataframe_from_directory("emails/ham", "ham")

    # This reads spam emails and ham emails, then combines them into one dataset.
    # So at the end, data contains both classes:
    # spam emails + ham emails
    data = pd.concat([data, spam_data])
    data = pd.concat([data, ham_data])

    # This is your labeled training dataset. This is why it is supervised learning.
    # The model is given examples where the correct answer is already known:
    #
    # this email → spam
    # this email → ham
    # This is spam. This is normal. This is spam. This is normal. Now try classifying a new one.
    print(f"Loaded {len(data)} emails.")
    print(data["class"].value_counts())

    # -----------------------------
    # 2. Split into train/test
    # -----------------------------

    train_messages, test_messages, train_labels, test_labels = train_test_split(
        data["message"].values,
        data["class"].values,
        test_size=0.2,
        random_state=42,
    )

    # -----------------------------
    # 3. Convert text into word counts
    # -----------------------------

    # create a table-like structure called a DataFrame.
    # Machine learning models cannot directly process raw text.
    vectorizer = CountVectorizer()

    # fit       → learn the vocabulary from the emails
    # transform → convert each email into numeric word-count features

    # learn the vocabulary from the training emails
    # +
    # convert the training emails into numbers
    train_counts = vectorizer.fit_transform(train_messages)

    # do NOT learn anything new
    # +
    # convert the test emails into numbers using the same vocabulary learned from training
    test_counts = vectorizer.transform(test_messages)

    # -----------------------------
    # 4. Train Naive Bayes classifier
    # -----------------------------
    # is the machine learning classifier. NB means Naive Bayes.
    #
    # This is a classic model for text classification, especially spam detection.
    classifier = MultinomialNB()
    classifier.fit(train_counts, train_labels)

    # -----------------------------
    # 5. Evaluate on test data
    # -----------------------------

    test_predictions = classifier.predict(test_counts)
    accuracy = accuracy_score(test_labels, test_predictions)

    print(f"Test accuracy: {accuracy:.4f}")

    # -----------------------------
    # 6. Predict new examples
    # -----------------------------

    examples = [
        "Limited-time supplement offer available now!!!",
        "Hi Bob, how about a game of golf tomorrow?",
    ]
    # This converts the new examples into the same numeric format used during training.
    example_counts = vectorizer.transform(examples)
    predictions = classifier.predict(example_counts)

    for example, prediction in zip(examples, predictions):
        print(f"Message: {example}")
        print(f"Prediction: {prediction}")
        print()


if __name__ == "__main__":
    main()
