import math
import pickle

import pandas as pd
from preprocess import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


DATASET_PATH = "resume_dataset.csv"
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "tfidf.pkl"


def train():
    df = pd.read_csv(DATASET_PATH)

    required_columns = {"Resume", "Category"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing_columns))}")

    df = df.dropna(subset=["Resume", "Category"]).copy()
    df["cleaned"] = df["Resume"].apply(clean_text)

    X = df["cleaned"]
    y = df["Category"]

    tfidf = TfidfVectorizer(max_features=5000)
    X_vectorized = tfidf.fit_transform(X)

    test_size = 0.2
    test_count = max(1, math.ceil(len(y) * test_size))
    class_count = y.nunique()
    can_stratify = y.value_counts().min() >= 2
    if can_stratify and test_count < class_count:
        test_size = class_count / len(y)
        test_count = class_count

    stratify = y if can_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)

    with open(VECTORIZER_PATH, "wb") as vectorizer_file:
        pickle.dump(tfidf, vectorizer_file)

    print("Model trained successfully!")
    print(f"Accuracy: {accuracy:.2%}")

    if len(set(y_test)) > 1:
        y_pred = model.predict(X_test)
        print("\nClassification report:")
        print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    train()
