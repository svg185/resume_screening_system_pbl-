import os
import pickle

from preprocess import clean_text


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf.pkl")


def _load_artifact(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{label} not found. Run `python train_model.py` before starting the app."
        )

    with open(path, "rb") as artifact_file:
        return pickle.load(artifact_file)


model = _load_artifact(MODEL_PATH, "model.pkl")
tfidf = _load_artifact(VECTORIZER_PATH, "tfidf.pkl")


def predict_resume(text):
    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])
    prediction = str(model.predict(vector)[0])
    return prediction


def predict_resume_with_confidence(text):
    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])
    prediction = str(model.predict(vector)[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vector)[0]
        confidence = float(probabilities.max())
    else:
        confidence = 0.0

    return prediction, confidence
