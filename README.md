# AI Resume Screening System

This project follows the resume screening ML pipeline:

1. Load the resume dataset.
2. Clean resume text.
3. Convert text to TF-IDF vectors.
4. Train a Multinomial Naive Bayes classifier.
5. Predict resume category from pasted text or uploaded files.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train_model.py
```

This creates:

- `model.pkl`
- `tfidf.pkl`

## Run

```bash
streamlit run app.py
```

## Dataset Format

```csv
Resume,Category
"Experienced Python developer with ML skills","Data Science"
"Java backend developer with Spring Boot","Software Engineer"
"HR management and recruitment experience","HR"
```
