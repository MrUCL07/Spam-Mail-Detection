
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import(
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT=Path(__file__).resolve().parent
DATA_PATH=ROOT /"data"/"SMSSpamCollection"
MODEL_DIR=ROOT/"models"
MODEL_PATH=MODEL_DIR/"spam_model.joblib"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}."
        "Download the extract the UCI SMS Spam Collection" \
        "Into the data folder"
    )

df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["Label", "Message"],
    encoding="utf-8"
)

df = df.dropna(subset=["Label", "Message"])

df["Label"] = df["Label"].str.strip().str.lower()
df["Message"] = df["Message"].astype(str).str.strip()

df = df[df["Label"].isin(["ham", "spam"])]
df = df[df["Message"] != ""]
df=df.drop_duplicates(subset=["Label","Message"])

x=df["Message"]
y=df["Label"].map({"ham":0,"spam":1})

x_train,x_test,y_train,y_test=train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

TfidfVectorizer(
    lowercase=True,
    ngram_range=(1,2),
    max_features=30000,
    sublinear_tf=True,
)

LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)

model=Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1,2),
            max_features=30000,
            sublinear_tf=True,
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
    ),

])

model.fit(x_train,y_train)
predictions=model.predict(x_test)

print("Accuracy:",round(
    accuracy_score(y_test, predictions),4
))

print(classification_report(
    y_test,predictions,
    labels=[0,1],
    target_names=["Legitimate","Spam"]
))

print(confusion_matrix(y_test,predictions,labels=[0,1]))

MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"\nSaved model to: {MODEL_PATH}")