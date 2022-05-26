import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from sklearn import metrics
from tabulate import tabulate
from data_gen import BertSemanticDataGenerator

# loading trained model
model = tf.keras.models.load_model("./sub_category_model")
print("trained model loaded...")

# labels
labels = pickle.load(open("./labels.sav", "rb"))
print("labels loaded...")

# loading test data
df = pd.read_csv(
    "../preprocessing/test.csv",
    low_memory=False,
    usecols=[
        "Text",
        "Sub Category"
    ]
)
df.dropna(axis=0, inplace=True)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
print(f"test data loaded...\n -shape: {df.shape}")


def prediction(sentence):
    sentence = np.array([str(sentence)])
    test_data = BertSemanticDataGenerator(
        sentence, labels=None, batch_size=1, shuffle=False, include_targets=False,
    )

    proba = model.predict(test_data[0])[0]
    idx = np.argmax(proba)
    proba = f"{proba[idx]*100: .2f}%"
    pred = labels[idx]
    return pred, proba


# inference
result = [prediction(text) for text in tqdm(df.Text.tolist())]
df["prediction"] = list(map(lambda x: x[0], result))
df["score"] = list(map(lambda x: x[1], result))
print("inference done...")

# classification report
report = metrics.classification_report(y_true=df["Sub Category"].tolist(), y_pred=df.prediction.tolist(), output_dict=True)
print(f"Accuracy: {round(report['accuracy'], 2)}", file=open("./classification_report.txt", "a"))
print(f"Macro Avg Precision: {round(report['macro avg'].get('precision'), 2)}", file=open("./classification_report.txt", "a"))
print(f"Weighted Avg Precision: {round(report['weighted avg'].get('precision'), 2)}", file=open("./classification_report.txt", "a"))

report = pd.DataFrame([
    {
        "label": key,
        "precision": value["precision"],
        "recall": value["recall"],
        "support": value["support"]
    }
    for key, value in report.items()
    if key not in ["accuracy", "macro avg", "weighted avg"]
])
print(
    tabulate(
        report,
        headers="keys",
        tablefmt="psql"
    ),
    file=open("./classification_report.txt", "a")
)
