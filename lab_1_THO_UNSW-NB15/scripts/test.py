
import pandas as pd

df = pd.read_csv("data/UNSW_NB15_testing-set.csv")
print("Columns contain label?", "label" in df.columns)
print("label value_counts:")
print(df["label"].value_counts(dropna=False))
print("label unique:", sorted(df["label"].dropna().unique().tolist())[:20])



import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report

df = pd.read_csv("artifacts/predictions.csv")

print("columns:", [c for c in ["label","is_anomaly","anomaly_score"] if c in df.columns])
print("label counts:\n", df["label"].value_counts())
print("pred counts (is_anomaly):\n", df["is_anomaly"].value_counts())

y_true = df["label"].astype(int)
y_pred = df["is_anomaly"].astype(int)

cm = confusion_matrix(y_true, y_pred, labels=[0,1])
print("\nConfusion matrix (rows=true 0/1, cols=pred 0/1):\n", cm)

print("\nPrecision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("\nReport:\n", classification_report(y_true, y_pred, digits=4))


import yaml, pandas as pd
from pathlib import Path

meta = Path("artifacts/models/isolation-forest_metadata.yaml")
print("Metadata exists:", meta.exists())
md = yaml.safe_load(meta.read_text(encoding="utf-8"))

features = md.get("features", [])
print("features in metadata:", len(features))
print("first 30:", features[:30])

df = pd.read_csv("data/UNSW_NB15_testing-set.csv")
missing = [f for f in features if f not in df.columns]
print("missing features in raw test csv:", missing)

