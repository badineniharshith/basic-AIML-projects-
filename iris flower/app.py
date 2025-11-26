import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings("ignore")

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
RND = 42

data = load_iris(as_frame=True)
X = data.data
y = data.target
feature_names = X.columns.tolist()
target_names = data.target_names.tolist()

print("DATA SHAPE:", X.shape)
print("FEATURES:", feature_names)
print("TARGET NAMES:", target_names)

df = X.copy()
df["target"] = y
sns.pairplot(df, vars=feature_names, hue="target")
plt.savefig(os.path.join(OUT, "pairplot.png"), bbox_inches="tight")
plt.close()

plt.figure(figsize=(8,4))
sns.heatmap(df.corr(), annot=True)
plt.title("Feature Correlation")
plt.savefig(os.path.join(OUT, "corr_matrix.png"), bbox_inches="tight")
plt.close()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RND, stratify=y)

models = {
    "KNN": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
    "SVM": Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True, random_state=RND))]),
    "RandomForest": Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=200, random_state=RND))]),
    "LogisticRegression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500, random_state=RND))])
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cr = classification_report(y_test, preds, target_names=target_names)
    cm = confusion_matrix(y_test, preds)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    results[name] = {"accuracy": acc, "report": cr, "confusion": cm, "cv_mean": cv_scores.mean()}
    joblib.dump(model, os.path.join(OUT, f"model_{name}.pkl"))

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(os.path.join(OUT, f"conf_{name}.png"), bbox_inches="tight")
    plt.close()

print("\n=== RESULTS ===")
for name, res in results.items():
    print(f"\nModel: {name}")
    print(f"Accuracy: {res['accuracy']:.4f}")
    print(f"CV Accuracy (5-fold mean): {res['cv_mean']:.4f}")
    print(res["report"])

# save a small summary csv
summary = pd.DataFrame([{"model": n, "accuracy": r["accuracy"], "cv_mean": r["cv_mean"]} for n, r in results.items()])
summary.to_csv(os.path.join(OUT, "summary_models.csv"), index=False)
print(f"\nArtifacts saved to ./{OUT}/ (models, plots, summary)")
