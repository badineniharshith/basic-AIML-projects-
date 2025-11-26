import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings("ignore")

RND_STATE = 42

def load_data(path="train.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}. Download from Kaggle and place in same folder.")
    df = pd.read_csv(path)
    return df

def eda_and_plots(df, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    print(df.head().to_string(index=False))
    print("\nMissing values:\n", df.isnull().sum())

    plt.figure(figsize=(6,4))
    sns.countplot(x="Survived", data=df)
    plt.title("Survival Count")
    plt.savefig(os.path.join(out_dir, "survival_count.png"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6,4))
    sns.countplot(x="Sex", hue="Survived", data=df)
    plt.title("Survival by Sex")
    plt.savefig(os.path.join(out_dir, "sex_survival.png"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6,4))
    sns.countplot(x="Pclass", hue="Survived", data=df, order=[1,2,3])
    plt.title("Survival by Pclass")
    plt.savefig(os.path.join(out_dir, "class_survival.png"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8,4))
    sns.kdeplot(df.loc[df.Survived==0, "Age"].dropna(), label="Died", shade=True)
    sns.kdeplot(df.loc[df.Survived==1, "Age"].dropna(), label="Survived", shade=True)
    plt.title("Age distribution by Survival")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "age_dist_by_survival.png"), bbox_inches="tight")
    plt.close()

def preprocess(df):
    df = df.copy()
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True, errors='ignore')

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    df['Sex'] = df['Sex'].map({'male':1, 'female':0}).astype(int)
    df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)

    X = df.drop(columns=['PassengerId','Survived'], errors='ignore')
    y = df['Survived']
    return X, y

def build_and_evaluate(X, y, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RND_STATE, stratify=y)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # LogisticRegression
    logreg = LogisticRegression(max_iter=500, random_state=RND_STATE)
    logreg.fit(X_train_scaled, y_train)
    y_pred_log = logreg.predict(X_test_scaled)
    acc_log = accuracy_score(y_test, y_pred_log)
    print(f"\nLogistic Regression Accuracy: {acc_log:.4f}")
    print(classification_report(y_test, y_pred_log))
    cm_log = confusion_matrix(y_test, y_pred_log)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_log, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Logistic Regression")
    plt.savefig(os.path.join(out_dir, "confmat_logreg.png"), bbox_inches="tight")
    plt.close()
    joblib.dump({'model': logreg, 'scaler': scaler, 'features': X.columns.tolist()}, os.path.join(out_dir, "model_logistic.pkl"))
    print(f"Saved Logistic model to {os.path.join(out_dir, 'model_logistic.pkl')}")

    # RandomForest
    rf = RandomForestClassifier(n_estimators=200, random_state=RND_STATE)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"\nRandom Forest Accuracy: {acc_rf:.4f}")
    print(classification_report(y_test, y_pred_rf))
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens")
    plt.title("Confusion Matrix - Random Forest")
    plt.savefig(os.path.join(out_dir, "confmat_rf.png"), bbox_inches="tight")
    plt.close()
    joblib.dump({'model': rf, 'features': X.columns.tolist()}, os.path.join(out_dir, "model_rf.pkl"))
    print(f"Saved Random Forest model to {os.path.join(out_dir, 'model_rf.pkl')}")

    fi = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_}).sort_values(by='importance', ascending=False)
    print("\nFeature importances (Random Forest):\n", fi.to_string(index=False))
    results = {
        'logistic': {'accuracy': acc_log, 'report': classification_report(y_test, y_pred_log, output_dict=True)},
        'random_forest': {'accuracy': acc_rf, 'report': classification_report(y_test, y_pred_rf, output_dict=True)},
        'feature_importance': fi
    }
    fi.to_csv(os.path.join(out_dir, "feature_importance_rf.csv"), index=False)
    return results

def main():
    print("Loading data...")
    df = load_data("train.csv")
    print("\nRunning EDA and saving plots...")
    eda_and_plots(df)
    print("\nPreprocessing data...")
    X, y = preprocess(df)
    print(f"Features used ({len(X.columns)}): {list(X.columns)}")
    print("\nTraining and evaluating models...")
    results = build_and_evaluate(X, y)
    print("\n=== Summary ===")
    print(f"Logistic Accuracy: {results['logistic']['accuracy']:.4f}")
    print(f"Random Forest Accuracy: {results['random_forest']['accuracy']:.4f}")
    print("\nFeature importance (top 5):")
    print(results['feature_importance'].head(5).to_string(index=False))
    print("\nAll outputs saved to ./outputs folder.")

if __name__ == "__main__":
    main()
