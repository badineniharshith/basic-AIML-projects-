import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings("ignore")

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
RND = 42

def load_data():
    if os.path.exists("train.csv"):
        df = pd.read_csv("train.csv")
        if "SalePrice" not in df.columns:
            raise ValueError("train.csv found but 'SalePrice' column missing.")
        source = "kaggle_train"
    else:
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing(as_frame=True)
        df = data.frame
        df = df.rename(columns={"MedHouseVal": "SalePrice"}) if "MedHouseVal" in df.columns else df
        # if using sklearn california housing, create SalePrice as target
        df["SalePrice"] = data.target
        source = "sklearn_california"
    return df, source

def eda(df):
    print("\nDATA SHAPE:", df.shape)
    print("\nCOLUMNS & TYPES:\n", df.dtypes)
    print("\nMISSING VALUES:\n", df.isnull().sum().sort_values(ascending=False).head(20))
    sns.histplot(df["SalePrice"], kde=True)
    plt.title("SalePrice distribution")
    plt.savefig(os.path.join(OUT_DIR, "saleprice_dist.png"), bbox_inches="tight")
    plt.close()
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric].corr()["SalePrice"].sort_values(ascending=False)
    corr.to_csv(os.path.join(OUT_DIR, "correlation_with_target.csv"))
    top_corr = corr.head(20)
    plt.figure(figsize=(8,6))
    sns.barplot(x=top_corr.values, y=top_corr.index)
    plt.title("Top correlations with SalePrice")
    plt.savefig(os.path.join(OUT_DIR, "top_correlations.png"), bbox_inches="tight")
    plt.close()

def preprocess_for_kaggle(df):
    df = df.copy()
    if "SalePrice" not in df.columns:
        raise ValueError("Target SalePrice missing")
    y = df["SalePrice"].copy()
    X = df.drop(columns=["SalePrice"])

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    cat_cols_keep = [c for c in cat_cols if X[c].nunique() < 30]

    # create OneHotEncoder in a version-robust way
    from sklearn.preprocessing import OneHotEncoder as _OHE
    try:
        ohe = _OHE(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = _OHE(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="NA")), ("ohe", ohe)]), cat_cols_keep)
    ], remainder="drop")

    X_trans = preprocessor.fit_transform(X)

    num_feats = numeric_cols
    cat_feats = []
    if cat_cols_keep:
        ohe_step = preprocessor.named_transformers_["cat"].named_steps["ohe"]
        try:
            cat_names = ohe_step.get_feature_names_out(cat_cols_keep).tolist()
        except Exception:
            # fallback for very old sklearn
            cat_names = []
            for i, col in enumerate(cat_cols_keep):
                n_unique = preprocessor.named_transformers_["cat"].named_steps["ohe"].categories_[i]
                cat_names += [f"{col}_{val}" for val in n_unique]
        cat_feats = cat_names

    feature_names = num_feats + cat_feats
    X_df = pd.DataFrame(X_trans, columns=feature_names, index=X.index)
    return X_df, y, preprocessor


def preprocess_generic(df):
    # fallback for sklearn california housing: minimal processing
    df = df.copy()
    if "SalePrice" in df.columns:
        y = df["SalePrice"].copy()
        X = df.drop(columns=["SalePrice"])
    else:
        raise ValueError("No target found")
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_num = imputer.fit_transform(X[numeric_cols])
    X_num = scaler.fit_transform(X_num)
    X_df = pd.DataFrame(X_num, columns=numeric_cols, index=X.index)
    return X_df, y, (imputer, scaler)

def train_and_eval(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RND)
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=RND),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=RND)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error")
        results[name] = {"model": model, "mae": mae, "rmse": rmse, "r2": r2, "cv_rmse_mean": -cv_scores.mean()}
        # save predictions plot
        plt.figure(figsize=(6,4))
        plt.scatter(y_test, preds, alpha=0.4)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"{name} Actual vs Predicted")
        plt.savefig(os.path.join(OUT_DIR, f"pred_actual_{name}.png"), bbox_inches="tight")
        plt.close()
        joblib.dump(model, os.path.join(OUT_DIR, f"model_{name}.pkl"))
    # gridsearch for RandomForest (simple)
    param_grid = {"n_estimators": [100, 200], "max_depth": [None, 10, 20], "min_samples_split": [2,5]}
    grid = GridSearchCV(RandomForestRegressor(random_state=RND), param_grid, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)
    grid.fit(X_train, y_train)
    best_rf = grid.best_estimator_
    preds_grid = best_rf.predict(X_test)
    rmse_grid = np.sqrt(mean_squared_error(y_test, preds_grid))
    joblib.dump(grid, os.path.join(OUT_DIR, "gridsearch_rf.pkl"))
    results["RandomForest_GridBest"] = {"model": best_rf, "rmse": rmse_grid, "best_params": grid.best_params_}
    return results

def save_summary(results):
    summary = {}
    for k,v in results.items():
        if k.startswith("RandomForest_GridBest"):
            summary[k] = {"rmse": v.get("rmse"), "best_params": v.get("best_params")}
        else:
            summary[k] = {"mae": v.get("mae"), "rmse": v.get("rmse"), "r2": v.get("r2"), "cv_rmse_mean": v.get("cv_rmse_mean")}
    with open(os.path.join(OUT_DIR, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary saved to outputs/results_summary.json")
    print(json.dumps(summary, indent=2))

def main():
    df, source = load_data()
    print("data source:", source)
    eda(df)
    # choose preprocess path
    if source == "kaggle_train":
        X, y, preproc = preprocess_for_kaggle(df)
        joblib.dump(preproc, os.path.join(OUT_DIR, "preprocessor.pkl"))
    else:
        X, y, preproc = preprocess_generic(df)
        joblib.dump(preproc, os.path.join(OUT_DIR, "preprocessor_generic.pkl"))
    results = train_and_eval(X, y)
    save_summary(results)
    print("\nModels and artifacts saved in outputs/")

if __name__ == "__main__":
    main()
