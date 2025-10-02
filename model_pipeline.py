from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# =============================
# Config global
# =============================
RANDOM_STATE = 42

# =============================
# 1. Préparation des données
# =============================
def prepare_data(
    csv_path: str,
    sep: str = ";",
    target: str = "Salary",
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    encoding: str = "utf-8-sig",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, List[str]]]:
    """
    Lire le CSV, splitter train/test et détecter colonnes numériques / catégorielles.
    encoding='utf-8-sig' gère le BOM Windows.
    """
    df = pd.read_csv(csv_path, sep=sep, encoding=encoding)
    if target not in df.columns:
        raise ValueError(f"Target '{target}' introuvable. Colonnes: {list(df.columns)}")

    n0 = len(df)
    df = df.dropna(subset=[target])
    if len(df) < n0:
        print(f"[prepare_data] Lignes supprimées (y NaN): {n0 - len(df)}")

    X = df.drop(columns=[target])
    y = df[target]

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    meta = {"num_cols": num_cols, "cat_cols": cat_cols, "all_cols": X.columns.tolist()}

    print(f"[prepare_data] Num: {num_cols}")
    print(f"[prepare_data] Cat: {cat_cols}")
    return X_train, X_test, y_train, y_test, meta

# =============================
# 2. Pipelines
# =============================
def _ohe_dense() -> OneHotEncoder:
    """Retourne un OneHotEncoder compatible toutes versions sklearn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_pipeline(num_cols: List[str], cat_cols: List[str], model_choice: str = "rf") -> Pipeline:
    """Assembler prétraitements + modèle."""
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", _ohe_dense()),
    ])
    preproc = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    if model_choice == "rf":
        model = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    elif model_choice == "linreg":
        model = LinearRegression()
    else:
        raise ValueError("Choisir 'rf' ou 'linreg'.")

    return Pipeline([("preproc", preproc), ("model", model)])

# =============================
# 3. Entraînement & Évaluation
# =============================
def train_model(pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit du pipeline complet."""
    pipe.fit(X_train, y_train)
    return pipe

def evaluate_model(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Retourne r2, mae, rmse sur le test set."""
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    return {"r2": float(r2), "mae": float(mae), "rmse": rmse}

def cross_validate_model(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict[str, float]:
    """CV sur tout le dataset (scores moyens + std)."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    r2_scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2", n_jobs=-1)
    mae_scores = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    rmse_scores = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    return {
        "cv_mean_r2": float(r2_scores.mean()), "cv_std_r2": float(r2_scores.std()),
        "cv_mean_mae": float(mae_scores.mean()), "cv_std_mae": float(mae_scores.std()),
        "cv_mean_rmse": float(rmse_scores.mean()), "cv_std_rmse": float(rmse_scores.std()),
    }

# =============================
# 4. Sauvegarde / Chargement
# =============================
def save_model(pipe: Pipeline, path: Path | str) -> None:
    """Sauvegarder le pipeline complet (préproc + modèle)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)

def load_model(path: Path | str) -> Pipeline:
    """Charger un pipeline entraîné."""
    return joblib.load(path)
