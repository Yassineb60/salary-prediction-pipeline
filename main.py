import argparse
import pandas as pd
from model_pipeline import (
    prepare_data,
    build_pipeline,
    train_model,
    evaluate_model,
    cross_validate_model,
    save_model,
    load_model,
)

MODEL_PATH = "salary_model.joblib"
CSV_PATH = "Salary_Data.csv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de régression salariale")
    parser.add_argument("--csv", type=str, default=CSV_PATH, help="Chemin du fichier CSV")
    parser.add_argument("--sep", type=str, default=";", help="Séparateur CSV")
    parser.add_argument("--model", type=str, choices=["rf", "linreg"], default="rf", help="Choix du modèle")
    parser.add_argument("--action", type=str, choices=["train", "evaluate", "predict", "cv"], required=True,
                        help="Action à exécuter")
    args = parser.parse_args()

    # Préparer les données
    try:
        X_train, X_test, y_train, y_test, meta = prepare_data(args.csv, sep=args.sep)
    except FileNotFoundError:
        print(f"❌ Erreur : le fichier {args.csv} est introuvable.")
        exit(1)

    if args.action == "train":
        pipe = build_pipeline(meta["num_cols"], meta["cat_cols"], model_choice=args.model)
        pipe = train_model(pipe, X_train, y_train)
        save_model(pipe, MODEL_PATH)
        print(f"✅ Modèle {args.model} entraîné et sauvegardé → {MODEL_PATH}")

    elif args.action == "evaluate":
        pipe = build_pipeline(meta["num_cols"], meta["cat_cols"], model_choice=args.model)
        pipe = train_model(pipe, X_train, y_train)
        results = evaluate_model(pipe, X_test, y_test)
        print("📊 Résultats test:", results)

    elif args.action == "predict":
        model = load_model(MODEL_PATH)
        preds = model.predict(X_test.iloc[:5])
        print("🔮 Exemple de prédictions:", preds)

    elif args.action == "cv":
        pipe = build_pipeline(meta["num_cols"], meta["cat_cols"], model_choice=args.model)
        X_full = pd.concat([X_train, X_test], axis=0)
        y_full = pd.concat([y_train, y_test], axis=0)
        cv_scores = cross_validate_model(pipe, X_full, y_full, n_splits=5)
        print("📈 Résultats cross-validation:", cv_scores)

