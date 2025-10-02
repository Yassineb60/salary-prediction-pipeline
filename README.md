## Contenu du projet
- **model_pipeline.py** : contient toutes les fonctions (préparation des données, entraînement, évaluation, etc.).
- **main.py** : fichier principal qui exécute les différentes étapes via des arguments en ligne de commande.
- **Salary_Data.csv** : dataset utilisé pour l’entraînement.
- **requirements.txt** : liste des dépendances nécessaires pour exécuter le projet.

## Installation
1. Créer et activer un environnement virtuel :
   ```bash
   python3 -m venv venv
   source venv/bin/activate
Installer les dépendances :

pip install -r requirements.txt

Utilisation

Exemples de commandes pour exécuter le pipeline :

Entraîner le modèle :

python3 main.py --action train


Évaluer le modèle :

python3 main.py --action evaluate


Faire des prédictions avec un modèle sauvegardé :

python3 main.py --action predict


Lancer une cross-validation :

python3 main.py --action cv
