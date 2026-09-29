
# Prédiction du prix de voitures d'occasion

Projet de machine learning (régression) visant à prédire le prix d'une voiture d'occasion à partir de ses caractéristiques (marque, modèle, année, kilométrage, motorisation, historique d'accidents, etc.).

## Objectif

Construire et comparer plusieurs modèles de régression afin d'estimer le prix d'un véhicule, en passant par toutes les étapes d'un projet ML : exploration, nettoyage des données, encodage, entraînement, optimisation et évaluation.

## Données
Source : [ajouter le lien du dataset, ex. Kaggle]
Taille : 4 009 annonces, 12 colonnes (3 931 lignes après nettoyage)
Variables :
Variable	Description
brand, model	Marque et modèle du véhicule
model_year	Année du modèle
milage	Kilométrage (en miles)
fuel_type	Type de carburant
engine	Description du moteur
transmission	Type de boîte de vitesses
ext_col, int_col	Couleur extérieure / intérieure
accident	Accident ou dommage déclaré
clean_title	Titre de propriété "propre" ou non
price	Variable cible : prix en dollars

## Méthodologie
Nettoyage des données
Valeurs manquantes : fuel_type remplacé par le mode (Gasoline), accident et clean_title marqués comme Unknown
Vérification des doublons (aucun trouvé)
Conversion de price ($15,000 → 15000.0) et de milage (51,000 mi. → 51000.0) en valeurs numériques
Extraction de la valeur numérique de la colonne engine
Suppression des lignes où le moteur n'a pas pu être extrait
Encodage : LabelEncoder pour les variables catégorielles, encodage manuel pour accident, clean_title et fuel_type
Séparation : 80 % entraînement / 20 % test
Modèles testés : Régression linéaire, Random Forest (avec GridSearchCV), XGBoost
Évaluation : MAE, RMSE et R²

## Résultats
Modèle	MAE ($)	RMSE ($)	R²
Régression linéaire	23 855	43 871	0.21
Random Forest	14 231	33 290	0.54
Random Forest (GridSearchCV)	14 008	–	0.57
XGBoost	11 703	28 334	0.67

➡️ XGBoost obtient les meilleures performances : en moyenne, ses prédictions s'écartent d'environ 11 700 $ du prix réel.

(Ajouter ici une capture du graphique « Prix réel vs prédit ».)

## Structure du projet
├── PROJET_CARS_PRICE.ipynb   # Notebook complet (analyse + modèles)
├── data/                     # Dataset (ou lien de téléchargement)
├── requirements.txt          # Dépendances
└── README.md

⚙️ Installation et utilisation
bash
git clone https://github.com/<ton-pseudo>/<nom-du-repo>.git
cd <nom-du-repo>
pip install -r requirements.txt
jupyter notebook PROJET_CARS_PRICE.ipynb

Le notebook a été réalisé sur Google Colab : adapter le chemin du fichier CSV (pd.read_csv(...)) si vous l'exécutez en local.

⚠️ Limites et pistes d'amélioration
Le dataset est de petite taille et contient des prix extrêmes (jusqu'à ~2,9 M$) qui pénalisent les modèles : à traiter (outliers, transformation logarithmique de la cible)
Les variables brand, model et les couleurs sont encodées avec LabelEncoder, ce qui impose un ordre artificiel : tester le One-Hot ou le Target Encoding
Extraire séparément la puissance (HP) et la cylindrée (L) depuis la colonne engine
Utiliser un Pipeline scikit-learn (prétraitement + modèle) et de la validation croisée
Analyser l'importance des variables et tester d'autres modèles (LightGBM, CatBoost)
Déployer le modèle via une petite application (Streamlit, Flask)
🛠️ Technologies

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · XGBoost · Joblib

## Auteur

[Fanilo FETRAHARIMANANA] – Étudiante en M1 Informatique et Technologie .
