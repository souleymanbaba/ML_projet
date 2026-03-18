# Prediction des Prix Immobiliers a Nouakchott

Projet Capstone Master 1 — SupNum
Compétition Kaggle : prédiction des prix immobiliers en Mauritanie

---

## Contexte

Ce projet a été réalisé dans le cadre de notre capstone de fin d'année à SupNum. L'objectif est de construire un modéle de machine learning capable de prédire le prix de vente d'un bien immobilier à Nouakchott à partir d'annonces réelles collectées sur des plateformes mauritaniennes.

Le dataset contient 1153 annonces en entrainement et 289 annonces en test. Les descriptions sont écrites en arabe et en français, ce qui rend le traitement du texte un peu particulier.

---

## Membres de l'equipe

- 22018 Souleyman Baba
- 22051 Ahmed Elmewloud
- 22074 Mouna Elkhay

---

## Structure du projet

```
ML_projet/
    kaggle_train.csv          donnees d'entrainement (1153 lignes)
    kaggle_test.csv           donnees de test (289 lignes)
    Projet_machine.ipynb    notebook principal
    submission.csv            fichier de soumission Kaggle
    feature_importance.png    graphique des features importantes
    README.md
```

---

## Metrique

La métrique principale de la compétition est le **RMSLE** (Root Mean Squared Log Error). On travaille donc directement sur `log(1 + prix)` pendant l'entrainement, ce qui est naturel pour cette métrique.

---

## Pipeline

**1. Extraction du prix depuis le text de description**

On a découvert que plus de 50% des annonces mentionnent le prix directement dans la description en arabe (ex: "8 مليون"). On extrait ce prix avec des expressions régulières et on gère l'ambiguité MRO/MRU.

Pour éviter le data leakage dans le TF-IDF, on crée une version nettoyée du texte sans les mentions de prix.

**2. Nettoyage**

- Normalisation des noms de quartiers (variantes orthographiques)
- Imputation des valeurs manquantes par la médiane
- Suppression des outliers (1er et 99eme percentile)

**3. Geo-enrichissement via OpenStreetMap**

On utilise l'API Nominatim pour recuperer les coordonnées GPS réelles de chaque quartier, et l'API Overpass pour compter les points d'intérêt (écoles, mosquées, commerces, hopitaux) dans un rayon de 1km.

**4. Feature Engineering**

113 features au total :
- Features structurelles : surface (log, carré, tranches), nombre de pièces, taille de rue...
- Features de type de bien : is_villa, is_duplex, is_land, has_tf, has_garage...
- Features géographiques : dist_centre_km, dist_aeroport_km, nb_total_pois_1km...
- Features texte : TF-IDF 50 tokens sur description nettoyée, flags binaires
- Target encoding avec smoothing pour quartier et proche_de (recalculé par fold)

**5. Modélisation**

Cross-validation 5-fold avec 4 modèles :
- Ridge (baseline linéaire)
- LightGBM
- XGBoost
- CatBoost

Les poids de l'ensemble sont optimisés par algorithme Nelder-Mead sur les prédictions out-of-fold.

**6. Architecture MoE (Mixture of Experts)**

```
Si has_price = 1 :
    prediction = 0.80 x prix_texte + 0.20 x modele_structural

Si has_price = 0 :
    prediction = modele_structural
```

L'alpha de 0.80 a été trouvé par grille de recherche sur les OOF.

---

## Resultats

| Modele | RMSLE CV |
|--------|----------|
| Ridge (baseline) | 0.5875 |
| LightGBM | 0.5717 |
| XGBoost | 0.5644 |
| CatBoost | 0.5688 |
| Ensemble XGB + CB | 0.5630 |
| MoE final | **0.4969** |

Meuilleur Score Kaggle public  : 0.52173

---

## Comment executer

1. Installer les dépendences :

```bash
pip install lightgbm catboost xgboost scikit-learn pandas numpy requests
```

ou voir requirements.txt

2. Placer `kaggle_train.csv` et `kaggle_test.csv` dans le meme dossier que le notebook

3. Executer toutes les cellules du notebook `nouakchott_final.ipynb`

4. Le fichier `submission.csv` est généré automatiquement à la fin

Note : la cellule de geo-enrichissement fait des requetes HTTP vers OpenStreetMap, il faut donc une connexion internet. Le temps d'execution de cette cellule est d'environ 30 secondes (rate limit respecté).

---

## Particularites du dataset

- Les descriptions sont en arabe, parfois melangées avec du français
- Deux devises coexistent : MRO (ancienne) et MRU (nouvelle), 1 MRO = 10 MRU
- 50.9% des annonces train contiennent le prix dans la description
- La couverture OSM de Nouakchott est incomplète pour certains quartiers (Teyarett notamment)

---

## References

- LightGBM : https://lightgbm.readthedocs.io
- XGBoost : https://xgboost.readthedocs.io
- CatBoost : https://catboost.ai
- Nominatim API : https://nominatim.openstreetmap.org
- Overpass API : https://overpass-api.de
