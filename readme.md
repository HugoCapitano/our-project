# Projet Réseaux de Neurones Artificiels

Ce projet regroupe des outils et des modèles pour l'apprentissage et la validation de réseaux de neurones (Perceptron, Adaline, MLP) sur différents jeux de données, dont le langage des signes.

## Structure du projet

- **Modèles/** : Implémentations des modèles (perceptron, adaline, MLP, etc.)
- **Outils/** : Fonctions utilitaires pour la préparation des datasets, visualisation, initialisation, etc.
- **Validation/** : Scripts pour l'évaluation des modèles sur des jeux de validation.
- **Datas/** : Jeux de données utilisés pour l'entraînement et la validation.
- **Plot/** : Graphiques générés lors de l'entraînement ou de la validation.

## Principales fonctionnalités

- **Préparation des datasets** : Échantillonnage, équilibrage, formatage des données.
- **Entraînement des modèles** : Perceptron, Adaline, MLP (classification et régression).
- **Visualisation** : Courbes d'apprentissage, frontières de décision, matrices de confusion, prototypes de classes.
- **Validation** : Évaluation sur des jeux de validation, calcul d'accuracy, affichage détaillé des prédictions.

## Exécution

1. **Préparer les données**  
   Utiliser le scripts du dossier `Outils/prep_datasets` pour formater et équilibrer les datasets pour MLP (langage des signes).

2. **Entraîner un modèle**  
   Utiliser le scripts du dossier `Outils/train_mlp2` pour formater et équilibrer les datasets pour MLP (langage des signes).

3. **Visualiser les résultats**  
   Utiliser les scripts du dossier `Validation/` pour évaluer le modèle sur un jeu de validation.
   - Chaque résultat est enregistré dans `Plot/`

## Dépendances

- Python 3.10+
- numpy
- pandas
- matplotlib

## Exemple d'utilisation

```bash
python Outils/prep_datasets.py
python Outils/train_mlp2.py
python Validation/Langage_signe.py
```

## Auteurs

- Projet réalisé dans le cadre du cours "Réseaux de neurones artificiels" (HEPL MASI Bloc 1).
- Capitano Hugo & Dubrunquez Lorigenne Tobias 


