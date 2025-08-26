import pandas as pd
import numpy as np


def load_data(path, *, has_header=False, y_col=-1, sep=","):
    """
    Lit un fichier CSV et retourne les données d'entrée (X) et les cibles (y).

    Paramètres :
    ------------
    path : str
        Chemin vers le fichier CSV.
    has_header : bool
        True si le fichier contient une ligne d'en-tête, False sinon.
    y_col : int ou str
        Index (int) ou nom (str) de la colonne cible. Par défaut, la dernière colonne (-1).
    sep : str
        Séparateur de colonnes du CSV.

    Retour :
    --------
    X : ndarray shape (n_samples, n_features)
        Données d'entrée converties en float.
    y : ndarray shape (n_samples,)
        Cibles (aplatie en 1D).
    """
    df = pd.read_csv(path, header=0 if has_header else None, sep=sep)

    # Sélection de y (par position si int, par nom si str)
    y = df.iloc[:, y_col] if isinstance(y_col, int) else df[y_col]
    X = df.drop(df.columns[y_col], axis=1) if isinstance(y_col, int) else df.drop(columns=[y_col])

    # Sortie numpy propre
    X = X.to_numpy(dtype=float)
    y = y.to_numpy().ravel()
    return X, y

def add_x0(X):
    """
    Ajoute une colonne de 1 (biais) en première position de la matrice X.

    Exemple :
    ---------
    X = [[2, 3],
         [4, 5]]
    Résultat :
         [[1, 2, 3],
          [1, 4, 5]]
    """
    X_bias = np.c_[np.ones(X.shape[0]), X]
    return X_bias

def ini_weight(X_bias):
    """
    Initialise les poids à zéro pour un modèle linéaire.

    Paramètres :
    ------------
    X_bias : ndarray shape (n_samples, n_features+1)
        Données avec colonne de biais.

    Retour :
    --------
    weight : ndarray shape (n_features+1,)
        Poids initialisés à zéro.
    """
    weight = np.zeros(X_bias.shape[1])
    return weight

def ini_weight_multiclass(X_bias, num_classes):
    """
    Initialise une matrice de poids à zéro pour un problème multi-classes.

    Paramètres :
    ------------
    X_bias : ndarray shape (n_samples, n_features+1)
        Données avec colonne de biais.
    num_classes : int
        Nombre de classes.

    Retour :
    --------
    weights : ndarray shape (num_classes, n_features+1)
        Matrice de poids initialisée à zéro.
    """
    return np.zeros((num_classes, X_bias.shape[1]))



def load_data_3_5(file_path, n_classes=4):
    """
    Charge un CSV où les n_classes dernières colonnes sont les cibles en {-1,+1}
    (schéma one-vs-all). Retourne X (features) et y (étiquette 0..n_classes-1).

    Paramètres :
    ------------
    file_path : str
        Chemin du fichier CSV.
    n_classes : int
        Nombre de classes.

    Retour :
    --------
    X : ndarray shape (n_samples, n_features)
        Données d'entrée.
    y : ndarray shape (n_samples,)
        Cibles codées 0..n_classes-1.
    """

    df = pd.read_csv(file_path, header=None).astype(float)
    n_total_cols = df.shape[1]
    feat_cols = n_total_cols - n_classes

    X = df.iloc[:, :feat_cols].values
    Y_pm1 = df.iloc[:, feat_cols:].values          # (m, n_classes) en {-1,+1}
    Y01   = (Y_pm1 + 1) / 2.0                      # -> {0,1}
    y     = np.argmax(Y01, axis=1).astype(int)     # 0..n_classes-1

    # Affichage de la répartition des classes
    unique, counts = np.unique(y, return_counts=True)
    print("Répartition classes:", dict(zip(unique, counts)))

    return X, y


def predict(X_bias, weights):
    """
    Retourne la classe prédite pour chaque échantillon
    en choisissant l'indice du score maximum.

    Paramètres :
    ------------
    X_bias : ndarray shape (n_samples, n_features+1)
        Données avec colonne de biais.
    weights : ndarray shape (num_classes, n_features+1)
        Matrice des poids.

    Retour :
    --------
    y_pred : ndarray shape (n_samples,)
        Classes prédites.
    """

    scores = np.dot(weights, X_bias.T)          # (num_classes, m)
    return np.argmax(scores, axis=0)            # Indice de la classe max




def load_classification_lastcol(file_path):
    """
    Charge un CSV où la DERNIÈRE colonne est la cible.
    Gère aussi le cas {-1, +1} en le convertissant en {0, 1}.

    Paramètres :
    ------------
    file_path : str
        Chemin du fichier CSV.

    Retour :
    --------
    X : ndarray shape (n_samples, n_features)
        Données d'entrée.
    y : ndarray shape (n_samples,)
        Cibles converties en entiers.
    """
    df = pd.read_csv(file_path, header=None).astype(float)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Conversion éventuelle de {-1, +1} en {0, 1}
    uniq = np.unique(y)
    if set(uniq.tolist()) == {-1.0, 1.0} or set(uniq.tolist()) == {-1, 1}:
        y = ((y + 1) / 2).astype(int)  # -1/+1 -> 0/1
    else:
        y = y.astype(int)

    return X, y



def load_classification_n_classes(file_path, n_classes):
    """
    Charge un CSV où les n_classes dernières colonnes sont les cibles en {-1,+1}
    (one-vs-all). Retourne X et y en 0..n_classes-1.

    Paramètres :
    ------------
    file_path : str
        Chemin du fichier CSV.
    n_classes : int
        Nombre de classes.

    Retour :
    --------
    X : ndarray
        Données d'entrée.
    y : ndarray
        Cibles codées 0..n_classes-1.
    """
    df = pd.read_csv(file_path, header=None).astype(float)
    feat_end = df.shape[1] - n_classes
    X = df.iloc[:, :feat_end].values
    Ypm1 = df.iloc[:, feat_end:].values
    Y01 = (Ypm1 + 1.0) / 2.0
    y = np.argmax(Y01, axis=1).astype(int)
    return X, y

def load_regression_mlp(file_path):
    """
    Charge un CSV pour un problème de régression MLP :
    - Toutes les colonnes sauf la dernière = X
    - Dernière colonne = y (réelle)

    Paramètres :
    ------------
    file_path : str
        Chemin du fichier CSV.

    Retour :
    --------
    X : ndarray
        Données d'entrée.
    y : ndarray shape (n_samples, 1)
        Valeurs cibles réelles.
    """
    df = pd.read_csv(file_path, header=None).astype(float)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1,1)
    return X, y





