import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # ou 'Qt5Agg', 'Agg' (pour sauvegarder sans afficher)
import os

def _pm1_to_labels(Y_pm1: np.ndarray) -> np.ndarray:
    """
    Convertit un codage {-1, +1} en labels 0..C-1 (schéma one-vs-all).

    Paramètres :
    ------------
    Y_pm1 : ndarray shape (m, C)
        Matrice avec les classes codées en {-1, +1}.

    Retour :
    --------
    labels : ndarray shape (m,)
        Labels entiers dans [0, C-1].
    """
    Y01 = (Y_pm1 + 1) / 2.0
    return np.argmax(Y01, axis=1).astype(int)

def plot_class_prototypes_csv(
    file_path: str,
    n_classes: int = 4,
    grid_shape: tuple = (5, 5),
    feature_count: int = None,
    save: bool = False,
    title: str = None,
    type: str = None
) -> None:
    """
    Affiche un 'prototype' par classe (moyenne des échantillons de la classe).
    Utile pour visualiser les caractéristiques moyennes dans des données type image.

    Paramètres :
    ------------
    file_path : str
        Chemin du fichier CSV contenant X + Y codé en {-1, +1}.
    n_classes : int
        Nombre de classes.
    grid_shape : tuple(int, int)
        Dimensions (lignes, colonnes) de chaque "image".
    feature_count : int | None
        Nombre de features à utiliser (par défaut = rows * cols).
    save : bool
        True pour sauvegarder l'image.
    title : str
        Titre utilisé pour le nom du fichier si sauvegarde.
    type : str
        Nom du sous-dossier pour la sauvegarde.

    Effet :
    -------
    Affiche les prototypes par classe et sauvegarde si demandé.
    """
    df = pd.read_csv(file_path, header=None).astype(float)
    rows, cols = grid_shape
    if feature_count is None:
        feature_count = rows * cols

    # Séparation des features et des labels
    feat_end = df.shape[1] - n_classes
    X = df.iloc[:, :feat_end].values
    Y_pm1 = df.iloc[:, feat_end:].values
    y = _pm1_to_labels(Y_pm1)

    # Sélection du nombre de features à afficher
    X = X[:, :feature_count]
    if X.min() < 0: X = (X + 1) / 2.0 # Passage en [0, 1] si données en [-1, +1]

    # Création de la figure avec un subplot par classe
    fig, axes = plt.subplots(1, n_classes, figsize=(n_classes*2.4, 2.4))
    if n_classes == 1: axes = [axes]

    for c in range(n_classes):
        mask = (y == c)
        proto = X[mask].mean(axis=0).reshape(rows, cols) if mask.any() else np.zeros((rows, cols))
        ax = axes[c]
        im = ax.imshow(proto, cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"Classe {c} • prototype")
        ax.axis("off")

    plt.suptitle(f"Prototypes par classe — {os.path.basename(file_path)}", y=0.98)
    plt.tight_layout()

    if save:
        os.makedirs(f"../Plot/{type}", exist_ok=True)
        filename = f"../Plot/{type}/{title.replace(' ', '_')}.png"
        plt.savefig(filename)
        print(f"Graphe Emoy sauvegardé sous : {filename}")

    plt.show()

def confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] = None,
    normalize: str = None,
    type: str = "Perceptron_monocouche",
    save: bool = False,
    figsize: tuple = (6, 6),
    cmap: str = "Blues",
    title: str = "aucune titre défini",
    save_path: str = None
) -> np.ndarray:
    """
    Calcule et trace une matrice de confusion sans sklearn.

    Paramètres :
    ------------
    y_true : array-like shape (m,)
        Labels réels.
    y_pred : array-like shape (m,)
        Labels prédits.
    class_names : list[str] | None
        Liste des noms de classes (par défaut "Classe 0", ..., "Classe N").
    normalize : str | None
        - None  : valeurs brutes
        - "true": normalisation par ligne (rappel)
        - "pred": normalisation par colonne (précision)
        - "all" : normalisation par le total global
    type : str
        Type de modèle (utilisé pour sauvegarde si nécessaire).
    save : bool
        Non utilisé ici (prévu pour compatibilité).
    figsize : tuple
        Taille de la figure.
    cmap : str
        Colormap matplotlib.
    title : str
        Titre du graphique.
    save_path : str | None
        Chemin complet pour sauvegarder l'image.

    Retour :
    --------
    cm : ndarray shape (n_classes, n_classes)
        Matrice de confusion calculée.
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    num_classes = int(max(y_true.max(), y_pred.max()) + 1)

    # Définition des noms de classes
    if class_names is None:
        class_names = [f"Classe {i}" for i in range(num_classes)]
    else:
        assert len(class_names) == num_classes, "class_names ne correspond pas au nombre de classes"

    # Calcul de la matrice brute
    cm = np.zeros((num_classes, num_classes), dtype=float)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1.0

    # Normalisation éventuelle
    if normalize == "true":
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums
    elif normalize == "pred":
        col_sums = cm.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        cm = cm / col_sums
    elif normalize == "all":
        total = cm.sum()
        if total > 0:
            cm = cm / total

    # Tracé
    plt.figure(figsize=figsize)
    im = plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # Valeurs dans les cellules
    fmt = ".2f" if normalize else ".0f"
    thresh = cm.max() / 2.0 if cm.size > 0 else 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm[i, j]
            text = f"{val:{fmt}}"
            color = "white" if val > thresh else "black"
            plt.text(j, i, text, ha="center", va="center", color=color)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    plt.show()

    return cm