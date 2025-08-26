import matplotlib.pyplot as plt
import numpy as np
from Outils.initialisation import add_x0
import os
from matplotlib import cm


def plot_decision_regions(X, y, weights, type, title="Perceptron monocouche - table_3_1_decision.png", save=False):
    """
    Affiche les régions de décision d'un perceptron monocouche (multi-classes) sur un plan 2D.

    Paramètres :
    ------------
    X : ndarray shape (m, n_features)
        Données d'entrée (features).
        Les deux premières colonnes sont utilisées pour l'affichage.
    y : ndarray shape (m,)
        Labels réels (0..C-1).
    weights : ndarray shape (n_classes, n_features + 1)
        Poids du perceptron (inclut le biais x0).
    type : str
        Nom du sous-dossier pour la sauvegarde si `save=True`.
    title : str
        Titre du graphique.
    save : bool
        True pour sauvegarder le graphique, False sinon.

    Effet :
    -------
    - Affiche les régions de décision colorées pour chaque classe.
    - Trace les points de données avec une couleur par classe.
    - Sauvegarde le graphique si demandé.
    """

    # On ne conserve que les deux premières features pour affichage
    X_plot = X[:, :2]

    # Définition des limites du graphique avec une marge
    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1

    # Création d'une grille de points couvrant la zone
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # On complète les features manquantes par des zéros si X > 2 colonnes
    grid_full = np.c_[grid, np.zeros((grid.shape[0], X.shape[1] - 2))]

    # Ajout du biais (colonne x0 = 1) pour passer au perceptron
    grid_bias = add_x0(grid_full)

    # Calcul des scores pour chaque classe
    Z = np.dot(weights, grid_bias.T)

    # Attribution de la classe = argmax des scores
    Z = np.argmax(Z, axis=0)
    Z = Z.reshape(xx.shape)

    # Création du graphique
    plt.figure()
    cmap = cm.get_cmap("Set3", weights.shape[0])  # palette de couleurs par classe
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)

    # Affichage des points de données avec contour noir
    scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y, edgecolors='k', cmap=cmap)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(title)

    # Sauvegarde éventuelle du graphique
    if save:
        os.makedirs(f"../Plot/{type}", exist_ok=True)
        filename = f"../Plot/{type}/{title.replace(' ', '_')}.png"
        plt.savefig(filename)
        print(f"Graphe Emoy sauvegardé sous : {filename}")

    plt.show()






