import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # ou 'Qt5Agg', 'Agg' (pour sauvegarder sans afficher)
import numpy as np
import os


def plot_learning_curve(
    errors: list[float] | np.ndarray,
    type: str,
    title: str = "Courbe d'apprentissage Adaline - 2.11",
    save_plot: bool = True
) -> None:
    """
    Trace la courbe d'apprentissage (erreur quadratique moyenne) d'un modèle ADALINE.

    Paramètres :
    ------------
    errors : list or ndarray
        Liste des erreurs quadratiques moyennes (Emoy) par époque.
    type : str
        Nom du sous-dossier où sauvegarder le graphe (ex : 'adaline', 'perceptron').
    title : str
        Titre du graphe (utilisé aussi pour le nom du fichier).
    save_plot : bool
        True pour sauvegarder l'image dans ../Plot/{type}/.

    Effet :
    -------
    Affiche la courbe et, si save_plot=True, la sauvegarde dans un fichier PNG.
    """
    plt.figure()
    plt.plot(range(1, len(errors) + 1), errors, marker='o')
    plt.xlabel("Époque")
    plt.ylabel("Erreur quadratique moyenne (Emoy)")
    plt.title("Courbe d'apprentissage - ADALINE (Régression)")
    plt.grid(True)

    # Sauvegarde éventuelle du graphe
    if save_plot:
        os.makedirs(f"../Plot/{type}", exist_ok=True)
        filename = f"../Plot/{type}/{title.replace(' ', '_')}.png"
        plt.savefig(filename)
        print(f"Graphe sauvegardé sous : {filename}")

    # Affichage
    plt.show()


def plot_regression_result(
    X: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    type: str,
    title: str = "Perceptron Adaline Table 2.11",
    save_plot: bool = True
) -> None:
    """
    Trace le résultat d'une régression linéaire avec ADALINE :
    - Points bleus : données réelles (X, y)
    - Ligne rouge : prédictions du modèle (X, y_pred)

    Paramètres :
    ------------
    X : ndarray
        Données d'entrée (feature(s)).
    y : ndarray
        Valeurs cibles réelles.
    y_pred : ndarray
        Valeurs prédites par le modèle.
    type : str
        Nom du sous-dossier où sauvegarder le graphe.
    title : str
        Titre du graphe (utilisé aussi pour le nom du fichier).
    save_plot : bool
        True pour sauvegarder l'image dans ../Plot/{type}/.

    Effet :
    -------
    Affiche la régression (points réels + prédiction) et, si save_plot=True,
    la sauvegarde dans un fichier PNG.
    """
    plt.figure()
    plt.scatter(X, y, color='blue', label='Données réelles')
    plt.plot(X, y_pred, color='red', label='Prédiction ADALINE')
    plt.xlabel("x")
    plt.ylabel("y / Prédiction")
    plt.title("Régression linéaire avec ADALINE")
    plt.legend()
    plt.grid(True)

    # Sauvegarde éventuelle du graphe
    if save_plot:
        os.makedirs(f"../Plot/{type}", exist_ok=True)
        filename = f"../Plot/{type}/{title.replace(' ', '_')}.png"
        plt.savefig(filename)
        print(f"Graphe sauvegardé sous : {filename}")

    # Affichage
    plt.show()
