import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # ou 'Qt5Agg', 'Agg' (pour sauvegarder sans afficher)
import numpy as np
import os

def plot_decision_boundary(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    X_has_bias: bool = False,
    bias_pos: str = "first",
    type: str = None,
    title: str = "Perceptron – Frontière de décision",
    save_plot: bool = False
) -> None:
    """
    Trace la frontière de décision pour un modèle linéaire (Perceptron, Adaline, etc.) en 2D.

    Paramètres :
    ------------
    X : ndarray
        Données d'entrée (features), éventuellement avec biais.
    y : ndarray
        Labels des échantillons.
    weights : ndarray
        Poids du modèle, incluant éventuellement le biais.
    X_has_bias : bool
        True si la matrice X contient une colonne de biais (x0=1).
    bias_pos : str
        "first" si le biais est en première position dans le vecteur de poids,
        "last" s'il est en dernière position.
    type : str
        Nom du sous-dossier pour sauvegarder le graphique.
    title : str
        Titre du graphique.
    save_plot : bool
        True pour sauvegarder le graphique.

    Effet :
    -------
    Affiche la frontière de décision et les points de données.
    Sauvegarde optionnelle du graphique.
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    w = np.asarray(weights).ravel()

    # Si X inclut le biais, on le retire pour l'affichage 2D
    if X_has_bias and X.shape[1] == 3:
        # Supposition : colonne de biais en première position
        X2 = X[:, 1:3]
    else:
        X2 = X

    if X2.shape[1] != 2:
        raise ValueError(f"Ce graphe ne marche que pour 2 features. X.shape={X2.shape}")

    # Extraire b, w1, w2 selon la position du biais
    if bias_pos == "first":
        if w.size < 3:
            raise ValueError("weights doit contenir au moins [b, w1, w2].")
        b, w1, w2 = w[0], w[1], w[2]
    elif bias_pos == "last":
        if w.size < 3:
            raise ValueError("weights doit contenir au moins [w1, w2, b].")
        w1, w2, b = w[0], w[1], w[-1]
    else:
        raise ValueError("bias_pos doit être 'first' ou 'last'.")

    plt.figure(figsize=(8, 6))

    # Tracer les points de chaque classe
    for label in np.unique(y):
        plt.scatter(X2[y == label, 0], X2[y == label, 1], label=f"Classe {label}")

    # Calcul des points pour la frontière
    x_vals = np.linspace(X2[:, 0].min(), X2[:, 0].max(), 200)

    if w2 != 0:
        # Forme y = -(b + w1*x) / w2
        y_vals = -(b + w1 * x_vals) / w2
        plt.plot(x_vals, y_vals, '--', label='Frontière de décision')
    else:
        # Cas frontière verticale : x = -b/w1
        plt.axvline(-b / w1, linestyle='--', label='Frontière verticale')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Sauvegarde optionnelle
    if save_plot:
        os.makedirs(f"../Plot", exist_ok=True)
        os.makedirs(f"../Plot/{type}", exist_ok=True)
        filename = f"../Plot/{type}/{title.replace(' ', '_')}.png"
        plt.savefig(filename)
        print(f"Graphe sauvegardé sous : {filename}")

    plt.show()

def plot_emoy_evolution(
    emoy_list: list[float] | np.ndarray,
    type: str,
    title: str = "Évolution de l'erreur quadratique moyenne",
    save_plot: bool = False
) -> None:
    """
    Trace l'évolution de l'erreur quadratique moyenne (Emoy) au fil des époques.

    Paramètres :
    ------------
    emoy_list : list or ndarray
        Liste contenant la valeur d'Emoy à chaque époque.
    type : str
        Nom du sous-dossier pour sauvegarder le graphique.
    title : str
        Titre du graphique.
    save_plot : bool
        True pour sauvegarder le graphique.

    Effet :
    -------
    Affiche et, si demandé, sauvegarde la courbe d'évolution d'Emoy.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(emoy_list) + 1), emoy_list, 'o-', label="Emoy")
    plt.xlabel("Itération (Époque)")
    plt.ylabel("Erreur quadratique moyenne (Emoy)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Sauvegarde optionnelle
    if save_plot:
        os.makedirs(f"../Plot/{type}", exist_ok=True)
        filename = f"../Plot/{type}/{title.replace(' ', '_')}.png"
        plt.savefig(filename)
        print(f"Graphe Emoy sauvegardé sous : {filename}")

    plt.show()

