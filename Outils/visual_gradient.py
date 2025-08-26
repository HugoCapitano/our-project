import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

def plot_decision_regions_shaded(
        X, y, weights, *,
        X_has_bias=True,          # True si X inclut x0=1 en 1re colonne
        bias_pos="first",         # "first": [b,w1,w2], "last": [w1,w2,b]
        title="BatchGD Decision Boundary - Non-Linear Data (+/-1)",
        save_path=None
):
    """
    Affiche les régions de décision et la frontière pour un classifieur binaire
    (valeurs cibles {-1, +1}) sur deux features.

    Paramètres :
    ------------
    X : ndarray shape (n_samples, n_features) ou (n_samples, n_features+1)
        Données d'entrée. Peut inclure ou non la colonne de biais selon X_has_bias.
    y : ndarray shape (n_samples,)
        Cibles réelles en {-1, +1}.
    weights : ndarray shape (n_features+1,)
        Poids appris du modèle (y compris le biais).
    X_has_bias : bool
        True si X inclut déjà la colonne de biais (x0=1), False sinon.
    bias_pos : str
        Position du biais dans le vecteur de poids : "first" ou "last".
    title : str
        Titre du graphique.
    save_path : str ou None
        Chemin complet pour sauvegarder le graphique, ou None pour juste l'afficher.

    Effet :
    -------
    Affiche les régions de décision colorées, la frontière, et les points
    du jeu de données. Sauvegarde le graphique si save_path est spécifié.
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    w = np.asarray(weights).ravel()

    # --- Ne garder que 2 features pour l'affichage ---
    if X_has_bias:
        # Si x0 est en 1re colonne, on prend les deux features suivantes
        X2 = X[:, 1:3] if X.shape[1] >= 3 else X[:, :2]
    else:
        X2 = X[:, :2]

    if X2.shape[1] != 2:
        raise ValueError(f"Ce plot ne marche qu'avec 2 features visibles. X2.shape={X2.shape}")

    # --- Extraction du biais et des poids w1, w2 ---
    if bias_pos == "first":
        b, w1, w2 = w[0], w[1], w[2]
    elif bias_pos == "last":
        w1, w2, b = w[0], w[1], w[-1]
    else:
        raise ValueError("bias_pos doit être 'first' ou 'last'.")

    # --- Création d'une grille pour représenter les régions ---
    pad = 0.5
    x_min, x_max = X2[:, 0].min() - pad, X2[:, 0].max() + pad
    y_min, y_max = X2[:, 1].min() - pad, X2[:, 1].max() + pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 600),
        np.linspace(y_min, y_max, 600)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # --- Calcul des scores et des classes (-1 ou +1) ---
    score = w1 * grid[:, 0] + w2 * grid[:, 1] + b
    Z = np.where(score >= 0, 1, -1).reshape(xx.shape)

    # --- Palette de couleurs (violet = -1, jaune = +1) ---
    cmap = ListedColormap(["#7F6DBD", "#F0F08A"])

    # --- Affichage des régions colorées ---
    plt.figure(figsize=(8.5, 6.5))
    plt.contourf(xx, yy, Z, levels=[-1, 0, 1], cmap=cmap, alpha=0.9, antialiased=True)

    # --- Affichage des points de données ---
    cls_order = [-1, 1]  # Pour que la légende affiche toujours -1 puis 1
    colors_pts = { -1: "#3f2d7a",  1: "#cdbf29" }
    for lab in cls_order:
        mask = (y == lab)
        plt.scatter(
            X2[mask, 0], X2[mask, 1],
            s=65, c=colors_pts[lab],
            edgecolors="k", linewidths=0.8,
            label=f"{float(lab):.1f}", zorder=3
        )

    # --- Tracé de la frontière de décision (score=0) ---
    xs = np.linspace(x_min, x_max, 600)
    if abs(w2) > 1e-12:
        # Forme y = -(b + w1*x) / w2
        ys = -(b + w1 * xs) / w2
        plt.plot(xs, ys, color="#5A8E3E", linewidth=2.5, zorder=4)
    else:
        # Cas frontière verticale : x = -b / w1
        x_vert = -b / w1 if abs(w1) > 1e-12 else 0.0
        plt.axvline(x_vert, color="#5A8E3E", linewidth=2.5, zorder=4)

    # --- Personnalisation des axes ---
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    xt = np.arange(np.floor(x_min), np.ceil(x_max) + 1e-9, 1)
    yt = np.arange(np.floor(y_min), np.ceil(y_max) + 1e-9, 1)
    plt.xticks(xt)
    plt.yticks(yt)
    plt.grid(True, alpha=0.25)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend(title="Classes", loc="upper right")

    # --- Sauvegarde du graphique ---
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    # --- Affichage ---
    plt.show()
