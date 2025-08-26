import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os


def _ensure_dir(path):
    """
    Crée le dossier parent du chemin fourni si nécessaire.
    - path : chemin complet d'un fichier ou dossier
    """
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)

def plot_decision_boundary_1d_fn(
        predict_fn, X, y,
        title="Decision boundary (1D)",
        save_path=None,
        num_points=600,
        xlim=None,          # Limites X (xmin, xmax) ou None
        ylim=(-2, 2),       # Limites Y par défaut
        tick_step=0.5,      # Pas des ticks (x et y)
        jitter_amp=0.08     # Amplitude du bruit vertical
):
    """
        Trace une frontière de décision en 1D avec fond coloré par classe prédite.

        Paramètres :
        ------------
        predict_fn : fonction
            Fonction prenant un array X et retournant la classe prédite.
        X, y : ndarray
            Données (features et labels).
        title : str
            Titre du graphique.
        save_path : str ou None
            Chemin de sauvegarde du graphique.
        """
    x = X[:, 0]
    if xlim is None:
        x_min, x_max = x.min() - 0.5, x.max() + 0.5
    else:
        x_min, x_max = xlim
    xs = np.linspace(x_min, x_max, num_points).reshape(-1, 1)

    # Prédiction sur la grille
    y_grid = predict_fn(xs)

    # Couleurs selon le nombre de classes
    n_classes = int(np.max(y) + 1)
    cmap = cm.get_cmap("Set3", n_classes)

    # Fond coloré par région
    plt.figure(figsize=(9, 4))
    current_class = y_grid[0]
    start_idx = 0
    for i in range(1, len(y_grid) + 1):
        if i == len(y_grid) or y_grid[i] != current_class:
            plt.axvspan(xs[start_idx, 0], xs[i-1, 0],
                        color=cmap(current_class), alpha=0.25, lw=0)
            start_idx = i
            if i < len(y_grid):
                current_class = y_grid[i]

    # Points avec léger décalage vertical
    jitter = (np.random.rand(len(x)) - 0.5) * jitter_amp
    plt.scatter(x, jitter, c=y, edgecolors="k", cmap=cmap, zorder=3)

    # Axes et ticks
    plt.xlim(x_min, x_max)
    plt.ylim(ylim[0], ylim[1])
    plt.xticks(np.arange(x_min, x_max + 1e-9, tick_step))
    plt.yticks(np.arange(ylim[0], ylim[1] + 1e-9, tick_step))

    # Légende
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=cmap(c), markeredgecolor='k',
                          markersize=8, label=f"Classe {c}")
               for c in range(n_classes)]
    plt.legend(handles=handles, title="Classes", loc="upper right")

    plt.xlabel("x"); plt.ylabel("y"); plt.title(title)
    _ensure_dir(save_path)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

def plot_decision_regions_2d_fn(
        predict_fn, X, y,
        title="Decision regions (2D)",
        save_path=None,
        xlim=None, ylim=None, tick_step=None
):
    """
    Trace les régions de décision pour un jeu de données 2D.
    Si X a plus de 2 features, les autres sont fixées à leur moyenne.
    """
    if X.shape[1] == 1:
        return plot_decision_boundary_1d_fn(
            predict_fn, X, y, title=title.replace("2D", "1D"),
            save_path=save_path
        )

    X2 = X[:, :2]
    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    # Compléter si > 2 features
    grid2 = np.c_[xx.ravel(), yy.ravel()]
    if X.shape[1] > 2:
        others_mean = X[:, 2:].mean(axis=0, keepdims=True)
        others = np.repeat(others_mean, grid2.shape[0], axis=0)
        grid = np.c_[grid2, others]
    else:
        grid = grid2

    Z = predict_fn(grid).reshape(xx.shape)

    # Affichage
    plt.figure()
    cmap = cm.get_cmap("Set3", int(np.max(y) + 1))
    plt.contourf(xx, yy, Z, alpha=0.35, cmap=cmap)
    sc = plt.scatter(X2[:, 0], X2[:, 1], c=y, edgecolors="k", cmap=cmap)
    plt.legend(*sc.legend_elements(), title="Classes")
    plt.title(title)

    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    if tick_step is not None:
        if xlim is None:
            xlim = plt.xlim()
        if ylim is None:
            ylim = plt.ylim()
        plt.xticks(np.arange(xlim[0], xlim[1] + 1e-9, tick_step))
        plt.yticks(np.arange(ylim[0], ylim[1] + 1e-9, tick_step))

    _ensure_dir(save_path)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


# === Courbes d'apprentissage ===

def plot_learning_curve(losses, title="MLP – Courbe d'apprentissage", save_path=None):
    """
    Trace la perte (loss) par époque.
    - losses : liste/array de valeurs de perte (une par epoch)
    """
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(losses)+1), losses, marker="o")
    plt.xlabel("Époque")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    _ensure_dir(save_path)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_accuracy_curve(accs, title="MLP – Courbe d'accuracy", save_path=None):
    """
    Trace la courbe d'accuracy par époque.
    """
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(accs)+1), accs, marker="o")
    plt.xlabel("Époque")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    _ensure_dir(save_path)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

def plot_regression_4_17(
        predict_fn, X, y,
        feature_index=0, n_points=600,
        xlim=None, ylim=None, tick_step=None,
        title="MLP Non-Linear Regression Fit",
        save_path=None
):
    """
    Trace la courbe de régression (prédictions vs données réelles)
    en fonction d'une feature sélectionnée.
    """

    x = X[:, feature_index]
    if xlim is None:
        x_min, x_max = x.min() - 0.1, x.max() + 0.1
    else:
        x_min, x_max = xlim

    xs = np.linspace(x_min, x_max, n_points)

    if X.shape[1] > 1:
        others_mean = X.mean(axis=0)
        grid = np.tile(others_mean, (n_points, 1))
        grid[:, feature_index] = xs
    else:
        grid = xs.reshape(-1, 1)

    y_line = predict_fn(grid).reshape(-1)

    plt.figure(figsize=(9, 5))
    plt.scatter(x, y, s=28, label="Données", alpha=0.9)
    plt.plot(xs, y_line, linewidth=2.5, label="Régression", zorder=3)

    if xlim is not None: plt.xlim(*xlim)
    if ylim is not None: plt.ylim(*ylim)
    if tick_step is not None:
        if xlim is None: xlim = plt.xlim()
        if ylim is None: ylim = plt.ylim()
        plt.xticks(np.arange(xlim[0], xlim[1] + 1e-9, tick_step))
        plt.yticks(np.arange(ylim[0], ylim[1] + 1e-9, tick_step))

    plt.grid(True, alpha=0.3)
    plt.xlabel("x"); plt.ylabel("y")
    plt.title(title)
    plt.legend()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_residuals(
        y_true, y_pred, bins=20,
        title="Residuals (y - ŷ)", save_path=None
):
    """
    Trace l'histogramme des résidus (y - prédiction).
    """
    r = (y_true.reshape(-1) - y_pred.reshape(-1))
    plt.figure(figsize=(7,4))
    plt.hist(r, bins=bins, edgecolor="k", alpha=0.8)
    plt.title(title); plt.xlabel("Résidu"); plt.ylabel("Fréquence")
    plt.grid(True, alpha=0.25)
    _ensure_dir(save_path)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

# === Matrice de confusion (classification) ===
def plot_confusion_matrix(
        y_true, y_pred,
        labels=None,
        normalize=False,
        title="Matrice de confusion",
        save_path=None
):
    """
    Trace une matrice de confusion avec option de normalisation.
    """

    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    cmatrix = np.zeros((n, n), dtype=float)

    # Remplissage de la matrice
    for t, p in zip(y_true, y_pred):
        cmatrix[label_to_idx[t], label_to_idx[p]] += 1

    # Normalisation éventuelle
    if normalize:
        row_sums = cmatrix.sum(axis=1, keepdims=True) + 1e-12
        cmatrix = cmatrix / row_sums

    plt.figure(figsize=(6, 5))
    im = plt.imshow(cmatrix, interpolation='nearest', cmap=cm.get_cmap("Blues"))
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.arange(n)
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    thresh = cmatrix.max() / 2.0 if cmatrix.size else 0.5
    for i in range(n):
        for j in range(n):
            txt = f"{cmatrix[i, j]:.2f}" if normalize else f"{int(cmatrix[i, j])}"
            plt.text(j, i, txt,
                     ha="center", va="center",
                     color="white" if cmatrix[i, j] > thresh else "black",
                     fontsize=10)

    plt.ylabel("Vrai")
    plt.xlabel("Prédit")
    plt.tight_layout()

    _ensure_dir(save_path)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
