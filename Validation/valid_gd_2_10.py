from Outils.initialisation import load_data, add_x0, ini_weight
from Outils.visual_gradient import *
from Modèles.perceptron_gradient import perceptron_gradient
import numpy as np

from Outils.visual_perceptron_2D import plot_decision_boundary, plot_emoy_evolution

if __name__ == "__main__":
    # === 1. Chargement des données ===
    # Lecture du CSV "table_2_10.csv" sans en-tête, avec la dernière colonne comme label (y)
    X, y = load_data("../Datas/table_2_10.csv", has_header=False, y_col=-1)
    print("load_data: ok")

    # Vérification du codage des classes :
    # Si les labels sont {0,1}, on les convertit en {-1,1} pour correspondre au perceptron
    uniq = np.unique(y)
    if set(uniq.tolist()) == {0, 1}:
        y = np.where(y == 0, -1, 1)

    # === 2. Ajout du biais ===
    # Ajoute une colonne x0=1 à gauche de X (utile pour calculer le terme de biais)
    X_bias = add_x0(X)
    print("add_x0: ok")

    # === 3. Initialisation des poids ===
    # Crée un vecteur de poids initialisé à 0, de taille = nombre de features + biais
    w_init = ini_weight(X_bias)
    print("init_weight: ok")

    # ---------- Perceptron avec descente de gradient ----------
    print("\n=== Perceptron Gradient ===")
    new_Xg, new_yg, w_grad, emoy_list = perceptron_gradient(
        X, y, X_bias.copy(), w_init.copy(),
        learning_rate=0.001,
        max_epochs=500,
        emoy_threshold=0.001
    )
    print("perceptron_gradient: ok")
    print(f"Poids finaux : {w_grad}")

    # === 5. Visualisation : frontière de décision ===
    plot_decision_boundary(
        X_bias, y, w_grad,
        type="perceptron_gradient",
        X_has_bias=True, bias_pos="first",
        title="Perceptron gradient - table_2.10",
        save_plot=True
    )
    # === 6. Visualisation : évolution de l'erreur quadratique moyenne ===
    plot_emoy_evolution(emoy_list, type="perceptron_gradient",title="Perceptron Gradient - table 2.10 - error", save_plot=True)

    