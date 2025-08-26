from Outils.initialisation import load_data, add_x0, ini_weight
from Outils.visual_gradient import *
from Modèles.perceptron_gradient import perceptron_gradient
import numpy as np
from Outils.visual_perceptron_2D import plot_decision_boundary, plot_emoy_evolution

if __name__ == "__main__":
    # === 1. Chargement des données ===
    # Lecture du fichier CSV "table_2_9.csv"
    # - Pas d'en-tête (has_header=False)
    # - Colonne cible = dernière colonne (y_col=-1)
    X, y = load_data("../Datas/table_2_9.csv", has_header=False, y_col=-1)
    print("load_data: ok")

    # === 1b. Conversion éventuelle des labels ===
    # Si les classes sont {0, 1}, conversion vers {-1, 1} pour le perceptron
    uniq = np.unique(y)
    if set(uniq.tolist()) == {0, 1}:
        y = np.where(y == 0, -1, 1)

    # === 2. Ajout du biais (x0 = 1) ===
    # Ajoute une colonne de 1 à gauche de X pour représenter le biais
    X_bias = add_x0(X)
    print("add_x0: ok")

    # === 3. Initialisation des poids ===
    # Poids initialisés à zéro (taille adaptée à X_bias)
    w_init = ini_weight(X_bias)
    print("init_weight: ok")

    # === 4. Entraînement avec Perceptron + Descente de Gradient ===
    # Appel à perceptron_gradient avec :
    # - learning_rate = 0.001
    # - max_epochs = 700
    # - emoy_threshold = 0.001 (critère d'arrêt si l'erreur est faible)
    # Retourne :
    #   - new_Xg, new_yg : X et y inchangés
    #   - w_grad : poids appris
    #   - emoy_list : liste des erreurs quadratiques moyennes par époque
    print("\n=== Perceptron Gradient ===")
    new_Xg, new_yg, w_grad, emoy_list = perceptron_gradient(
        X, y, X_bias.copy(), w_init.copy(),
        learning_rate=0.001,
        max_epochs=700,
        emoy_threshold=0.001
    )
    print("perceptron_gradient: ok")
    print(f"Poids finaux : {w_grad}")


    # === 5. Visualisation ===
    # a) Tracer la frontière de décision (données 2D)
    plot_decision_boundary(
        X_bias, y, w_grad,
        type="perceptron_gradient",
        X_has_bias=True, bias_pos="first",
        title="Perceptron gradient - table_2.9",
        save_plot=True
    )
    # b) Tracer l'évolution de l'erreur quadratique moyenne (Emoy)
    plot_emoy_evolution(emoy_list,
                        type="perceptron_gradient",
                        title="Perceptron Gradient - table 2.9 - error",
                        save_plot=True)

